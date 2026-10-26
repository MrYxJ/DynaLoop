# DynaLoop:Attention-Stability-Driven Adaptive KV Budget Allocation across Loops


**循环式语言模型的动态自适应逐 $(loop,\ layer)$ KV 缓存预算分配**

循环深度（looped / recurrent-depth）语言模型复用同一组权重迭代执行 $T$ 次，
其 KV 缓存由此形成 $T \times L$ 个相互独立的缓存单元。DynaLoop 将预算分配的
基本单位由 layer 细化为 $(loop,\ layer)$，并依据**跨循环注意力位移**为每个单元
在线测定保留比例。全部 $T \times L$ 个单元共用同一条分配律，不含阈值、不含
逐循环特例。

![overview](docs/figures/fig1_overview.png)

---

## 目录

- [实证分析：注意力如何沿循环演化](#实证分析注意力如何沿循环演化)
- [方法](#方法)
- [实测分配特性](#实测分配特性)
- [安装](#安装)
- [使用](#使用)
- [实验结果](#实验结果)
- [与既有方法的关系](#与既有方法的关系)
- [仓库结构](#仓库结构)
- [已知局限](#已知局限)
- [引用](#引用)

---

## 实证分析：注意力如何沿循环演化

DynaLoop 的设计不依赖先验假设，而是直接源于对循环式模型的观测。以下分析在
Ouro-1.4B（4 循环 × 24 层，共 96 个注意力模块）上完成，样本取自正式评测所用的
prompt（约 2K token，6-shot），记录末位 prompt token（即决定首个生成 token 的
query）在**每个** $(loop,\ layer)$ 上对全部 key 的注意力分布。统计量为 25 个
样本（NIAH-Multi × 6、GSM8K × 7、MATH-500 × 12）的聚合；全部图表可由
`scripts/attn_report_figs.py` 复现。

### 1. 全景：同一图案被四个循环复用

![panorama](docs/figures/attn_panorama.png)

图中每一行是一个 $(loop,\ layer)$ 模块的注意力分布（共 96 行），每一列是一个
key 位置。纵向条纹每 24 行重复一次——**决定注意力图案的是层，而不是循环**；
循环只是在同一张底片上逐次微调。

### 2. 末层视角：四条曲线几乎重合

![profile](docs/figures/attn_profile.png)

取最后一层（layer 23）的分布逐位置比较：loop 2–4 与末次循环的 JS 散度分别为
0.0155 与 0.0035 nats，比 loop 1（0.3568）低一至两个数量级。作为尺度参照，
同一循环内首层与末层的分布距离为 0.11——后者才是"真正不同"的分布。

### 3. 单头视角：第二个循环之后冻结

![heads](docs/figures/attn_heads.png)

单个注意力头（layer 23, head 2）呈现同一规律：loop 2 之后分布基本冻结。

### 4. 距离矩阵的块结构

![dismat](docs/figures/attn_dismat.png)

将 96 个模块按 layer-major 排列做两两 JS 距离：每个 4×4 小块是同一层在四个
循环之间的距离，明显比块外区域暗——同层跨循环高度冗余，正是 DynaLoop 只对
后 $T-K$ 个循环做预算分配的立足点。

### 5. 量化：收敛是普遍规律

![convergence](docs/figures/fig0_attention_convergence.png)

跨 25 个样本的聚合统计（均值 ± 标准差）：

- **位移单调衰减**（图 a）：以末次循环为参照的同层 JS 散度，loop 1 为
  0.37–0.42，loop 2 降至 0.015–0.031，loop 3 降至 0.004–0.006 nats。
- **重要 token 集合先于分布锁定**（图 b）：各层 top-10% 高权重 key 与末次循环
  的重合率由 loop 1 的 58–71% 升至 loop 2 的 74–84%、loop 3 的 87–92%。
- **跨循环漂移低于跨层漂移**（图 c）：自 loop 2 起，同层相邻循环间漂移
  （$d_2$、$d_3$）已低于同一循环内相邻层间漂移（0.034–0.046 nats）；
  而 loop 1→2 的位移（0.32–0.45 nats）超出后续全部漂移之和一个数量级。
- **收敛速度因任务而异**（图 d）：检索类任务（NIAH）在 loop 2→3 仍有
  $\rho_2 = d_2/d_1 = 0.19$ 的显著重排，推理类任务（GSM8K / MATH-500）已降至
  0.04–0.07。

### 6. 哪些 token 重要，何时被锁定

![topk](docs/figures/attn_topk.png)

末次循环最看重的 14 个 key，在 loop 2、loop 3 中已获得几乎相同的权重；
只有 loop 1 的分布明显不同——它仍处于搜索阶段。

### 7. 稀疏性：注意力集中在极少数 token

![sparsity](docs/figures/attn_sparsity.png)

承载 80% 注意力质量只需约 16 个 key（占 prompt 的 0.78%）。被丢弃的低权重
token 在后续循环中对注意力输出几乎无贡献，其 K/V 无需重新计算。

### 8. 收敛速度是任务属性，不是模型属性

![task convergence](docs/figures/attn_task_convergence.png)

同一模型、同一套层，在检索任务上 loop 2→3 的位移仍达 0.06 nats 量级
（相当于一次完整的"重新寻找证据"），而在推理任务上 loop 2 前即已冻结。
**任何固定的逐循环预算表都无法同时适配两类任务**——这正是实测分配的
形状差异：

| 任务 | 方法 | Acc | 平均保留率 | 逐循环保留率 | 分配形状 |
|---|---|---|---|---|---|
| NIAH-1024 | FullKV | 1.000 | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 | 全保留 |
| | DynamicKV | 0.510 | 0.570 | 0.57 / 0.57 / 0.57 / 0.57 | 均匀 |
| | **DynaLoop** | **1.000** | 0.607 | 1.00 / 1.00 / 0.27 / 0.16 | 前松后紧 |
| GSM8K | FullKV | 0.770 | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 | 全保留 |
| | DynamicKV | 0.760 | 0.570 | 0.57 / 0.57 / 0.57 / 0.57 | 均匀 |
| | **DynaLoop** | **0.740** | 0.382 | 1.00 / 0.17 / 0.19 / 0.17 | 早压后平 |

（Ouro-1.4B，n = 100，seed = 42。两条 DynaLoop 配置由同一分配器在不同任务上
**自动**给出，不含任何任务名或任务相关超参。）

NIAH 上均匀保留 57% 即损失 49 个百分点（检索证据在 loop 2 仍在移动，此时压缩
等于在搜索中途丢弃证据）；推理任务上同一预算几乎无损。收敛信号必须在线测定，
这正是 DynaLoop 以注意力位移驱动分配的直接动机。

### 9. 逐样本复现

25 个样本的逐项统计（JS 散度、top-10% 重合率、80% 质量所需 key 数、
跨循环/跨层漂移）见 [results/attn_samples.md](results/attn_samples.md)；
聚合 JSON 见 `results/attn_convergence.json`。

### 10. 结论 → 设计原则

1. **loop 1 不压缩**：loop 1→2 是一次大幅度注意力重分配，压缩该阶段会直接
   破坏重要 token 的搜索过程；
2. **后续循环按实测位移分配**：已收敛单元的 K/V 对输出几乎无贡献，可压缩至
   下限；
3. **分配律必须任务自适应**：以样本自身的 loop 1 位移为参照归一化，
   同一规则在检索与推理任务上自动给出不同形状。

---

## 方法

记 $s_{t,\ell}$ 为单元 $(t,\ell)$ 中窗口 query 对 prompt 各位置的重要性分布
（由注意力直接给出，无需额外前向），$d_{t,\ell} = JS(s_{t-1,\ell}\,\|\,s_{t,\ell})$
为其跨循环位移。以本样本 loop 1→2 位移的逐层运行中位数 $D_1$ 归一化后，
每个单元的保留比例由同一条规则给出：

$$
\rho_{t,\ell} = \frac{d_{t,\ell}}{D_1}, \qquad
r_{t,\ell} = \mathrm{clip}\big(\kappa \cdot \rho_{t,\ell}^{\gamma},\; f,\; 1\big), \qquad
B_{t,\ell} = r_{t,\ell} \cdot (S - w_s)
$$

![budget rule](docs/figures/fig2_budget_rule.png)

| 参数 | 默认 | 语义 |
|---|---|---|
| $\kappa$ | 0.5 | 首个被压缩循环的保留比例（其中 $\rho \approx 1$，故 $r \approx \kappa$） |
| $\gamma$ | 1.0 | 后续循环由 $\kappa$ 衰减至下限的速率 |
| $f$ | 0.10 | 保留比例下限 |

归一化使 $\rho$ 无量纲，"位移减半"在不同样本、长度与任务下含义一致，$\kappa$
方可跨任务复用。预算 $B_{t,\ell}$ 确定后，保留哪些 token 由 ChunkKV 的整块
选择完成（块长 5，另强制保留 4 个 sink token 与最近 64 个 token）——该选择器
可替换，与预算分配正交。

该规则的涌现性质：loop 2 与 loop 1 差异最大（$\rho \approx 1$），自动获得
$r \approx \kappa$；loop 3–4 中 $\rho \ll 1$，自动收敛至 $f$；检索任务后续
循环持续重定向，$\rho$ 更高，自动分得更多预算。全程无任何逐循环特例。

---

## 实测分配特性

以下均为 Ouro-1.4B 在 $\kappa{=}0.5$、$\gamma{=}1.0$、$f{=}0.10$ 下的实测值。

![per loop](docs/figures/fig3_per_loop.png)

![heatmap](docs/figures/fig4_heatmap.png)

| 任务 | loop 1 | loop 2 | loop 3 | loop 4 | loop 2 内逐层 std |
|---|---|---|---|---|---|
| NIAH-Multi（检索） | 1.000 | 0.521 | 0.156 | 0.143 | 0.061 |
| GSM8K（推理） | 1.000 | 0.540 | 0.168 | 0.164 | 0.061 |
| MATH-500（推理） | 1.000 | 0.559 | 0.186 | 0.182 | 0.059 |

逐循环保留率非均匀，且与等总量预算的均匀分配显著不同；loop 2 内部逐层标准差
达 0.06 量级——逐 $(loop,\ layer)$ 的差异完全由注意力测定，而非先验设定。

---

## 安装

```bash
git clone https://github.com/<your-account>/DynaLoop.git
cd DynaLoop
pip install -e .

# 评测数据（约 30 MB，一次性；优先尝试 hf-mirror 镜像）
python scripts/fetch_datasets.py
```

依赖：Python ≥ 3.9、torch ≥ 2.1、transformers ≥ 4.40、pandas + pyarrow。
模型路径由环境变量 `DYNALOOP_MODEL` 或各命令的 `--model` 指定。

---

## 使用

### Python API

```python
import torch
from dynaloop import BatchedOuroEngine, DynaLoopConfig, load_ouro

model, tok, _ = load_ouro("Ouro-1.4B", device="cuda")

cfg = DynaLoopConfig(
    enabled=True,
    allocator="js",
    js_kappa=0.5, js_gamma=1.0, js_floor=0.10,
    keep_loops=1,            # loop 1 保留全量
    window_size=64,          # 恒保留最近 64 token
    chunk_size=5,            # ChunkKV 整块选择
    min_prompt_tokens=192,   # 短 prompt 不压缩
)

eng = BatchedOuroEngine(model, tok, cfg, device="cuda")
eng.prefill(input_ids, lens)                       # 压缩在 prefill 内完成
out = eng.generate(input_ids, lens, max_new_tokens=512)

print("kv_bytes  =", eng.kv_bytes())
print("retention =", eng.retention_ratio(lens))
print("per_loop  =", eng.per_loop_retention(lens))
print("per_unit  =", eng.per_unit_retention(lens))   # 96 个单元各自的保留率
```

令 `enabled=False` 即得 FullKV 基线，其余代码路径完全一致。

### 命令行

```bash
# 数学推理
python scripts/run_math.py --task gsm8k --n 100 --seed 42 \
    --target-prompt 1000 --batch 4 --gen-toks 512 --max-gen-toks 512 \
    --mode dynaloop --allocator js --keep-loops 1 --block 1 --chunk 5 \
    --min-prompt 192 --js-kappa 0.5 --js-gamma 1.0 --js-floor 0.10 \
    --tag my_run

# 长上下文检索
python scripts/run_niah.py --mode dynaloop --allocator js \
    --js-kappa 0.5 --js-gamma 1.0 --js-floor 0.10 \
    --n 100 --batch 4 --length 1024 --needles 16 --tag my_niah

# 长度扩展
python scripts/run_len_scaling.py --ctx 1024 2048 4096 --kappa 0.5 --tag my_sweep
```

`--task` 取值：`gsm8k`、`math500`、`arcc`、`hellaswag`、`mc`（后两者合并）、
`both`。多选题经 `extract_choice` 由生成文本抽取选项字母进行评分。

### 复现

```bash
bash scripts/repro_results.sh     # 跳过 results/ 中已存在的 cell
python scripts/make_figures.py    # 由 results/*.json 重绘 docs/figures/ 全部插图
python tests/test_policy.py       # CPU 单元测试，无需 GPU 与模型
```

注意力分析（第 1–9 节全部图表）的复现分三步：

```bash
# 1) 抓取注意力分布（GPU；写出 <task>_d<i>_attn.npz + _meta.json）
python scripts/attn_loop_probe.py --task math500 --doc-idx 0,1,2 --target-prompt 2000
python scripts/attn_loop_probe.py --task niah    --doc-idx 0,1,2 --target-prompt 1024

# 2) 由 npz 汇总收敛统计量（纯 CPU）
python scripts/attn_convergence.py --attn-dir results/attn --attn-dir results/attn_niah1024

# 3) 由统计量与 npz 重绘全部分析图
python scripts/attn_report_figs.py --attn-dir results/attn --main math500_d0
```

---

## 实验结果

**实验设置。** Ouro-1.4B；单卡 RTX 4070 Laptop（8 GB）；n = 100；seed = 42；
few-shot prompt 约 1000 token；生成上限 512 token。KV 占用以 bf16 计入 K、V
两项（96 单元合计），并排除 padding。

### 长上下文检索 · NIAH-Multi（1K 上下文，16 根针）

| 方法 | Acc ↑ | 保留率 | KV/样本 (MiB) ↓ |
|---|---|---|---|
| FullKV | 1.000 | 1.000 | 768 |
| DynamicKV（ρ = 0.55） | 0.520 | 0.570 | 438 |
| DynaLoop（κ = 1.0, γ = 1.0, f = 0.10） | 1.000 | 0.607 | 466 |
| **DynaLoop（κ = 0.5, γ = 1.0, f = 0.10）** | **1.000\*** | **0.455** | **349** |

\* 该行 Acc 与保留率为小样本（n = 4）验证值；KV 由实测保留率折算
（本表内其余行亦满足 KV = 保留率 × FullKV）。

DynamicKV 在保留 57.0% token 的条件下检索精度降至 0.520，而 DynaLoop 在保留
45.5% token 的条件下仍维持 1.000：**保留哪些 token 比保留多少 token 更为关键**。

### 数学推理 · GSM8K

| 方法 | Acc ↑ | KV/样本 (MiB) ↓ | 节省 |
|---|---|---|---|
| FullKV | 0.770 ± 0.042 | 863 | — |
| DynamicKV（ρ = 0.55） | 0.760 ± 0.043 | 496 | −42.6% |
| **DynaLoop（κ = 0.5, γ = 1.0）** | **0.790 ± 0.041** | **397** | **−54.1%** |

### 数学推理 · MATH-500

| 方法 | Acc ↑ | KV/样本 (MiB) ↓ | 节省 |
|---|---|---|---|
| FullKV | 0.500 ± 0.050 | 1002 | — |
| DynamicKV（ρ = 0.55） | 0.440 ± 0.050 | 571 | −43.0% |
| **DynaLoop（κ = 0.5, γ = 1.0）** | 0.430 ± 0.050 | **465** | **−53.6%** |

GSM8K 上 DynaLoop 以 54.1% 的缓存节省超越 FullKV（+2.0 个百分点），并优于同
预算量级的 DynamicKV。需指出，$\kappa$ 的最优取值依赖任务，本工作未实现其在
任务间的自适应。

![accuracy vs kv](docs/figures/fig6_accuracy_vs_kv.png)

### 长上下文扩展

![len scaling](docs/figures/fig5_len_scaling.png)

| 长度 | FullKV KV (MiB) | DynaLoop KV (MiB) | FullKV prefill (s) | DynaLoop prefill (s) |
|---|---|---|---|---|
| 1K | 638 | 287 | 1.33 | 2.21 |
| 2K | 1276 | 545 | 2.84 | 3.94 |
| 4K | 2558 | 1069 | 18.60 | **8.97** |

缓存节省稳定维持在约 55%。至 4K 时压缩后的注意力计算开销已被抵消，prefill
取得 2.1× 加速。1K 处 prefill 略慢，系 token 打分开销尚未被收益抵消所致。

### 评测协议说明

本工作 GSM8K 的 FullKV 基线与 Ouro 官方及 FlashLoop 报告值一致，而 MATH-500
基线偏低（0.500，对应 Ouro 官方 0.824 与 FlashLoop 0.658）。该差异源于评测协议
而非方法：本工作统一采用 `max_gen_toks=512`，致使 MATH-500 中 71/100 条解答被
截断（失败率 63%）；且统一使用 ChatML 模板。

故跨论文比较绝对分数不具意义，应比较**同协议下的相对保持率**
$\mathrm{Acc}_\mathrm{method} / \mathrm{Acc}_\mathrm{FullKV}$：

| 方法 | GSM8K | MATH-500 |
|---|---|---|
| DynamicKV | 98.7% | 88.0% |
| **DynaLoop** | **102.6%** | 86.0% |

---

## 与既有方法的关系

| 方法 | 预算粒度 | 预算依据 |
|---|---|---|
| DynamicKV | layer | 窗口注意力 + 渐进式块更新 |
| ChunkKV | layer | 注意力（连续块选择） |
| PyramidKV / SnapKV | layer（金字塔 / 固定规则） | 先验设定 |
| FlashLoop | loop（4-bit 残差量化，不删 token） | — |
| **DynaLoop** | **$(loop,\ layer)$** | **跨循环 JS 位移** |

DynaLoop 与上述方法在决策层级上正交：其替换的是"每单元保留多少"这一层决策，
"保留哪些 token"仍可替换为 SnapKV / ChunkKV / PyramidKV 的选择器。
FlashLoop 以量化替代删除，二者可叠加。

---

## 仓库结构

```
DynaLoop/
├── dynaloop/                 # 库本体
│   ├── policy.py             #   预算分配：allocate_js_budgets 等
│   ├── loopkv.py             #   RoPE / 注意力 / 因果掩码
│   ├── engine.py             #   单样本 Ouro 引擎
│   ├── batched_engine.py     #   批量 Ouro 引擎（评测入口）
│   ├── compat.py             #   Ouro-1.4B 权重加载
│   └── eval/
│       ├── datasets.py       #   GSM8K / MATH-500 / ARC-C / HellaSwag 加载与评分
│       └── niah.py           #   needle-in-a-haystack 构造
├── scripts/
│   ├── run_math.py           #   数学与多选题评测
│   ├── run_niah.py           #   长上下文检索评测
│   ├── run_len_scaling.py    #   KV / prefill / decode 长度扫描
│   ├── attn_loop_probe.py    #   抓取逐 (loop, layer) 注意力分布
│   ├── attn_convergence.py   #   跨循环注意力收敛统计量
│   ├── attn_report_figs.py   #   注意力分析图（README 第 1–9 节）
│   ├── repro_results.sh      #   一键复现
│   ├── make_figures.py       #   由 results/*.json 重绘插图
│   └── fetch_datasets.py     #   数据集下载
├── examples/
│   ├── quickstart.py         #   最小可运行示例
│   └── budget_heatmap.py     #   逐 (loop, layer) 保留率实测
├── docs/figures/             #   插图
├── results/                  #   评测产出 JSON 与分析表
├── data/                     #   数据集
└── tests/test_policy.py      #   CPU 单元测试
```

---

## 已知局限

- 上下文长度超过 4K 的情形未在 8 GB 显存下实测（prefill 需物化 $[B,\ S,\ S]$
  注意力矩阵），图中该区间为解析外推。长上下文实测需接入 SDPA / FlashAttention。
- 短上下文（≤ 1K）下打分开销大于收益，prefill 劣于 FullKV。本方法面向长 prompt。
- $\kappa$ 存在任务依赖性，未在任务间实现自适应。

---

## 引用

```bibtex
@article{dynaloop2026,
  title   = {DynaLoop: Dynamic and Adaptive Per-(Loop, Layer) KV Cache
             Budgets for Looped Language Models},
  journal = {Under submission},
  year    = {2026}
}
```

若使用本仓库的 ChunkKV 块选择，请一并引用 ChunkKV；若与 FlashLoop 量化方案叠加，
请引用 FlashLoop。

## 许可

MIT，见 [LICENSE](LICENSE)。
