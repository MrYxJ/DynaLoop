"""Local GSM8K / MATH-500 / ARC-Challenge / HellaSwag loading and grading.

All datasets are read from ``<repo>/data`` (downloaded once by
``scripts/fetch_datasets.py``, or overridden with the ``DYNALOOP_DATA_DIR``
environment variable) so evaluation never depends on the network.

Grading follows the usual practice for these two benchmarks:

* **GSM8K** -- the reference answer ends with ``#### <number>``; the model's
  answer is the last number in its generation, compared numerically.
* **MATH-500** -- the reference is a LaTeX string; the model's answer is the
  content of the last ``\\boxed{...}`` (falling back to the last number).
  Comparison is string equality after LaTeX normalisation, then symbolic
  equality via ``sympy`` when both sides parse.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))          # <repo>/dynaloop/eval -> <repo>
DATA_DIR = os.environ.get("DYNALOOP_DATA_DIR") or os.path.join(PKG_ROOT, "data")
if not os.path.isdir(DATA_DIR):                            # pip-installed outside a checkout
    DATA_DIR = os.path.join(os.getcwd(), "data")

MATH500_PATH = os.path.join(DATA_DIR, "math500", "test.jsonl")
GSM8K_PATH = os.path.join(DATA_DIR, "gsm8k", "test.parquet")
# multiple-choice benchmarks (FlashLoop §5.1 also reports ARC-C / HellaSwag)
ARC_C_PATH = os.path.join(DATA_DIR, "allenai_ai2_arc_parquet_ARC-Challenge_test_0.parquet")
ARC_C_TRAIN_PATH = os.path.join(DATA_DIR, "allenai_ai2_arc_parquet_ARC-Challenge_train_0.parquet")
HELLASWAG_PATH = os.path.join(DATA_DIR, "hs_val.parquet")
HELLASWAG_TRAIN_PATH = os.path.join(DATA_DIR, "hs_train.parquet")

LETTERS = "ABCDEFGH"


# ---------------------------------------------------------------------- #
# loading                                                                 #
# ---------------------------------------------------------------------- #

def load_math500(path: str = MATH500_PATH) -> List[dict]:
    """``[{problem, solution, answer, level, subject, unique_id}, ...]``."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsm8k(path: str = GSM8K_PATH) -> List[dict]:
    """``[{question, answer}, ...]`` with the gold final number in ``answer``."""
    df = pd.read_parquet(path)
    out = []
    for _, r in df.iterrows():
        raw = str(r["answer"])
        m = re.search(r"####\s*(-?[\d,\.]+)", raw)
        gold = m.group(1).replace(",", "") if m else None
        out.append({"question": str(r["question"]),
                    "solution": raw,
                    "answer": gold})
    return out


_LETTERS = "ABCDEFGH"


def load_arc_challenge(path: str = ARC_C_PATH) -> List[dict]:
    """``[{question, choices, answer}, ...]`` with answers as ``A``-``H``."""
    df = pd.read_parquet(path)
    out = []
    for _, r in df.iterrows():
        ch = r["choices"]
        labels = [str(x) for x in list(ch["label"])]
        texts = [str(x) for x in list(ch["text"])]
        key = str(r["answerKey"]).strip()
        if key not in labels:
            continue                       # unusable row
        out.append({"question": str(r["question"]).strip(),
                    "choices": [t.strip() for t in texts],
                    "answer": _LETTERS[labels.index(key)]})
    return out


def load_hellaswag(path: str = HELLASWAG_PATH) -> List[dict]:
    """``[{ctx, endings, answer}, ...]`` -- answers as ``A``-``D``."""
    df = pd.read_parquet(path)
    out = []
    for _, r in df.iterrows():
        ctx = str(r.get("ctx") or r.get("ctx_a") or "").strip()
        endings = [str(e).strip() for e in list(r["endings"])]
        try:
            gold = int(r["label"])
        except (TypeError, ValueError):
            continue
        if gold < 0 or gold >= len(endings):
            continue
        # preprocessing after lm-eval's hellaswag config
        ctx = ctx.replace("[header]", "").replace("[title]", "").strip()
        endings = [e.replace("[header]", "").replace("[title]", "").strip()
                   for e in endings]
        out.append({"ctx": ctx, "endings": endings,
                    "answer": _LETTERS[gold]})
    return out


def _mc_pool(rows, exclude, limit):
    """Render multiple-choice rows as ``[{question, target}, ...]`` exemplars."""
    ex = set(exclude)
    return [r for i, r in enumerate(rows) if i not in ex][:limit]


# ---------------------------------------------------------------------- #
# normalisation                                                           #
# ---------------------------------------------------------------------- #

_REMOVE = [
    ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
    ("\\ ", ""), ("\\%", ""), ("%", ""), ("$", ""), ("\\$", ""),
    ("^{\\circ}", ""), ("^\\circ", ""), ("\\circ", ""),
    ("\\dots", ""), ("\\ldots", ""),
    ("\\text{ }", " "), ("\\mbox{ }", " "),
]


def _rm_latex_cmd(s: str, cmd: str, keep_braces: bool) -> str:
    """Remove ``\\cmd{...}`` (keeping the content if ``keep_braces`` is False)."""
    pat = re.compile(r"\\" + cmd + r"\s*\{([^{}]*)\}")
    repl = r"\1" if keep_braces else ""
    return pat.sub(repl, s)


def normalize_answer(s: Optional[str]) -> str:
    """Aggressive but standard LaTeX/number normalisation for answer matching."""
    if s is None:
        return ""
    s = str(s).strip()
    s = s.split("\n")[0].strip()          # only the first line can be an answer
    for a, b in _REMOVE:
        s = s.replace(a, b)

    # unwrap \text{...} / \mbox{...}
    for _ in range(3):
        s = _rm_latex_cmd(s, "text", True)
        s = _rm_latex_cmd(s, "mbox", True)
        s = _rm_latex_cmd(s, "mathrm", True)

    # \frac{a}{b} -> (a)/(b)
    for _ in range(4):
        s = re.sub(r"\\d?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    # \sqrt[n]{x} -> (x)**(1/n) ; \sqrt{x} -> sqrt(x)
    s = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", s)

    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi").replace("\\theta", "theta")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("\\", "")

    # numbers
    s = re.sub(r"(\d),(\d\d\d)", r"\1\2", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".").strip()
    if s.endswith("\\"):
        s = s[:-1]
    # 0.50 == 0.5
    m = re.fullmatch(r"(-?\d+)\.(\d*?)0+", s)
    if m:
        s = f"{m.group(1)}.{m.group(2)}" if m.group(2) else m.group(1)
    return s


def _to_sympy(s: str):
    try:
        import sympy
        expr = sympy.sympify(s, rational=True)
        return expr
    except Exception:
        return None


def is_equiv(pred: Optional[str], gold: Optional[str]) -> bool:
    """String equality after normalisation, else symbolic equality."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    sp, sg = _to_sympy(p), _to_sympy(g)
    if sp is None or sg is None:
        return False
    try:
        import sympy
        diff = sympy.simplify(sp - sg)
        if diff == 0:
            return True
        return bool(abs(float(sympy.N(diff))) < 1e-6)
    except Exception:
        return False


# ---------------------------------------------------------------------- #
# extraction                                                              #
# ---------------------------------------------------------------------- #

_LAST_NUM = re.compile(r"(-?\d[\d,]*\.?\d*)")


def extract_gsm8k(text: str) -> Optional[str]:
    """Last number, preferring the ``#### x`` convention if the model used it."""
    m = re.findall(r"####\s*(-?[\d,\.]+)", text)
    if m:
        return m[-1]
    nums = _LAST_NUM.findall(text)
    return nums[-1] if nums else None


def _find_boxed(text: str) -> Optional[str]:
    """Content of the **last** ``\\boxed{...}`` (brace-balanced)."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def extract_math(text: str) -> Optional[str]:
    b = _find_boxed(text)
    if b is not None:
        return b
    nums = _LAST_NUM.findall(text)
    return nums[-1] if nums else None


# Multiple-choice grading.  Our engine only implements ``generate_until`` (there
# is no log-likelihood path through the looped KV cache), so ARC-C / HellaSwag
# are graded from the *generated letter* rather than from choice log-probs.
_MC_EXPLICIT = re.compile(
    r"(?:answer|choice|option|correct)\s*(?:is|:|=)\s*\*{0,2}\(?\s*([A-H])\b",
    re.I)
_MC_BARE = re.compile(r"(?:^|[\s\(\[\.,;:\*\"'])(" + "|".join(_LETTERS) +
                      r")(?:\)|\.|:|,|;|\s|$)")


def extract_choice(text: str, n_opts: int = 0) -> Optional[str]:
    """Pick the chosen letter out of a generated answer string.

    Preference order: an explicit ``Answer: X`` statement (last occurrence, a
    model often restates it), then the first stand-alone letter -- browsing the
    option text before answering usually produces exactly one such letter, and
    taking the first avoids picking up a letter quoted from the reasoning that
    follows.  Returns ``None`` when nothing unambiguous is found.
    """
    if not text:
        return None
    allow = set(_LETTERS[:n_opts]) if 0 < n_opts <= len(_LETTERS) \
        else set(_LETTERS)
    explicit = [m.group(1).upper() for m in _MC_EXPLICIT.finditer(text)
                if m.group(1).upper() in allow]
    if explicit:
        return explicit[-1]
    for m in _MC_BARE.finditer(text):
        c = m.group(1).upper()
        if c in allow:
            return c
    return None


# ---------------------------------------------------------------------- #
# prompts                                                                 #
# ---------------------------------------------------------------------- #

GSM8K_FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees "
     "in the grove today. After they are done, there will be 21 trees. How many "
     "trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some "
     "more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, "
     "how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. "
     "The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, "
     "how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
     "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. "
     "The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
     "has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to "
     "Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
]


# Four hand-written worked examples in the style of the MATH dataset.  They are
# fixed (not sampled) so that every configuration sees exactly the same prompt.
MATH_FEWSHOT = [
    ("What is $3^2 + 4^2$?",
     "We have $3^2 = 9$ and $4^2 = 16$, so $9 + 16 = 25$. "
     "The answer is $\\boxed{25}$."),
    ("Solve for $x$: $2x + 5 = 13$.",
     "Subtracting 5 from both sides gives $2x = 8$, so $x = 4$. "
     "The answer is $\\boxed{4}$."),
    ("What is the area of a circle with radius 3?",
     "The area is $\\pi r^2 = \\pi \\cdot 3^2 = 9\\pi$. "
     "The answer is $\\boxed{9\\pi}$."),
    ("Compute $\\frac{1}{2} + \\frac{1}{3}$.",
     "Using the common denominator 6, we get "
     "$\\frac{3}{6} + \\frac{2}{6} = \\frac{5}{6}$. "
     "The answer is $\\boxed{\\frac{5}{6}}$."),
]


# ---------------------------------------------------------------------- #
# 8-shot variants                                                         #
# ---------------------------------------------------------------------- #
# FlashLoop evaluates with the EleutherAI LM Harness, whose default prompts
# are 5-shot for GSM8K and 4-shot (minerva-style) for MATH-500.  Our own
# prompts are *multi-turn* chat exemplars, which are more verbose, so 4
# exemplars already give ~480 / ~370 tokens.  To study the effect of a longer
# *input* (KV compression only pays off once the prompt is long enough -- see
# the ``min_prompt_tokens`` guard), we also ship an 8-shot variant that roughly
# doubles the prompt length.
GSM8K_FEWSHOT_8 = GSM8K_FEWSHOT + [
    ("Betty is saving money for a new wallet which costs $100. Betty has "
     "only half of the money she needs. Her parents decided to give her $15 "
     "for that purpose, and her grandparents twice as much as her parents. "
     "How much more money does Betty need to buy the wallet?",
     "Betty has half of $100, which is $50. Her parents give her $15 and her "
     "grandparents give twice that, so $30. In total she now has "
     "50 + 15 + 30 = 95. She still needs 100 - 95 = 5. The answer is 5."),
    ("A box contains 24 crayons. If each of the 6 students gets the same "
     "number of crayons, how many crayons does each student get?",
     "We divide 24 crayons equally among 6 students: 24 / 6 = 4. "
     "The answer is 4."),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
     "minutes of babysitting. How much did she earn?",
     "Weng earns $12 per 60 minutes, so per minute she earns 12 / 60 = $0.2. "
     "For 50 minutes she earns 0.2 * 50 = 10. The answer is 10."),
    ("A store sells notebooks for $3 each and pens for $2 each. Julie buys "
     "4 notebooks and 5 pens. How much does she spend in total?",
     "The notebooks cost 4 * $3 = $12 and the pens cost 5 * $2 = $10. "
     "In total she spends 12 + 10 = 22. The answer is 22."),
]


MATH_FEWSHOT_8 = MATH_FEWSHOT + [
    ("Factor $x^2 - 5x + 6$.",
     "We look for two numbers with sum $-5$ and product $6$: $-2$ and $-3$. "
     "So $x^2 - 5x + 6 = (x - 2)(x - 3)$. "
     "The answer is $\\boxed{(x-2)(x-3)}$."),
    ("What is the sum of the first 100 positive integers?",
     "The sum is $\\frac{100 \\cdot 101}{2} = 5050$. "
     "The answer is $\\boxed{5050}$."),
    ("A fair coin is flipped 3 times. What is the probability of getting "
     "exactly 2 heads?",
     "There are $2^3 = 8$ equally likely outcomes and $\\binom{3}{2} = 3$ of "
     "them have exactly 2 heads, so the probability is $3/8$. "
     "The answer is $\\boxed{\\frac{3}{8}}$."),
    ("Simplify $\\log_2 8 + \\log_2 4$.",
     "We have $\\log_2 8 = 3$ and $\\log_2 4 = 2$, so the sum is $3 + 2 = 5$. "
     "The answer is $\\boxed{5}$."),
]


def build_math_prompt(problem: str, fewshot: str = MATH_FEWSHOT,
                      suffix: str = "Solution: Let's solve this step by step.") -> str:
    return f"{fewshot}Problem: {problem}\n{suffix}"


def build_gsm8k_prompt(question: str, fewshot: bool = True) -> str:
    """Plain (non-chat) GSM8K prompt; kept for few-shot ablations."""
    if fewshot:
        return f"{GSM8K_FEWSHOT}Question: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


_CALC = re.compile(r"<<[^>]*>>")


def clean_gsm_solution(raw: str) -> str:
    """GSM8K references carry ``<<48/2=24>>`` calculator annotations and end
    with ``#### N``.  Turn one into a plain worked answer."""
    s = _CALC.sub("", str(raw))
    s = re.sub(r"####\s*(-?[\d,\.]+)", r"The answer is \1.", s.strip())
    return re.sub(r"\n{2,}", "\n", s).strip()


def eval_indices(n_total: int, n: int, seed: int) -> List[int]:
    """Fixed random subset shared by every method being compared."""
    return sorted(random.Random(seed).sample(range(n_total), min(n, n_total)))


def fewshot_pools(exclude_gsm=(), exclude_math=(), limit: int = 400):
    """Real worked examples used *only* to lengthen the prompt.

    The standard would be to draw few-shot exemplars from the training split,
    but only the test splits are available offline.  Since the exemplars are
    here purely to make the prompt long enough for KV compression to matter
    (and since ``exclude_*`` removes the evaluated items), drawing them from
    the held-out remainder of the *same* test split is fine -- the model sees
    no label it is graded on.

    Returns ``(gsm_pool, math_pool)`` as ``[{question, target}, ...]``.
    """
    gsm, math = load_gsm8k(), load_math500()
    gi, mi = set(exclude_gsm), set(exclude_math)
    gp = [{"question": gsm[i]["question"],
           "target": clean_gsm_solution(gsm[i]["solution"])}
          for i in range(len(gsm)) if i not in gi]
    mp = []
    for i in range(len(math)):
        if i in mi:
            continue
        sol = math[i].get("solution") or ""
        if "[asy]" in sol:            # Asymptote figure source, pure noise
            continue
        mp.append({"question": math[i]["problem"], "target": sol.strip()})
    return gp[:limit], mp[:limit]


def build_chat_prompt(tok, question: str, task: str = "gsm8k",
                      system: str = "You are a helpful assistant.",
                      fewshot=None) -> str:
    """Wrap ``question`` in Ouro's ChatML template.

    Ouro-1.4B ships a chat template (``<|im_start|>...<|im_end|>``); feeding it
    a raw "Question: ... Answer:" string makes it continue the *question*
    instead of answering, so evaluation must go through the template.

    ``fewshot`` is a list of ``(user, assistant)`` exemplar pairs.  They are
    injected as **real alternating chat turns** -- concatenating them as plain
    text inside the final user message makes the model treat the prompt as a
    text-completion exercise (it just continues the question and emits EOS),
    which was measured to collapse accuracy.  Multi-turn exemplars also make
    the prompt long enough for KV compression to be meaningful on these
    short benchmarks.

    The suffix pins the answer format so the extractor has something to grab:
    a final bare number for GSM8K, ``\\boxed{...}`` for MATH-500.
    """
    if task == "gsm8k":
        q = (f"{question}\nWork through the reasoning step by step, then give "
             "the final numeric answer on the last line as: Answer: <number>")
    else:
        q = (f"{question}\nWork through the reasoning step by step, then give "
             "the final answer inside \\boxed{}.")
    msgs = [{"role": "system", "content": system}]
    for u, a in (fewshot or []):
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
