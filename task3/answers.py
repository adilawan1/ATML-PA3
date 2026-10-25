"""Answer normalization, equivalence grouping, and grading (gold answers are used ONLY by `grade_groups`).

MATH: math-verify equivalence. Groups are formed by a symmetric check verify(a,b) or verify(b,a) against
      each existing group's representative (first member in sorted string order), without gold answers.
MathVista: the official MathVista normalization (answer type, choices, precision), with multiple-choice
      answers represented as option letters.
Missing extractions map to a single NO_ANSWER group that is always graded incorrect.
"""
import logging
import math
import os
import re

logging.getLogger("math_verify").setLevel(logging.ERROR)
# math-verify enforces timeouts with SIGALRM, which only works on POSIX (fine inside worker processes on Colab)
TIMEOUT = None if os.name == "nt" else 5

NO_ANSWER = "NO_ANSWER"


# ----------------------------------------------------------------------------- shared
def is_missing(a):
    if a is None:
        return True
    if isinstance(a, float) and math.isnan(a):
        return True
    return str(a).strip() == ""


def extract_last_boxed(text):
    """Last \\boxed{...} (or \\fbox{...}) content using balanced braces; None if absent or unbalanced."""
    if text is None:
        return None
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # "\boxed 5" form
        m = re.match(r"\\boxed\s+([^\s$]+)", text[idx:])
        return m.group(1) if m else None
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


# ----------------------------------------------------------------------------- MATH (math-verify)
class MathVerifier:
    def __init__(self):
        from math_verify import parse, verify

        self._parse, self._verify = parse, verify
        self._pcache, self._vcache = {}, {}

    def parsed(self, s):
        if s not in self._pcache:
            try:
                self._pcache[s] = self._parse("\\boxed{" + s + "}", parsing_timeout=TIMEOUT)
            except TypeError:  # older math-verify without the timeout kwarg
                self._pcache[s] = self._parse("\\boxed{" + s + "}")
            except Exception:
                self._pcache[s] = []
        return self._pcache[s]

    def _v(self, a, b):
        try:
            return bool(self._verify(a, b, timeout_seconds=TIMEOUT))
        except TypeError:
            return bool(self._verify(a, b))
        except Exception:
            return False

    def equal(self, a, b):
        if a == b:
            return True
        key = (a, b) if a < b else (b, a)
        if key not in self._vcache:
            pa, pb = self.parsed(a), self.parsed(b)
            self._vcache[key] = bool(pa) and bool(pb) and (self._v(pa, pb) or self._v(pb, pa))
        return self._vcache[key]

    def correct(self, gold, pred):
        pg, pp = self.parsed(str(gold)), self.parsed(pred)
        return bool(pg) and bool(pp) and self._v(pg, pp)


def group_math(answers, mv: MathVerifier):
    """answers: list[str|None] -> (group index per answer, list of group representatives)."""
    reps, gid = [], []
    norm = [NO_ANSWER if is_missing(a) else str(a).strip() for a in answers]
    order = sorted(set(norm))
    rep_of = {}
    for s in order:
        if s == NO_ANSWER:
            rep_of[s] = NO_ANSWER
            continue
        for r in reps:
            if r != NO_ANSWER and mv.equal(r, s):
                rep_of[s] = r
                break
        else:
            reps.append(s)
            rep_of[s] = s
    if NO_ANSWER in rep_of and NO_ANSWER not in reps:
        reps.append(NO_ANSWER)
    gid = [reps.index(rep_of[s]) for s in norm]
    return gid, reps


# ----------------------------------------------------------------------------- MathVista
def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_mathvista(extraction, meta):
    """Port of MathVista's normalize_extracted_answer; multiple-choice results are returned as letters."""
    if is_missing(extraction):
        return NO_ANSWER
    qtype, atype = meta["question_type"], meta["answer_type"]
    if qtype == "multi_choice":
        choices = list(meta["choices"])
        ext = str(extraction).strip()
        letter = re.findall(r"\(([a-zA-Z])\)", ext)
        if letter:
            ext = letter[0].upper()
        options = [chr(ord("A") + i) for i in range(len(choices))]
        if ext.upper() in options and len(ext) == 1:
            return ext.upper()
        if ext in choices:
            return options[choices.index(ext)]
        # official rule: most similar option by edit distance
        d = [_levenshtein(ext, str(c)) for c in choices]
        return options[d.index(min(d))]
    s = str(extraction).strip().replace(",", "")
    if atype == "integer":
        try:
            return str(int(float(s)))
        except Exception:
            return NO_ANSWER
    if atype == "float":
        try:
            return str(round(float(s), int(meta["precision"])))
        except Exception:
            return NO_ANSWER
    return str(extraction).strip()


def gold_mathvista(meta):
    if meta["question_type"] == "multi_choice":
        choices = list(meta["choices"])
        return chr(ord("A") + choices.index(meta["answer"])) if meta["answer"] in choices else str(meta["answer"])
    return normalize_mathvista(meta["answer"], meta)


def group_mathvista(answers, meta):
    norm = [normalize_mathvista(a, meta) for a in answers]
    reps = sorted(set(n for n in norm if n != NO_ANSWER)) + ([NO_ANSWER] if NO_ANSWER in norm else [])
    return [reps.index(n) for n in norm], reps


# ----------------------------------------------------------------------------- grading (gold used here only)
def grade_groups(reps, gold, grader, mv=None, meta=None):
    out = []
    for r in reps:
        if r == NO_ANSWER:
            out.append(False)
        elif grader == "math_verify":
            out.append(mv.correct(gold, r))
        else:
            out.append(r == gold_mathvista(meta))
    return out
