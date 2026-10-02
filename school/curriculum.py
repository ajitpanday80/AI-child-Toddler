"""The school: grades, their lesson generators, and their exams.

Every task is a text prompt ending in '=' and an answer ending in '\n'.
Exam questions are held out: a question whose hash lands in the exam bucket is
NEVER used for lessons, so passing means the model generalised, not memorised.
"""
import hashlib
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

Q = Tuple[str, str]  # (prompt, answer)
LETTERS = "abcdefghij"


def _is_exam(prompt: str) -> bool:
    return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % 5 == 0  # 20% held out


def _copy(r):
    s = "".join(r.choice(LETTERS) for _ in range(r.randint(1, 4)))
    return f"copy {s}=", s


def _reverse(r):
    s = "".join(r.choice(LETTERS) for _ in range(r.randint(2, 5)))
    return f"rev {s}=", s[::-1]


def _next(r):
    n = r.randint(0, 998)
    return f"next {n}=", str(n + 1)


def _add1(r):
    a, b = r.randint(0, 9), r.randint(0, 9)
    return f"{a}+{b}=", str(a + b)


def _add2(r):
    a, b = r.randint(10, 99), r.randint(10, 99)
    return f"{a}+{b}=", str(a + b)


def _sort(r):
    s = "".join(str(r.randint(0, 9)) for _ in range(r.randint(3, 6)))
    return f"sort {s}=", "".join(sorted(s))


@dataclass
class Grade:
    level: int
    name: str
    skill: str
    gen: Callable
    pass_mark: float = 0.95     # exam accuracy needed on this grade's new material
    review_mark: float = 0.90   # accuracy needed on every previous grade (no forgetting)
    holdout: bool = True        # False for tiny 'facts' grades (e.g. 100 sums): exam draws from the same facts

    def sample(self, rng: random.Random, exam: bool) -> Q:
        while True:
            p, a = self.gen(rng)
            if not self.holdout or _is_exam(p) == exam:
                return p, a

    def exam_set(self, n: int = 200) -> List[Q]:
        rng = random.Random(1000 + self.level)  # fixed seed: same exam paper every time
        seen, out = set(), []
        for _ in range(n * 20):
            p, a = self.sample(rng, exam=True)
            if p not in seen:
                seen.add(p)
                out.append((p, a))
            if len(out) == n:
                break
        return out


GRADES = [
    Grade(1, "Grade 1", "copy a word", _copy),
    Grade(2, "Grade 2", "reverse a word", _reverse),
    Grade(3, "Grade 3", "count: next number", _next),
    Grade(4, "Grade 4", "add single digits (facts)", _add1, holdout=False),
    Grade(5, "Grade 5", "add two-digit numbers", _add2),
    Grade(6, "Grade 6", "sort digits", _sort),
]

CHARS = sorted(set("abcdefghij0123456789+=\n copynextrvsrt"))
PAD = "\x00"
VOCAB = [PAD] + CHARS
STOI = {c: i for i, c in enumerate(VOCAB)}


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return "".join(VOCAB[i] for i in ids if i != 0)
