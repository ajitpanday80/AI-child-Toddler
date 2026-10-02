"""Exams for a language model.

cloze  : a sentence with one missing word, 4 same-length candidate words (1 true, 3 look-alikes
         drawn from the grade's own vocabulary). The model must find the true one more likely.
moral  : an unseen fable (story only) and 4 candidate morals (1 true, 3 from other fables).
review : earlier stages are re-examined; they may not slip far below the score they passed with.
Papers are generated with a fixed seed from held-out documents, so every exam paper is identical.
"""
import random
import re
from collections import Counter
from typing import List, Optional

import torch
import torch.nn.functional as F

from .data import Doc

CTX = 200       # characters of context shown before the missing word
MORAL_CTX = 130


def encode(s: str) -> List[int]:
    return [min(ord(c), 127) for c in s]


def decode(ids) -> str:
    return "".join(chr(i) for i in ids)


@torch.no_grad()
def span_logprob(model, seqs: List[str], starts: List[int], mean: bool = False) -> List[float]:
    """log P(seq[start:] | seq[:start]) for each sequence (sum, or mean per character)."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for b in range(0, len(seqs), 128):                       # batches keep memory small
        chunk, st_ = seqs[b: b + 128], starts[b: b + 128]
        L = max(map(len, chunk))
        x = torch.zeros(len(chunk), L, dtype=torch.long, device=dev)
        for i, s in enumerate(chunk):
            x[i, : len(s)] = torch.tensor(encode(s))
        lp = F.log_softmax(model(x[:, :-1]).float(), -1).gather(2, x[:, 1:, None])[..., 0]   # lp[:, t-1] = log P(x_t | x_<t)
        for i, (s, st) in enumerate(zip(chunk, st_)):
            v = lp[i, st - 1: len(s) - 1]
            out.append((v.mean() if mean else v.sum()).item())
    return out


def vocabulary(docs: List[Doc]) -> Counter:
    return Counter(w.lower() for d in docs for w in re.findall(r"[A-Za-z]{3,}", d.text))


def cloze_paper(exam_docs: List[Doc], vocab: Counter, n: int = 60, seed: int = 7):
    rng = random.Random(seed)
    pool = [d.story if d.story else d.text for d in exam_docs]
    pool = [p for p in pool if len(p) >= 60]
    by_len = {}
    for w, c in vocab.items():
        if c >= 2:
            by_len.setdefault(len(w), []).append((w, c))
    paper = []
    for _ in range(n * 30):
        if len(paper) == n or not pool:
            break
        p = rng.choice(pool)
        ms = [m for m in re.finditer(r"[A-Za-z]{3,}", p) if m.start() >= 15]
        if not ms:
            continue
        m = rng.choice(ms)
        true = m.group()
        opts = [(w, c) for w, c in by_len.get(len(true), []) if w != true.lower()]
        if len(opts) < 3:
            continue
        ws, cs = zip(*opts)
        decoys = set()
        while len(decoys) < 3:                       # frequency-weighted, so decoys are plausible words
            decoys.add(rng.choices(ws, cs)[0])
        cap = (lambda w: w.capitalize()) if true[0].isupper() else (lambda w: w)
        paper.append((p[max(0, m.start() - CTX): m.start()], [true] + [cap(w) for w in decoys], p[m.end(): m.end() + 1]))
    return paper


def cloze_score(model, paper) -> float:
    if not paper:
        return 0.0
    seqs, starts = [], []
    for prefix, cands, nxt in paper:
        for c in cands:
            seqs.append(prefix + c + nxt)
            starts.append(len(prefix))
    sc = span_logprob(model, seqs, starts)
    right = sum(max(range(4), key=lambda k: sc[4 * i + k]) == 0 for i in range(len(paper)))
    return right / len(paper)


def moral_paper(exam_docs: List[Doc], seed: int = 11, draws: int = 3):
    fables = [d for d in exam_docs if d.moral]
    rng = random.Random(seed)
    paper = []
    for d in fables:
        for _ in range(draws):
            others = rng.sample([f for f in fables if f is not d], min(3, len(fables) - 1))
            paper.append((d.story[-MORAL_CTX:] + "\n\nMoral: ", [d.moral[:90]] + [o.moral[:90] for o in others]))
    return paper


def moral_score(model, paper) -> Optional[float]:
    paper = [p for p in paper if len(p[1]) == 4]
    if not paper:
        return None
    seqs, starts = [], []
    for ctx, cands in paper:
        for c in cands:
            seqs.append(ctx + c + "\n")
            starts.append(len(ctx))
    sc = span_logprob(model, seqs, starts, mean=True)   # per-character, so short morals aren't favoured
    right = sum(max(range(4), key=lambda k: sc[4 * i + k]) == 0 for i in range(len(paper)))
    return right / len(paper)


@torch.no_grad()
def heldout_bpc(model, exam_docs: List[Doc], n: int = 40, seed: int = 3) -> float:
    """Bits per character on unseen text (lower is better; ~8 = knows nothing, ~2 = knows English well)."""
    rng = random.Random(seed)
    texts = [d.text for d in exam_docs if len(d.text) > 40]
    if not texts:
        return float("nan")
    seqs = [rng.choice(texts)[:250] for _ in range(n)]
    sc = span_logprob(model, seqs, [1] * n, mean=True)
    return -sum(sc) / n / 0.6931


def label_paper(exam_docs: List[Doc], task: str, n: int = 200, seed: int = 5):
    """Labelled examples (emotion / right-or-wrong). BALANCED: the same number per answer, so always
    guessing the common answer scores only chance (50% for right/wrong, 17% for 6 emotions)."""
    by = {}
    for d in sorted((d for d in exam_docs if d.task == task), key=lambda d: d.key):
        by.setdefault(d.answer, []).append(d)
    if len(by) < 2:
        return []
    per = min(n // len(by), min(len(v) for v in by.values()))
    rng = random.Random(seed)
    docs = [d for v in by.values() for d in rng.sample(v, per)]
    rng.shuffle(docs)
    return [(d.prompt, d.choices, d.choices.index(d.answer)) for d in docs]


def label_score(model, paper) -> Optional[float]:
    if not paper:
        return None
    seqs, starts, spans = [], [], []
    for prompt, choices, _ in paper:
        spans.append(len(choices))
        for c in choices:
            seqs.append(prompt + c + "\n")
            starts.append(len(prompt))
    sc = span_logprob(model, seqs, starts)
    right, pos = 0, 0
    for (_, _, gold), k in zip(paper, spans):
        right += max(range(k), key=lambda j: sc[pos + j]) == gold
        pos += k
    return right / len(paper)
