"""Grade-gated training.

    train a few steps -> exam -> pass? promote to next grade : keep studying (up to max_attempts)

State is saved after every exam, so a stopped run resumes at the same grade.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from .curriculum import GRADES, STOI, VOCAB, encode, decode
from .model import GPT, ModelConfig

NL = STOI["\n"]


def make_batch(grades, current, rng, bs, review_frac):
    """Mostly the current grade; review_frac of questions come from earlier grades."""
    rows, masks = [], []
    for _ in range(bs):
        g = current
        if current.level > 1 and rng.random() < review_frac:
            g = rng.choice(grades[: current.level - 1])
        p, a = g.sample(rng, exam=False)
        ids = encode(p + a + "\n")
        rows.append(ids)
        masks.append([0] * len(p) + [1] * (len(a) + 1))  # learn only the answer
    T = max(map(len, rows))
    x = torch.zeros(bs, T, dtype=torch.long)
    m = torch.zeros(bs, T)
    for i, (r, k) in enumerate(zip(rows, masks)):
        x[i, : len(r)] = torch.tensor(r)
        m[i, : len(k)] = torch.tensor(k, dtype=torch.float)
    return x, m


def loss_fn(model, x, m):
    logits = model(x[:, :-1])
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction="none")
    mm = m[:, 1:].reshape(-1)
    return (ce * mm).sum() / mm.sum()


@torch.no_grad()
def exam(model, grade, n=200):
    """Exact-match accuracy on the grade's held-out exam paper."""
    model.eval()
    qs = grade.exam_set(n)
    correct = 0
    by_len = {}
    for p, a in qs:  # group by prompt length so each batch is rectangular
        by_len.setdefault(len(p), []).append((p, a))
    for group in by_len.values():
        x = torch.tensor([encode(p) for p, _ in group])
        out = model.generate(x, NL, max_new=max(len(a) for _, a in group) + 2)
        for (p, a), row in zip(group, out):
            gen = decode(row[len(p):].tolist()).split("\n")[0]
            correct += gen == a
    model.train()
    return correct / len(qs)


def report_card(model, grade, grades):
    """Score on this grade's new material, plus every earlier grade (review)."""
    card = {"new": exam(model, grade)}
    ok = card["new"] >= grade.pass_mark
    for g in grades[: grade.level - 1]:
        card[g.name] = exam(model, g)
        ok &= card[g.name] >= grade.review_mark
    return ok, card


def save(path, model, state):
    torch.save({"model": model.state_dict(), "cfg": model.c.__dict__, "state": state}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/school")
    ap.add_argument("--steps-per-lesson", type=int, default=300, help="training steps between exams")
    ap.add_argument("--max-attempts", type=int, default=15, help="exams a grade may take before the run halts")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--review-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grades", type=int, default=len(GRADES), help="stop after this grade")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    ckpt = os.path.join(a.out, "latest.pt")

    model = GPT(ModelConfig(vocab_size=len(VOCAB)))
    state = {"grade": 1, "step": 0, "attempt": 0, "history": []}
    if os.path.exists(ckpt):
        d = torch.load(ckpt)
        model.load_state_dict(d["model"])
        state = d["state"]
        print(f"resuming at grade {state['grade']}, step {state['step']}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"model: {model.n_params()/1e6:.2f}M params, random init")

    while state["grade"] <= a.grades:
        grade = GRADES[state["grade"] - 1]
        if state["attempt"] == 0:
            print(f"\n=== {grade.name}: {grade.skill} (pass >= {grade.pass_mark:.0%}) ===")
        t0 = time.time()
        for _ in range(a.steps_per_lesson):
            x, m = make_batch(GRADES, grade, rng, a.batch, a.review_frac)
            loss = loss_fn(model, x, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            state["step"] += 1
        state["attempt"] += 1
        passed, card = report_card(model, grade, GRADES)
        state["history"].append({"grade": grade.level, "step": state["step"], "passed": passed, "card": card})
        print(f"[{grade.name} exam {state['attempt']:>2} | step {state['step']} | loss {loss.item():.3f} | "
              f"{time.time()-t0:.0f}s] " + "  ".join(f"{k}={v:.0%}" for k, v in card.items())
              + ("  -> PASSED" if passed else "  -> not yet, back to study"))
        if passed:
            save(os.path.join(a.out, f"grade{grade.level}_passed.pt"), model, state | {"grade": grade.level + 1, "attempt": 0})
            state["grade"] += 1
            state["attempt"] = 0
        elif state["attempt"] >= a.max_attempts:
            save(ckpt, model, state)
            print(f"\n{grade.name} NOT passed after {a.max_attempts} exams. Halting: the model is held back.")
            break
        save(ckpt, model, state)
    else:
        print("\nAll grades passed. Graduated!")
    json.dump(state["history"], open(os.path.join(a.out, "report_card.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
