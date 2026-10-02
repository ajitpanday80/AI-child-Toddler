"""Stage-gated training with a human in the loop, split into short parts (for Kaggle / Colab sessions).

    study a lesson -> exam -> (pass) -> "move to the next stage? [y/n]"
    every --part-minutes (default 60) the run saves and asks "continue with the next part? [y/n]"
    No answer / 'n' / Ctrl-C / closed notebook: everything is saved; run the same command to resume.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from . import exams
from .data import load_source, load_user_content
from .model import GPT, ModelConfig
from .stages import STAGES

PASS_SCALE = 1.0     # --pass-scale: multiplies every pass mark (e.g. 0.9 = 10% easier)
SIZES = {"tiny": (4, 4, 128), "small": (6, 8, 256), "base": (8, 8, 384)}   # (layers, heads, width)  ~0.9M / 5M / 14M params


def log(*a):
    print(*a, flush=True)


class StageData:
    """Teaching material + held-out exam papers for one stage (downloaded once, then cached)."""

    def __init__(self, stage, cache_dir, content_dir):
        docs = []
        for s in stage.sources:
            try:
                docs += load_source(s, cache_dir)
            except Exception as e:
                log(f"  ! could not load {s.kind} {s.ref if not isinstance(s.ref, tuple) else ''}: {e}")
        mine = load_user_content(content_dir, stage.folder, cache_dir, log)
        if mine:
            log(f"  + {len(mine)} paragraphs from your own content/{stage.folder}/")
        docs += mine
        by = {}
        for d in docs:
            by.setdefault(d.subject, []).append(d)
        self.subjects, self.papers, self.counts = {}, {}, {}
        for subj, ds in by.items():
            study, held = [d for d in ds if not d.is_exam()], [d for d in ds if d.is_exam()]
            if not study:
                continue
            corpus, anchors, pos = [], [], 0
            for d in study:                                   # windows can end right after a moral / answer
                corpus.append(d.text)
                pos += len(d.text) + 2
                if d.moral or d.task:
                    anchors.append(pos - 2)
            self.subjects[subj] = ("\n\n".join(corpus), anchors)
            self.counts[subj] = (len(study), len(held))
            tasks = {d.task for d in ds if d.task}
            for t in tasks:
                self.papers[t] = ("label", exams.label_paper(held, t))
            if any(d.moral for d in held):
                self.papers["moral"] = ("moral", exams.moral_paper(held))
            if not tasks and len([d for d in held if len(d.story or d.text) >= 60]) >= 8:
                self.papers["cloze:" + subj] = ("cloze", exams.cloze_paper(held, exams.vocabulary(ds)))
        self.heldout = [d for d in docs if d.is_exam() and not d.task][:300]
        if not self.papers:
            raise RuntimeError(f"{stage.name}: no exam could be built (not enough material downloaded). "
                               "Check your internet connection and re-run, or add material to content/%s/." % stage.folder)

    def window(self, rng, T):
        corpus, anchors = self.subjects[rng.choice(list(self.subjects))]     # every subject gets equal study time
        if anchors and rng.random() < 0.3:
            start = max(0, rng.choice(anchors) - T - 1)
        else:
            start = rng.randrange(0, max(1, len(corpus) - T - 1))
        return corpus[start: start + T + 1]


def make_batch(data, stage_idx, rng, bs, T, review_frac, dev="cpu"):
    rows = []
    for _ in range(bs):
        k = rng.randrange(stage_idx) if stage_idx > 0 and rng.random() < review_frac else stage_idx   # revise older stages
        w = data[k].window(rng, T)
        rows.append(exams.encode(w + "\n" * (T + 1 - len(w))))
    x = torch.tensor(rows, device=dev)
    return x[:, :-1], x[:, 1:]


def exam_card(model, sd):
    card = {}
    for name, (kind, paper) in sd.papers.items():
        card[name] = {"cloze": exams.cloze_score, "moral": exams.moral_score, "label": exams.label_score}[kind](model, paper)
    cl = [v for k, v in card.items() if k.startswith("cloze:")]
    card["cloze"] = sum(cl) / len(cl) if cl else None
    card["bpc"] = exams.heldout_bpc(model, sd.heldout)
    return card


def quality(card):
    """One number for 'how well does it generalise': word-cloze average + the labelled-task scores."""
    return (card.get("cloze") or 0) + sum(card[t] or 0 for t in ("emotion", "judgment") if t in card)


def judge(model, idx, ds, state):
    """Pass = this stage's marks reached in EVERY subject area AND no earlier stage forgotten."""
    stage = STAGES[idx]
    card = exam_card(model, ds[idx])
    cl = [v for k, v in card.items() if k.startswith("cloze:")]
    mark = stage.cloze_pass * PASS_SCALE
    ok = bool(cl) and card["cloze"] >= mark and min(cl) >= mark - 0.20
    for t, tmark in stage.tasks_pass.items():
        if card.get(t) is None:
            log(f"  (no '{t}' exam available this time, skipping that mark)")
        else:
            ok &= card[t] >= tmark * PASS_SCALE
    review = {}
    if ok:                                                    # only worth checking the old stages once this stage is passed
        for k in range(idx):
            old, base = exam_card(model, ds[k]), state["baselines"].get(STAGES[k].name, {})
            review[STAGES[k].name] = old["cloze"]
            for m in [m for m in base if m == "cloze" or m in STAGES[k].tasks_pass]:
                if old.get(m) is not None and base[m] is not None:
                    ok &= old[m] >= base[m] - stage.review_tol
    return ok, card, review


# ---------------------------------------------------------------- persistence
def paths(out):
    return os.path.join(out, "state.json"), os.path.join(out, "latest.pt")


def build(a):
    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    sp, cp = paths(a.out)
    state = {"stage": 0, "phase": "studying", "attempt": 0, "step": 0, "parts": 0, "best_q": -1.0, "since_best": 0, "baselines": {}, "history": []}
    if os.path.exists(sp) and os.path.exists(cp):
        state.update(json.load(open(sp)))
        ck = torch.load(cp, map_location="cpu")
        model = GPT(ModelConfig(**ck["cfg"]))
        model.load_state_dict(ck["model"])
        model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
        opt.load_state_dict(ck["opt"])
        log(f"Resuming: {STAGES[min(state['stage'], len(STAGES) - 1)].name}, {state['step']} steps so far, {state['parts']} parts done.")
    else:
        L, H, C = SIZES[a.size]
        model = GPT(ModelConfig(vocab_size=128, n_layer=L, n_head=H, n_embd=C, dropout=0.2)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    return model, opt, state, dev


def save_state(out, model, opt, state):
    sp, cp = paths(out)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "cfg": model.c.__dict__}, cp + ".tmp")
    os.replace(cp + ".tmp", cp)
    json.dump(state, open(sp + ".tmp", "w"), indent=1)
    os.replace(sp + ".tmp", sp)


def status(out):
    sp, _ = paths(out)
    st = json.load(open(sp)) if os.path.exists(sp) else None
    if not st:
        return log("No training yet. Start with: python -m school.train")
    for i, s in enumerate(STAGES):
        mark = "PASSED" if s.name in st["baselines"] else (f"<- now ({st['phase']})" if i == st["stage"] else "")
        log(f"{i:>2}. {s.name:<12} {s.skill:<70} {mark}")
    log(f"steps trained: {st['step']}   exams taken: {len(st['history'])}   parts completed: {st['parts']}")


def fmt(card):
    cl = " ".join(f"{k[6:9]}={v:.0%}" for k, v in card.items() if k.startswith("cloze:"))
    tk = " ".join(f"{k}={v:.0%}" for k, v in card.items() if k in ("moral", "emotion", "judgment") and v is not None)
    return f"words[{cl}] avg={card['cloze'] or 0:.0%} {tk} bpc={card['bpc']:.2f}"


# ---------------------------------------------------------------- the school day
def run(a, ask=input):
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed + state_seed(a))
    model, opt, state, dev = build(a)
    amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    log(f"model: {model.n_params()/1e6:.2f}M parameters on {dev} (own weights, trained from scratch)")
    data, t_part = {}, time.time()

    def ds_upto(i):
        for k in range(i + 1):
            if k not in data:
                log(f"  fetching lessons for {STAGES[k].name} ...")
                data[k] = StageData(STAGES[k], a.data, a.content)
                n = data[k].counts
                log("  " + ", ".join(f"{s}:{c[0]}" for s, c in n.items()) + f"  (held back for exam: {sum(c[1] for c in n.values())})")
        return [data[k] for k in range(i + 1)]

    try:
        while state["stage"] < len(STAGES):
            i = state["stage"]
            stage = STAGES[i]
            if state["phase"] == "awaiting_approval":
                if not approve(i, state, ask):
                    save_state(a.out, model, opt, state)
                    return log("\nStopped. Run the same command again later and I'll ask you again.")
                if state["stage"] == len(STAGES):
                    state["phase"] = "graduated"
                    save_state(a.out, model, opt, state)
                    break
                state.update(stage=i + 1, phase="studying", attempt=0)
                save_state(a.out, model, opt, state)
                continue
            ds = ds_upto(i)
            log(f"\n=== {stage.name}: {stage.skill} ===")
            chars = sum(len(c) for c, _ in ds[i].subjects.values())
            steps = min(a.steps, max(40, int(a.epochs * chars / (a.batch * model.c.block_size))))   # ~a.epochs passes over the lessons per exam
            log(f"  {chars/1e3:.0f}K characters of lessons -> exam every {steps} steps")
            while True:
                if time.time() - t_part >= a.part_minutes * 60:      # part finished -> checkpoint and ask
                    state["parts"] += 1
                    save_state(a.out, model, opt, state)
                    log(f"\n--- Part {state['parts']} finished ({a.part_minutes:g} min). Progress saved. ---")
                    if not yes(ask, f"Continue with the next part (still {stage.name})? [y/n] "):
                        return log("Stopped. Run the same command to start the next part.")
                    t_part = time.time()
                t0 = time.time()
                model.train()
                for _ in range(steps):
                    x, y = make_batch(ds, i, rng, a.batch, model.c.block_size, a.review_frac, dev)
                    for g in opt.param_groups:
                        g["lr"] = a.lr * min(1.0, (state["step"] + 1) / 100)       # short warm-up after every (re)start
                    with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                        loss = F.cross_entropy(model(x).float().reshape(-1, 128), y.reshape(-1))
                    opt.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    state["step"] += 1
                state["attempt"] += 1
                ok, card, review = judge(model, i, ds, state)
                state["history"].append({"stage": stage.name, "step": state["step"], "passed": ok, "card": card, "review": review})
                need = f"need avg>={stage.cloze_pass * PASS_SCALE:.0%} " + " ".join(f"{k}>={v * PASS_SCALE:.0%}" for k, v in stage.tasks_pass.items())
                log(f"exam {state['attempt']:>2} | step {state['step']} | loss {loss.item():.2f} | {time.time()-t0:.0f}s | {fmt(card)} | {need}"
                    + (f" | review {' '.join(f'{k}={v:.0%}' for k, v in review.items())}" if review else "")
                    + ("  -> PASSED" if ok else "  -> not yet, studying more"))
                if ok:
                    state["baselines"][stage.name] = {k: v for k, v in card.items() if k != "bpc"}
                    state["phase"] = "awaiting_approval"
                    torch.save(model.state_dict(), os.path.join(a.out, f"passed_{i}_{stage.name.replace(' ', '_')}.pt"))
                    save_state(a.out, model, opt, state)
                    state["best_q"], state["since_best"] = -1.0, 0
                    break
                q = quality(card)
                if q > state["best_q"] + 0.005:
                    state["best_q"], state["since_best"] = q, 0
                    torch.save(model.state_dict(), os.path.join(a.out, "best_current_stage.pt"))
                else:
                    state["since_best"] += 1
                save_state(a.out, model, opt, state)
                if state["since_best"] >= a.patience:
                    best = os.path.join(a.out, "best_current_stage.pt")
                    if os.path.exists(best):
                        model.load_state_dict(torch.load(best, map_location=dev))          # go back to its best moment
                    state["since_best"] = 0
                    save_state(a.out, model, opt, state)
                    log(f"\n{stage.name}: no improvement for {a.patience} exams (best so far: {fmt(exam_card(model, ds[i]))}).")
                    log("More training will not help; it needs more or better material (content/%s/) or easier marks (--pass-scale 0.9)." % stage.folder)
                    if i + 1 < len(STAGES) and yes(ask, f"Move on to {STAGES[i + 1].name} anyway, using the best version so far? [y/n] "):
                        state["baselines"][stage.name] = {k: v for k, v in exam_card(model, ds[i]).items() if k != "bpc"}
                        torch.save(model.state_dict(), os.path.join(a.out, f"passed_{i}_{stage.name.replace(' ', '_')}.pt"))
                        state.update(stage=i + 1, phase="studying", attempt=0, best_q=-1.0, since_best=0)
                        save_state(a.out, model, opt, state)
                        break
                    return log("Stopped. Add material or adjust marks, then run the same command again.")
        log("\nAll stages passed. Graduated!")
    except KeyboardInterrupt:
        save_state(a.out, model, opt, state)
        log("\nInterrupted. Progress saved; run the same command to resume.")


def state_seed(a):
    sp, _ = paths(a.out)
    return json.load(open(sp)).get("step", 0) if os.path.exists(sp) else 0   # fresh randomness after each resume


def yes(ask, prompt):
    try:
        return ask(prompt).strip().lower().startswith("y")
    except EOFError:           # no keyboard (e.g. Kaggle "Save & Run All"): stop safely
        return False


def approve(i, state, ask):
    stage = STAGES[i]
    c = state["baselines"].get(stage.name, {})
    log(f"\n*** {stage.name} PASSED ***  " + " ".join(f"{k}={v:.0%}" for k, v in c.items() if v is not None and ":" not in k))
    if i + 1 >= len(STAGES):
        state["stage"] = len(STAGES)
        return True
    return yes(ask, f"Move on to {STAGES[i + 1].name} ({STAGES[i + 1].skill})? [y/n] ")


def main(argv=None, ask=input):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/school", help="checkpoints + progress (put this on Drive / Kaggle output to keep it)")
    ap.add_argument("--data", default="data", help="download cache")
    ap.add_argument("--content", default="content", help="your own material: content/<grade>/ (pdf, txt, links)")
    ap.add_argument("--part-minutes", type=float, default=60, help="length of one training part before asking to continue")
    ap.add_argument("--size", choices=SIZES, default="small", help="model size (fixed when a run is first created)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=400, help="training steps between exams")
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over a stage's lessons between exams (small lessons => fewer steps, avoids memorising)")
    ap.add_argument("--patience", type=int, default=8, help="exams without improvement before asking whether to move on anyway")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--review-frac", type=float, default=0.25, help="share of study time spent revising earlier stages")
    ap.add_argument("--pass-scale", type=float, default=1.0, help="multiply every pass mark (0.9 = 10%% easier, 1.1 = stricter)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--status", action="store_true", help="show the report card and exit")
    a = ap.parse_args(argv)
    global PASS_SCALE
    PASS_SCALE = a.pass_scale
    status(a.out) if a.status else run(a, ask)


if __name__ == "__main__":
    main()
