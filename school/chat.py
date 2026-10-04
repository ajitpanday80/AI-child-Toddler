"""Ask the model questions by hand, and see what it has been taught so far.

    python -m school.chat                      # shows what it learned, then lets you type questions
    python -m school.chat --learned            # only the "what it has learned" report
In a notebook:  from school.chat import main; main(["--out", "/kaggle/working/runs/school"])

Questions (a line without a command is a "say"):
    /say   The sun is            finish the text
    /feel  i lost my toy         which feeling? (sadness joy love anger fear surprise)
    /wrong I took his lunch      is it wrong? yes/no
    /word  The dog ran to the ___ | park | tree | moon     choose the word that fits
    /moral <a short story>       what lesson does it draw?
    /learned  /topics  /help  /quit
It is a small model trained from scratch: answers are what it picked up from its lessons, not reasoning.
Subjects it has not been taught yet are marked, and its answers there are guesses.
"""
import argparse
import json
import math
import os

import torch

from . import exams
from .data import EMOTIONS, wiki_cached
from .model import GPT, ModelConfig
from .stages import BOOKS, STAGES
from .sync import make_syncer

HELP = __doc__.split("Questions")[1].split("It is a small")[0]


# ---------------------------------------------------------------- what has it been taught?
def stage_topics(stage, data_dir, content_dir):
    """{subject: [lines]} describing the real lesson content of a stage (pages that failed to download are marked)."""
    out = {}
    for src in stage.sources:
        if src.kind == "wiki":
            lang, titles = src.ref
            got = [t for t in titles if wiki_cached(data_dir, lang, t)] if os.path.isdir(data_dir) else titles
            line = ", ".join(got) or "(nothing downloaded)"
            if len(got) < len(titles):
                line += f"   [not available: {', '.join(t for t in titles if t not in got)}]"
            out.setdefault(src.subject, []).append(line)
        elif src.kind == "gutenberg":
            out.setdefault("values" if src.fables else src.subject, []).append(BOOKS.get(src.ref, f"book {src.ref}"))
        elif src.kind == "tinystories":
            out.setdefault(src.subject, []).append("TinyStories (simple children's stories)")
        elif src.kind == "emotion":
            out.setdefault("heart", []).append("short texts labelled with a feeling: " + ", ".join(EMOTIONS))
        elif src.kind == "ethics":
            out.setdefault("judgment", []).append("everyday situations labelled wrong / not wrong")
    mine = os.path.join(content_dir, stage.folder)
    files = [f for _, _, fs in os.walk(mine) for f in fs if not f.startswith(".") and f != "README.md"] if os.path.isdir(mine) else []
    if files:
        out.setdefault("extra", []).append(f"your files: {', '.join(sorted(files)[:8])}" + (" ..." if len(files) > 8 else ""))
    return out


def learned_report(state, data_dir="data", content_dir="content"):
    cur = state.get("stage", 0)
    lines = ["=" * 70, "WHAT IT HAS LEARNED SO FAR", "=" * 70]
    for i, st in enumerate(STAGES):
        if i > cur:
            break
        passed = st.name in state.get("baselines", {})
        tag = ("ACCEPTED without passing" if st.name in state.get("accepted", []) else "PASSED") if passed else (
            "in progress" if i == cur and state.get("phase") != "graduated" else "")
        lines.append(f"\n{st.name}  -  {st.skill}   [{tag}]")
        if passed:
            b = state["baselines"][st.name]
            marks = [f"{k}={v:.0%}" for k, v in b.items() if v is not None and ":" not in k]
            subj = [(k[6:], v) for k, v in b.items() if k.startswith("cloze:") and v is not None]
            lines.append("    scores: " + "  ".join(marks))
            weak = [f"{k} ({v:.0%})" for k, v in subj if v < 0.35]
            if weak:
                lines.append("    WEAK (barely above guessing): " + ", ".join(weak))
        for subj, items in stage_topics(st, data_dir, content_dir).items():
            lines.append(f"    {subj:<9}: " + " | ".join(items))
    nxt = [s.name for s in STAGES[cur + 1:cur + 3]]
    lines.append("\nNot taught yet: " + (", ".join(s.name for s in STAGES[cur + 1:]) or "nothing, all stages done"))
    return "\n".join(lines)


def taught(state, kind):
    """Has any stage the model has reached taught this skill? (so we can warn that an answer is a guess)"""
    cur = state.get("stage", 0)
    kinds = {"emotion": "emotion", "judgment": "ethics"}
    return any(s.kind == kinds[kind] for st in STAGES[:cur + 1] for s in st.sources)


# ---------------------------------------------------------------- asking
def load(out, ckpt=None, device="auto"):
    dev = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    latest = os.path.join(out, "latest.pt")
    ck = torch.load(ckpt or latest, map_location="cpu")
    cfg = ck["cfg"] if "cfg" in ck else torch.load(latest, map_location="cpu")["cfg"]
    model = GPT(ModelConfig(**cfg))
    model.load_state_dict(ck.get("model", ck))
    sp = os.path.join(out, "state.json")
    state = json.load(open(sp)) if os.path.exists(sp) else {"stage": 0, "baselines": {}}
    return model.to(dev).eval(), state


def say(model, prompt, n=200, temperature=0.8, stop_line=False):
    dev = next(model.parameters()).device
    ids = torch.tensor([exams.encode(prompt)], device=dev)
    text = exams.decode(model.sample(ids, n, temperature)[0, len(prompt):].tolist())
    return text.split("\n")[0] if stop_line else text


def rank(model, prompt, choices):
    """Probability the model gives each choice after the prompt (softmax over their likelihoods)."""
    sc = exams.span_logprob(model, [prompt + c + "\n" for c in choices], [len(prompt)] * len(choices))
    m = max(sc)
    z = [math.exp(x - m) for x in sc]
    return sorted(zip(choices, [x / sum(z) for x in z]), key=lambda t: -t[1])


def fill_blank(model, text, options):
    pre, _, post = text.partition("___")
    sc = exams.span_logprob(model, [pre + o + post[:25] for o in options], [len(pre)] * len(options))
    m = max(sc)
    z = [math.exp(x - m) for x in sc]
    return sorted(zip(options, [x / sum(z) for x in z]), key=lambda t: -t[1])


def answer(model, state, line, temperature=0.8):
    cmd, _, rest = line.strip().partition(" ")
    rest = rest.strip()
    if not cmd.startswith("/"):
        cmd, rest = "/say", line.strip()
    if cmd == "/say":
        return rest + say(model, rest, 200, temperature)
    if cmd == "/moral":
        return "Moral:" + say(model, rest + "\n\nMoral:", 120, temperature, stop_line=True)
    if cmd == "/feel":
        note = "" if taught(state, "emotion") else "(feelings are taught from Grade 1 - this is a guess)\n"
        return note + "\n".join(f"  {c.strip():<9}{p:>5.0%}" for c, p in rank(model, f"Feeling: {rest}\nEmotion:", [" " + e for e in EMOTIONS]))
    if cmd == "/wrong":
        note = "" if taught(state, "judgment") else "(right-or-wrong is taught from the Values stage - this is a guess)\n"
        r = rank(model, f"Situation: {rest}\nIs it wrong?", [" yes", " no"])
        return note + "\n".join(f"  {'wrong' if c == ' yes' else 'not wrong':<10}{p:>5.0%}" for c, p in sorted(r, key=lambda t: t[0] != " yes"))
    if cmd == "/word":
        text, *opts = [x.strip() for x in rest.split("|")]
        if "___" not in text or len(opts) < 2:
            return "Use:  /word The dog ran to the ___ | park | tree | moon"
        return "\n".join(f"  {c:<14}{p:>5.0%}" for c, p in fill_blank(model, text, opts))
    return "Unknown command. Type /help"


def main(argv=None, ask=input):
    ap = argparse.ArgumentParser(description="Ask the model questions.")
    ap.add_argument("--out", default="runs/school")
    ap.add_argument("--data", default="data")
    ap.add_argument("--content", default="content")
    ap.add_argument("--ckpt", help="e.g. runs/school/passed_0_Values.pt: the model as it was when it passed that stage")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--hf-repo", help="restore the model from this Hugging Face backup first")
    ap.add_argument("--sync-dir", help="or from this backup folder")
    ap.add_argument("--sync-minutes", type=float, default=10)
    ap.add_argument("--learned", action="store_true", help="only show what it has learned")
    a = ap.parse_args(argv)
    sync = make_syncer(a)
    if sync:
        sync.pull()
    model, state = load(a.out, a.ckpt, a.device)
    print(learned_report(state, a.data, a.content))
    if a.learned:
        return
    print("\nAsk it something. /help for the question types, /quit to leave.\n" + HELP)
    while True:
        try:
            line = ask("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        if line in ("/learned", "/topics"):
            print(learned_report(state, a.data, a.content))
        elif line == "/help":
            print(HELP)
        else:
            print(answer(model, state, line, a.temperature))


if __name__ == "__main__":
    main()
