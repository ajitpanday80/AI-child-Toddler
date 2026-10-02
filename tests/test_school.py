import json
import os

import torch

from school import train
from school.data import Doc, Source, load_source, load_user_content, parse_fables, paragraphs, chunk, html_to_text
from school.exams import cloze_paper, cloze_score, label_paper, label_score, vocabulary, moral_paper
from school.model import GPT, ModelConfig


def tiny():
    return GPT(ModelConfig(vocab_size=128, n_layer=1, n_head=2, n_embd=32, block_size=256))


W = ['cat', 'dog', 'hen', 'cow', 'pig', 'owl', 'bee']
FABLES = ("INTRO\n\nSTART OF BOOK\n\n" + "THE FOX AND THE GRAPES\n\n" + "A hungry fox saw some grapes hanging high on a vine. " * 6 +
          "\n\n_It is easy to despise what you cannot get._\n\n")


def test_fable_parsing_extracts_story_and_moral():
    d = parse_fables(FABLES)
    assert len(d) == 1 and d[0].moral.startswith("It is easy") and "Moral:" in d[0].text and d[0].subject == "values"


def test_exam_docs_never_overlap_study_docs():
    docs = [Doc(f"paragraph number {i} about cats", f"k{i}") for i in range(200)]
    held = [d for d in docs if d.is_exam()]
    assert 20 < len(held) < 70 and not {d.key for d in held} & {d.key for d in docs if not d.is_exam()}


def test_long_paragraph_is_chunked():
    assert all(len(c) <= 950 for c in chunk("This is a sentence. " * 200))


def test_html_to_text_skips_scripts():
    t = html_to_text("<html><script>bad()</script><p>Hello kids</p><p>Be kind</p></html>")
    assert "bad" not in t and "Hello kids" in t


def test_user_content_txt_and_subject_folder(tmp_path):
    g = tmp_path / "grade1"
    (g / "math").mkdir(parents=True)
    (g / "story.txt").write_text("The little hen found a seed and planted it in the garden today.")
    (g / "math" / "add.md").write_text("Two plus two is four. Three plus three is six, said the teacher to the class.")
    docs = load_user_content(str(tmp_path), "grade1", str(tmp_path / "cache"), lambda *_: None)
    assert {d.subject for d in docs} == {"extra", "math"}


def test_user_content_pdf(tmp_path):
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(200, 200)
    (tmp_path / "grade2").mkdir()
    with open(tmp_path / "grade2" / "blank.pdf", "wb") as f:
        w.write(f)
    assert load_user_content(str(tmp_path), "grade2", str(tmp_path / "c"), lambda *_: None) == []   # no text, no crash


def test_exam_papers_deterministic_and_untrained_near_chance():
    docs = [Doc(f"The quick brown fox number {i % 7} jumps over the lazy dog and runs away into the green forest.", f"x{i}") for i in range(300)]
    held = [d for d in docs if d.is_exam()]
    p1, p2 = cloze_paper(held, vocabulary(docs)), cloze_paper(held, vocabulary(docs))
    assert p1 == p2 and p1 and all(len(c) == 4 for _, c, _ in p1)
    assert cloze_score(tiny(), p1) < 0.7


def test_label_exam():
    docs = [Doc(f"Situation: a{i}\nIs it wrong? yes", f"a{i}", "judgment", task="judgment", prompt=f"Situation: a{i}\nIs it wrong?",
                answer=" yes" if i % 2 else " no", choices=[" yes", " no"]) for i in range(60)]
    p = label_paper([d for d in docs if d.is_exam()], "judgment")
    assert p and 0 <= label_score(tiny(), p) <= 1


def test_moral_paper_needs_four_options():
    fab = [Doc("t", f"f{i}", "values", story="s" * 300, moral=f"moral {i}") for i in range(6)]
    assert all(len(c) == 4 for _, c in moral_paper(fab))


def test_part_budget_and_resume_and_gate(tmp_path, monkeypatch):
    """Offline end-to-end: a tiny school with one stage. Pass -> asked; answer 'n' -> stops; rerun asks again; 'y' finishes."""
    from school import stages
    from school.stages import Stage
    text = "\n\n".join(f"The {W[i % 7]} sat on the {W[(i * 3) % 7]} number {i} and the {W[(i * 5) % 7]} ran in the park with a {W[(i * 2) % 7]} today." for i in range(400))
    st = [Stage("One", "g1", "test", [Source("text", text, "language")], cloze_pass=0.0), Stage("Two", "g2", "test", [Source("text", text, "language")], cloze_pass=0.0)]
    monkeypatch.setattr(train, "STAGES", st)
    out = str(tmp_path / "run")
    base = ["--out", out, "--data", str(tmp_path / "d"), "--content", str(tmp_path / "c"), "--size", "tiny", "--steps", "2", "--batch", "2", "--device", "cpu"]
    asked = []
    def no(p): asked.append(p); return "n"
    train.main(base, ask=no)
    s = json.load(open(os.path.join(out, "state.json")))
    assert s["phase"] == "awaiting_approval" and s["stage"] == 0 and "One" in s["baselines"] and "Move on to Two" in asked[0]
    train.main(base, ask=no)                       # resume: asks again, still no
    assert len(asked) == 2 and json.load(open(os.path.join(out, "state.json")))["stage"] == 0
    train.main(base + ["--part-minutes", "100"], ask=lambda p: "y")
    s = json.load(open(os.path.join(out, "state.json")))
    assert s["stage"] == 2 and set(s["baselines"]) == {"One", "Two"}


def test_part_ends_and_asks_to_continue(tmp_path, monkeypatch):
    from school.stages import Stage
    text = "\n\n".join(f"The {W[i % 7]} sat on the {W[(i * 3) % 7]} number {i} and the {W[(i * 5) % 7]} ran in the park with a {W[(i * 2) % 7]} today." for i in range(400))
    monkeypatch.setattr(train, "STAGES", [Stage("One", "g1", "t", [Source("text", text)], cloze_pass=1.1)])   # unpassable
    out = str(tmp_path / "run")
    asked = []
    train.main(["--out", out, "--data", str(tmp_path / "d"), "--content", str(tmp_path / "c"), "--size", "tiny", "--steps", "2", "--batch", "2",
                "--device", "cpu", "--part-minutes", "0"], ask=lambda p: asked.append(p) or "n")
    assert "Continue with the next part" in asked[0]
    assert json.load(open(os.path.join(out, "state.json")))["parts"] == 1


def test_every_model_size_is_valid_and_runs():
    for name, (L, H, C) in train.SIZES.items():
        assert C % H == 0, name
        m = GPT(ModelConfig(vocab_size=128, n_layer=1, n_head=H, n_embd=C))   # same heads/width as the real size
        assert m(torch.zeros(2, 16, dtype=torch.long)).shape == (2, 16, 128)


def test_stuck_stage_asks_to_move_on_with_best_version(tmp_path, monkeypatch):
    from school.stages import Stage
    text = "\n\n".join(f"The {W[i % 7]} sat on the {W[(i * 3) % 7]} number {i} and the {W[(i * 5) % 7]} ran in the park with a {W[(i * 2) % 7]} today." for i in range(400))
    mk = lambda n, f: Stage(n, f, "t", [Source("text", text)], cloze_pass=1.1)       # can never pass
    monkeypatch.setattr(train, "STAGES", [mk("One", "g1"), Stage("Two", "g2", "t", [Source("text", text)], cloze_pass=0.0)])
    out, asked = str(tmp_path / "run"), []
    def ask(p):
        asked.append(p)
        return "y" if "anyway" in p else "n"
    train.main(["--out", out, "--data", str(tmp_path / "d"), "--content", str(tmp_path / "c"), "--size", "tiny", "--steps", "2", "--batch", "2",
                "--device", "cpu", "--patience", "2", "--part-minutes", "100"], ask=lambda p: (asked.append(p), "y" if "anyway" in p else "n")[1])
    s = json.load(open(os.path.join(out, "state.json")))
    assert any("anyway" in q for q in asked) and s["stage"] >= 1 and "One" in s["baselines"]


def test_label_exam_is_balanced_so_guessing_scores_chance():
    docs = [Doc(f"s{i}", f"k{i}", "judgment", task="judgment", prompt=f"P{i}", answer=" no" if i % 10 else " yes", choices=[" yes", " no"]) for i in range(2000)]
    p = label_paper([d for d in docs if d.is_exam()], "judgment")
    gold = [c[g] for _, c, g in p]
    assert gold.count(" yes") == gold.count(" no") > 20       # 90% of the data says "no", the exam does not


def test_chat_shows_learned_and_answers_every_command(tmp_path, capsys):
    from school import chat
    m = tiny()
    st = {"stage": 1, "phase": "studying", "attempt": 0, "step": 5, "parts": 0, "best_q": -1.0, "since_best": 0, "history": [],
          "baselines": {"Values": {"cloze": 0.47, "judgment": 0.55, "cloze:values": 0.6, "cloze:mind": 0.3, "moral": 0.2}}}
    out = str(tmp_path / "run")
    os.makedirs(out)
    train.save_state(out, m, torch.optim.AdamW(m.parameters()), st)
    lines = iter(["/learned", "The sun is", "/feel i lost my toy", "/wrong I took his lunch", "/word The dog ran to the ___ | park | tree | moon",
                  "/word broken", "/moral A fox saw grapes. He could not reach them.", "/nonsense", "/quit"])
    chat.main(["--out", out, "--data", str(tmp_path / "d"), "--device", "cpu"], ask=lambda p: next(lines))
    txt = capsys.readouterr().out
    assert "WHAT IT HAS LEARNED SO FAR" in txt and "Values" in txt and "PASSED" in txt and "WEAK" in txt   # mind=30% is flagged
    assert "sadness" in txt and "not wrong" in txt and "park" in txt and "Use:  /word" in txt and "Moral:" in txt and "Unknown command" in txt
    assert "Pre-Nursery" in txt          # the stage it is working on is listed too
