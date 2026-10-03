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


def _args(tmp, extra=()):
    return ["--out", str(tmp / "run"), "--data", str(tmp / "d"), "--content", str(tmp / "c"), "--size", "tiny", "--steps", "2", "--batch", "2",
            "--device", "cpu", *extra]


def _school(monkeypatch, cloze_pass=0.0):
    from school.stages import Stage
    text = "\n\n".join(f"The {W[i % 7]} sat on the {W[(i * 3) % 7]} number {i} and the {W[(i * 5) % 7]} ran in the park with a {W[(i * 2) % 7]} today." for i in range(400))
    monkeypatch.setattr(train, "STAGES", [Stage("One", "g1", "t", [Source("text", text)], cloze_pass=cloze_pass),
                                          Stage("Two", "g2", "t", [Source("text", text)], cloze_pass=cloze_pass)])


def test_backup_restores_progress_after_the_working_folder_is_wiped(tmp_path, monkeypatch):
    """The Kaggle problem: session ends, /kaggle/working is erased. With a backup the next session resumes."""
    import shutil
    _school(monkeypatch)
    bk = tmp_path / "backup"
    train.main(_args(tmp_path, ["--sync-dir", str(bk)]), ask=lambda p: "n")      # session 1: passes stage One, asked to move on, says no
    s1 = json.load(open(tmp_path / "run" / "state.json"))
    assert s1["phase"] == "awaiting_approval" and (bk / "state.json").exists() and (bk / "latest.pt").exists()
    shutil.rmtree(tmp_path / "run")                                             # session ends: everything local is gone
    asked = []
    train.main(_args(tmp_path, ["--sync-dir", str(bk)]), ask=lambda p: (asked.append(p), "n")[1])   # session 2
    s2 = json.load(open(tmp_path / "run" / "state.json"))
    assert s2["step"] == s1["step"] and "One" in s2["baselines"] and "Move on to Two" in asked[0]    # resumed, not fresh


def test_backup_never_overwrites_a_further_along_backup(tmp_path):
    from school.sync import DirBackend, Syncer
    bk = tmp_path / "bk"
    bk.mkdir()
    (bk / "state.json").write_text(json.dumps({"step": 5000}))
    (bk / "latest.pt").write_text("precious")
    out = tmp_path / "out"
    out.mkdir()
    (out / "state.json").write_text(json.dumps({"step": 10}))           # a fresh run by mistake
    (out / "latest.pt").write_text("new")
    msgs = []
    Syncer(str(out), DirBackend(str(bk)), 0, msgs.append).push(True)
    assert (bk / "latest.pt").read_text() == "precious" and any("further along" in m for m in msgs)


def test_backup_failure_does_not_stop_training(tmp_path):
    from school.sync import Syncer
    class Broken:
        def remote_step(self): raise ConnectionError("no network")
        def upload(self, *a): raise ConnectionError("no network")
    msgs = []
    Syncer(str(tmp_path), Broken(), 0, msgs.append).push(True)          # must not raise
    assert any("Backup failed" in m for m in msgs)


def test_hub_backend_needs_a_token():
    import pytest
    from school.sync import HubBackend
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        HubBackend("me/x", None)


def test_exam_scoring_survives_texts_longer_than_the_model_window():
    from school.exams import span_logprob
    m = tiny()                                                    # block_size 256
    long_seq = "x" * 700 + " the answer"
    out = span_logprob(m, [long_seq, "short one"], [len(long_seq) - 10, 3])
    assert len(out) == 2 and all(o == o for o in out)           # no crash, no NaN
    p = [("Feeling: " + "word " * 120 + "\nEmotion:", [" joy", " sadness"], 0)]
    assert 0 <= label_score(m, p) <= 1


def test_emotion_examples_too_long_for_the_window_are_dropped(monkeypatch, tmp_path):
    from school import data
    rows = [{"text": "i feel fine", "label": 1}, {"text": "i feel " + "very " * 80, "label": 0}]
    monkeypatch.setattr(data, "hf_rows", lambda *a, **k: rows)
    docs = data.load_source(Source("emotion", 2), str(tmp_path))
    assert len(docs) == 1 and docs[0].text.startswith("Feeling: i feel fine")


def test_answer_word_of_labelled_examples_gets_extra_loss_weight():
    rows = "Feeling: i am so happy today\nEmotion: joy\n\nSituation: he took her toy\nIs it wrong? yes\n\nplain story text here"
    class D:
        subjects = {"x": (rows, [])}
        def window(self, rng, T):
            return rows[: T + 1]
    x, y, wt = train.make_batch([D()], 0, random.Random(0), 1, len(rows) - 1, 0.0, "cpu", 8.0)
    w = wt[0].tolist()
    target = "".join(chr(int(c)) for c in y[0])
    j = target.index("joy")
    assert all(w[j + k] == 8.0 for k in range(3)) and w[0] == 1.0 and w[-1] == 1.0     # " joy" weighted, ordinary text not
    k = target.index("yes")
    assert w[k] == 8.0


import random


STORY = ("Once a crow was very thirsty. He found a jug with a little water at the bottom, but his beak could not reach it. "
         "He dropped pebbles into the jug one by one until the water rose, and then he drank and flew away happily. ")


def test_user_stories_with_morals_are_parsed(tmp_path):
    from school.data import parse_user_stories
    text = (f"The Thirsty Crow\n\n{STORY}\n\nMoral: Where there is a will, there is a way.\n\n"
            f"The Lion and the Mouse\n\n{STORY}x\n\nThe moral of the story is that even the small can help the great.\n\n"
            f"{STORY}y Lesson - Be patient and think before you act.\n\nA plain paragraph about the weather today that has no moral at all.")
    docs, rest = parse_user_stories(text)
    assert [d.moral for d in docs[:2]] == ["Where there is a will, there is a way.", "that even the small can help the great."]
    assert docs[0].text.startswith("The Thirsty Crow\n\nOnce a crow") and docs[0].text.endswith("Moral: Where there is a will, there is a way.")
    assert docs[0].subject == "values" and docs[0].story.startswith("Once a crow") and "Moral" not in docs[0].story
    assert "plain paragraph" in rest and "Thirsty Crow" not in rest        # stories are not studied twice


def test_short_or_missing_morals_are_ordinary_text():
    from school.data import parse_user_stories
    docs, rest = parse_user_stories("A tiny bit of text.\n\nMoral: Be kind.\n\nMore text follows here that is plain.")
    assert docs == [] and "tiny bit" in rest


def test_user_story_file_feeds_values_and_the_moral_exam(tmp_path):
    (tmp_path / "values").mkdir()
    body = "\n\n".join(f"Story {i}\n\n{STORY} Variation {i} of the tale.\n\nMoral: Lesson number {i} is to keep trying." for i in range(40))
    (tmp_path / "values" / "tales.txt").write_text(body)
    docs = load_user_content(str(tmp_path), "values", str(tmp_path / "cache"), lambda *_: None)
    morals = [d for d in docs if d.moral]
    assert len(morals) == 40 and all(d.subject == "values" for d in morals)
    from school.exams import moral_paper
    held = [d for d in morals if d.is_exam()]
    assert held and all(len(c) == 4 for _, c in moral_paper(held))           # the moral exam can be built from your stories


def test_plain_fables_with_one_line_morals_are_parsed():
    from school.data import parse_plain_fables
    story = ("A Fox saw some Grapes hanging high on a vine and tried again and again to reach them, but at last he gave up and "
             "walked away with his nose in the air, saying that they were surely sour anyway. ")
    long_last = "And so the story simply carries on for a while longer, with many more details about the vineyard and the weather that day, " * 2
    text = ("Aesop's Fables\n\n\nSome introduction text.\n\nIt is only the book title and must not become a fable.\n\n\n"
            f"The Fox and the Grapes\n\n\n{story}\n\nIt is easy to despise what you cannot get.\n\n\n"
            f"The Plain Story\n\n\n{story}\n\n{long_last}\n\n\n")
    docs = parse_plain_fables(text)
    assert [d.moral for d in docs] == ["It is easy to despise what you cannot get."]          # book title and the moral-less story are skipped
    assert docs[0].key == "fable:THE FOX AND THE GRAPES" and docs[0].subject == "values"       # same key as the caps-title edition: same split


def test_plain_fables_source_builds_a_moral_exam(tmp_path):
    from school.stages import Stage
    story = "A Fox saw some Grapes hanging high on a vine and tried again and again to reach them but could not. " * 3
    names = [f"The {a} and the {b}" for a in ("Fox", "Dog", "Cat", "Hen", "Cow", "Pig", "Owl", "Bee") for b in ("Crow", "Wolf", "Lamb", "Mule", "Frog", "Hare", "Toad")]
    book = "\n\n\n".join(f"{n}\n\n\n{story}\n\nKeep on trying and you will win in the end, said the {n.split()[1]}." for n in names)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "pg9999.txt").write_text("*** START OF THE PROJECT GUTENBERG EBOOK X ***\n" + book + "\n*** END OF THE PROJECT GUTENBERG EBOOK X ***")
    st = Stage("T", "t", "t", [Source("gutenberg", 9999, subject="values", plain_fables=True)])
    sd = train.StageData(st, str(tmp_path / "d"), str(tmp_path / "c"))
    assert "moral" in sd.papers and sd.counts["values"][0] > 30
