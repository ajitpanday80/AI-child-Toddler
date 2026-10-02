import random

from school.curriculum import GRADES, _is_exam, encode, decode
from school.model import GPT, ModelConfig
from school.train import exam, report_card


def test_exam_questions_never_in_lessons():
    rng = random.Random(0)
    for g in (g for g in GRADES if g.holdout):
        exam_prompts = {p for p, _ in g.exam_set()}
        for _ in range(500):
            assert g.sample(rng, exam=False)[0] not in exam_prompts


def test_exam_paper_is_stable():
    assert GRADES[4].exam_set() == GRADES[4].exam_set()


def test_answers_correct():
    rng = random.Random(1)
    for _ in range(50):
        p, a = GRADES[4].sample(rng, False)
        x, y = p[:-1].split("+")
        assert int(x) + int(y) == int(a)


def test_untrained_model_fails_the_gate():
    from school.curriculum import VOCAB
    m = GPT(ModelConfig(vocab_size=len(VOCAB)))
    ok, card = report_card(m, GRADES[0], GRADES)
    assert not ok and card["new"] < 0.5


def test_roundtrip():
    assert decode(encode("12+3=\n")) == "12+3=\n"
