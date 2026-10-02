# AI-child-Toddler — a model that goes to school

A small GPT (≈0.8M params) trained **from scratch** (random weights, no pretrained model) under a
strict grade system: it only moves up a grade after passing an exam.

```
train N steps → exam → passed? → next grade
                     ↘ no → keep studying (up to --max-attempts, then the run halts: "held back")
```

## Rules of the school
- **Pass mark**: ≥95% exact-match on the grade's exam (`Grade.pass_mark`).
- **No forgetting**: the same exam also re-tests *every earlier grade* at ≥90% (`review_mark`). Lessons also mix in old material (`--review-frac`).
- **Held-out exams**: 20% of questions are hash-reserved for exams and never used in lessons, so a pass means generalisation, not memorisation. (Grade 4, 100 addition "facts", is the exception: `holdout=False`.)
- **Resumable**: state saved after every exam (`runs/school/latest.pt`); a snapshot is kept for each passed grade (`gradeN_passed.pt`). `report_card.json` logs every exam.

## Grades (edit `school/curriculum.py` to add your own)
1 copy a word · 2 reverse a word · 3 next number · 4 add single digits · 5 add two-digit numbers · 6 sort digits

## Use
```
pip install torch pytest
python -m school.train --out runs/school          # ~10 min on 4 CPU cores
python -m school.ask runs/school/latest.pt "27+45=" "rev abc="
python -m pytest tests
```
Knobs: `--steps-per-lesson` (steps between exams), `--max-attempts`, `--lr`, `--review-frac`, `--grades`.

To add a grade: write a generator returning `(prompt, answer)` (prompt ends in `=`), append a `Grade(...)` to `GRADES`.
