# AI-child-Toddler: a model that grows up in school

A GPT-style model (own weights, random start, no pretrained model) that is raised like a child:
**Values -> Pre-Nursery -> Nursery -> Grade 1 ... Grade 6** (extendable to adulthood).
It may only move up after passing an exam, and **you** decide whether it moves up.

```
study a lesson -> exam -> passed? -> "Move on to Grade 2? [y/n]"
                                      y: next grade     n: stop (run again later -> asked again)
every 60 min: part ends -> saved -> "Continue with the next part? [y/n]"   (built for Kaggle / Colab)
```

## What it learns in every grade (all subjects, at the grade's level)
| Subject | Examples of the content (all fetched automatically) |
|---|---|
| values | Aesop's fables, Fifty Famous Stories, honesty / kindness / fairness / justice / ethics, moral development |
| language | nursery rhymes, TinyStories, McGuffey's graded readers 1-6 |
| math | counting, +/-, fractions, algebra, geometry (Simple/English Wikipedia) |
| science | nature, weather, forces, energy, chemistry, biology, physics |
| history | early civilisations to revolutions and world wars |
| civics | rules, rights, democracy, constitutions, political science |
| mind | psychology: emotions, empathy, motivation, cognitive development |
| heart | **labelled feelings** (`dair-ai/emotion`): "Feeling: i feel lost... Emotion: sadness" |
| judgment | **right or wrong?** (`hendrycks/ethics`): "Situation: ... Is it wrong? yes/no" (from Grade 4) |
| extra | **your own files** in `content/<grade>/` (PDF, text, web links) - see `content/README.md` |

## The exams (held-out questions the model never studied)
- **Words**: pick the missing word out of 4 look-alikes, separately **per subject**. The average must reach the mark and no subject may lag far behind.
- **Moral** (Values stage): read an unseen fable, pick its true moral out of 4.
- **Emotion** and **Right/wrong**: classify unseen labelled examples.
- **Review**: every earlier grade is re-examined and must not slip (no forgetting).
- Printed per exam, e.g. `words[val=62% mat=48% sci=55% ...] avg=54% moral=47% emotion=52%`.

## Start
```
pip install torch pypdf
python -m school.train                      # trains in 60-minute parts, asks before moving on
python -m school.train --status             # report card
python -m school.ask "Once upon a time"     # let it write
```
Options: `--part-minutes 60`, `--size tiny|small|base`, `--steps 400` (between exams), `--pass-scale 0.9` (all marks 10% easier), `--out`, `--data`, `--content`.
**Kaggle / Colab: see [KAGGLE_COLAB.md](KAGGLE_COLAB.md).**

## Change the curriculum
Everything is in `school/stages.py`: each `Stage` lists its sources and pass marks. To go beyond Grade 6, append stages
(e.g. `Stage("Grade 7", "grade7", "...", [wiki("math", "Algebra|...", EN), ...])`) and make a `content/grade7/` folder.

## Honest limits
This is a small model trained on small data. It learns language, facts and the *patterns* of feelings and right/wrong from
examples; it does not "feel" or truly reason morally. The exams measure whether it picks the human-labelled answer more often
than chance. Pass marks are starting values: read the exam lines and tune with `--pass-scale` or `stages.py`.
