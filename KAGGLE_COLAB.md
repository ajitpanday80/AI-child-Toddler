# Running the school on Kaggle or Google Colab

Free GPUs have limits (Kaggle ~30 GPU-hours/week, Colab disconnects idle sessions), so training runs in **parts**
(default **60 minutes**). At the end of a part it saves everything and asks *"Continue with the next part? [y/n]"*.
Answer `n` (or just stop) and nothing is lost: the next session resumes at the same stage.
Moving up a grade is a separate question, and it is always yours to answer.

**Only one thing must survive between sessions: the `runs/school/` folder** (model + progress). Downloads in `data/` are re-fetched if lost.

Each cell below is complete: copy it whole into its own cell and run. Run them in order: **Cell 1, then Cell 2**.

---------------------------------------------------------------------------

# Kaggle

## Before you start (once per session)
Right panel -> **Session options**: **Accelerator = GPU (T4 x2 or P100)**, **Internet = On** (needs a phone-verified account).

## Cell 1: get / update the code
```python
%cd /kaggle/working
import os
if os.path.isdir("school_repo/.git"):
    !git -C school_repo pull origin claude/progressive-model-training-gates-33mehr
else:
    !rm -rf school_repo
    !git clone -b claude/progressive-model-training-gates-33mehr https://github.com/ajitpanday80/ai-child-toddler school_repo
%cd /kaggle/working/school_repo
!pip -q install pypdf
!ls
```
Expected: `content  KAGGLE_COLAB.md  README.md  school  tests`.
After any update that Cell 1 pulls: **Run -> Restart session**, then run Cell 1 and Cell 2 again (Python keeps old code in memory otherwise).

## Cell 2: train one part (first session AND every later session)
```python
%cd /kaggle/working/school_repo
import os, glob, shutil

OUT = "/kaggle/working/runs/school"                      # progress is saved here
CONTENT = next(iter(glob.glob("/kaggle/input/*/content")), "content")   # your own PDFs/links dataset if you added one

# Resume: if last session's saved progress was added as an Input, copy it in (only when nothing is here yet)
if not os.path.exists(OUT + "/state.json"):
    found = glob.glob("/kaggle/input/**/state.json", recursive=True)
    if found:
        src = os.path.dirname(found[0])
        shutil.copytree(src, OUT, dirs_exist_ok=True)
        print("Resumed progress from", src)
    else:
        print("Starting fresh (no saved progress found).")

from school.train import main
main(["--out", OUT,
      "--content", CONTENT,
      "--part-minutes", "60",
      "--size", "small"])          # small ~5M params; "base" ~14M learns more but is slower (size is fixed when a run is first created)
```
What you'll see:
- `model: 4.87M parameters on cuda` (if it says `cpu`, the GPU is off in Session options).
- A lesson summary like `values:1178, mind:150 ...`. Very small numbers (for example `mind:10`) mean Wikipedia refused downloads: see Troubleshooting.
- One `exam` line per 400 steps: `words[...]` = score per subject, `moral=`, `emotion=`, `judgment=`. About 25% is chance level for the 4-choice words/moral exams.
- On a pass: `*** Values PASSED ***` then `Move on to Pre-Nursery ...? [y/n]` -> type your answer in the box under the cell.
- After 60 minutes: `Continue with the next part? [y/n]`. Type `n` to stop and save quota.

## Start over from nothing (optional)
After a big code update (for example when the training was memorising), restart the school from the beginning. This deletes saved progress, so run it only when you mean it:
```python
!rm -rf /kaggle/working/runs/school
```
Also remove the old progress Input (or it will be copied back in Cell 2).

## Cell 3 (any time): report card
```python
%cd /kaggle/working/school_repo
from school.train import main
main(["--out", "/kaggle/working/runs/school", "--status"])
```

## Cell 4 (any time): let the model write
```python
%cd /kaggle/working/school_repo
!python -m school.ask "Once upon a time" --ckpt /kaggle/working/runs/school/latest.pt
```
Use `--ckpt /kaggle/working/runs/school/passed_0_Values.pt` to hear the model as it was when it passed that stage.

## Keeping progress between Kaggle sessions
Kaggle deletes `/kaggle/working` when the session ends **unless you save a version**:
1. When you are done (after answering `n`): **Save Version -> Save & Run All (Commit)** (or Quick Save). The `runs/` folder becomes that version's Output. *(Save & Run All re-runs your cells with no keyboard: questions are auto-answered `n`, so it trains one part and ends. If you only want to save, use Quick Save.)*
2. **Run -> Stop session** so the GPU stops using your quota.
3. Next session: **Add Input -> Your Work -> (notebook with the saved output)**, then run Cell 1 and Cell 2. Cell 2 finds `state.json` in your input and copies it back automatically.
   Alternative: Output tab -> **New Dataset** from the `runs/school` folder, then add that dataset as input.
   Only `latest.pt` + `state.json` are needed to resume; `passed_*.pt` files are keepsake snapshots of each grade.

## Your own material (PDFs, text, links)
1. On your computer make a folder named `content` with `grade1/`, `grade2/` ... inside (see `content/README.md`).
2. Kaggle -> **Datasets -> New Dataset** -> upload that `content` folder.
3. In the notebook: **Add Input -> your dataset**. Cell 2 finds it automatically (`/kaggle/input/<name>/content`).

## Tips for the 30 hours/week
- 1 part = 1 hour, so about 25 parts a week at most. Use `"--part-minutes", "30"` for shorter parts.
- Stop the session when you answer `n`: an idle session still burns quota.
- If a grade does not pass: read the per-subject scores in the log. A low subject needs more material in `content/<grade>/<subject>/`, or make all marks easier with `"--pass-scale", "0.9"` (add to the list in Cell 2).

---------------------------------------------------------------------------

# Google Colab

Setup: **Runtime -> Change runtime type -> T4 GPU**. Progress is kept on your Google Drive, so it survives disconnects.

## Cell 1: connect Drive and get / update the code
```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
import os
if os.path.isdir("school_repo/.git"):
    !git -C school_repo pull origin claude/progressive-model-training-gates-33mehr
else:
    !rm -rf school_repo
    !git clone -b claude/progressive-model-training-gates-33mehr https://github.com/ajitpanday80/ai-child-toddler school_repo
%cd /content/school_repo
!pip -q install pypdf
!ls
```

## Cell 2: train one part (first time and every later time; it resumes by itself)
```python
%cd /content/school_repo
import os
BASE = "/content/drive/MyDrive/ai-child"                 # everything is kept here
os.makedirs(BASE + "/content", exist_ok=True)            # put your grade1/, grade2/ ... folders in this "content" folder

from school.train import main
main(["--out", BASE + "/runs",
      "--data", BASE + "/data",
      "--content", BASE + "/content",
      "--part-minutes", "60",
      "--size", "small"])
```
Answer the `[y/n]` questions in the box under the cell. If Colab disconnects: reconnect, run Cell 1 and Cell 2 again. At worst you lose the minutes since the last exam (saved at every exam).

To start over from nothing (deletes saved progress): `!rm -rf /content/drive/MyDrive/ai-child/runs`

## Cell 3: report card
```python
%cd /content/school_repo
from school.train import main
main(["--out", "/content/drive/MyDrive/ai-child/runs", "--status"])
```

## Cell 4: let the model write
```python
%cd /content/school_repo
!python -m school.ask "Once upon a time" --ckpt /content/drive/MyDrive/ai-child/runs/latest.pt
```

---------------------------------------------------------------------------

# How to read the exam line
```
exam 12 | step 2400 | loss 1.9 | 15s | words[values=58% mind=41%] avg=49% judgment=57% moral=27% bpc=3.1 | need avg>=50% judgment>=56%
```
- `words[subject=..%, ...]`: pick the missing word out of 4, one score per subject (25% = guessing). `avg` is their average; no subject may lag far behind the average.
- `judgment` / `emotion`: right-or-wrong and feeling tests on unseen examples, with the same number of each answer, so guessing scores only 50% (judgment) or 17% (emotion). `moral`: the fable test is shown for information only (the model has too few fables to learn it).
- `loss` is the error on text it is *studying*; `bpc` is the error on text it has *never seen* (lower is better, 2-3 is good). **If `loss` is near 0 but `bpc` stays high or rises, the model is memorising.** It then cannot pass, and training longer makes it worse.
- To limit memorising, each exam now uses few steps when the lessons are small, and the model has stronger regularisation.
- If there is no improvement for 8 exams (`--patience`), it goes back to its best version, explains why, and asks `Move on ... anyway? [y/n]`. Answer `y` to accept the best version and continue, or `n` to stop and add material (`content/<grade>/`) or ease the marks (`"--pass-scale", "0.9"`).

# The "Move on anyway?" question
```
Values: no improvement for 8 exams (best so far: ...)
Move on to Pre-Nursery anyway, using the best version so far? [y/n]
```
It appears when the exams stop improving. Training longer would only make the model memorise (watch `loss` falling while `bpc` stays flat). It has already gone back to its best version.
- **`y`** (usually right): continue to the next stage with the best version. Later stages re-test earlier ones, so what it learned is kept.
- **`n`**: stop. Add material in `content/<grade>/` (PDFs, links) or ease the marks (`"--pass-scale", "0.9"`), then run Cell 2 again.
A low subject (for example `mind`) is nearly always a lack of material, not of training time.

# Why is a subject small? (diagnose downloads)
Shows, page by page, which Wikipedia downloads worked (run it after Cell 1; `0` = the first stage, `"mind"` = the subject):
```python
%cd /kaggle/working/school_repo
from school.diagnose import main
main(0, "mind")
```
`FAILED` lines name the error (for example a rate limit): wait a few minutes and re-run Cell 2. `EMPTY` means the page does not exist.

# Troubleshooting
| Problem | Fix |
|---|---|
| Clone asks `Username for 'https://github.com'` | The repo is private. Make it public (GitHub -> Settings -> General -> Change visibility). |
| `getcwd` / "folder no longer found" errors | You deleted the folder you were standing in. **Run -> Restart session**, then Cell 1. |
| `fatal: destination path ... already exists` | Use Cell 1 as written: it pulls if the folder exists. |
| `shape '[32, 256, 6, 42]' is invalid` | Old code (fixed). Restart session, Cell 1, Cell 2. |
| Nothing changed after an update | Python keeps old code in memory: **Run -> Restart session**, Cell 1, Cell 2. |
| `! N of M ... pages could not be downloaded`, or tiny counts like `mind:10` (about 150-200 is normal) | Wikipedia rate-limited the notebook, or an old failed download is cached: clear it with `!rm -rf /kaggle/working/school_repo/data` and run Cell 2 again. Finished pages are kept. Answer `n`, wait a few minutes, run Cell 2 again: only missing pages are fetched. |
| `could not download ...` | Kaggle Internet is Off. Turn it on and re-run (finished downloads are cached). |
| The question never appears | You ran it with `!python ...`. Use the Python `main([...])` call as in Cell 2. |
| Out of memory | Add `"--batch", "16"` to the list in Cell 2, or use `"--size", "tiny"`. |
| `loss` near 0 but scores low / `bpc` high | Memorising small lessons (see above). Pull the latest code, restart the session, and start over (`!rm -rf /kaggle/working/runs/school`). |
| Stuck at a grade for many exams | See the tips above: more material, or `"--pass-scale", "0.9"`. |

---------------------------------------------------------------------------

# Clean the old model (start over from nothing)

Use this when the model went wrong (for example it memorised), or after a big code update. **It deletes all saved progress.**

**Kaggle**
1. **Run -> Restart session** (so old code is dropped from memory).
2. Run **Cell 1** (gets the latest code).
3. Run this cell once:
   ```python
   !rm -rf /kaggle/working/runs/school
   ```
4. If you added an earlier saved output as an **Input** (right panel), remove it. Otherwise Cell 2 copies the old model back in.
5. Run **Cell 2**. It prints `Starting fresh (no saved progress found).`

**Colab**
```python
!rm -rf /content/drive/MyDrive/ai-child/runs
```
Then run Cell 1 and Cell 2. (Downloaded lessons in `.../ai-child/data` are kept, so it does not re-download them.)

Keep a copy first? Download or copy `latest.pt` and `state.json` before deleting. `passed_*.pt` files are snapshots from each passed grade.
