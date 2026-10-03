# Running the school on Kaggle or Google Colab

Free GPUs have limits (Kaggle ~30 GPU-hours/week; sessions end and **erase their working folder**), so:

- Training runs in **parts** (default **60 minutes**). At the end of a part it saves and asks *"Continue with the next part? [y/n]"*.
- Progress is **backed up automatically to a private Hugging Face repo** (free). A new session restores it by itself, so closing a session loses nothing.
- Moving up a grade is a separate question, and it is always yours to answer.

Each code block below is a complete cell: copy it whole into its own cell. Order: **Setup (once), Cell 1, Cell 2**.

---------------------------------------------------------------------------

# One-time setup: the progress backup (Hugging Face)

1. Create a free account at huggingface.co.
2. **Settings -> Access Tokens -> Create new token**, type **Write**. Copy it (you will not see it again).
3. Store it as a secret so it never appears in your notebook:
   - **Kaggle:** in the notebook menu **Add-ons -> Secrets -> Add secret**. Label `HF_TOKEN`, value = the token. Make sure the secret is switched **on for this notebook**.
   - **Colab:** the key icon in the left bar -> **Add new secret**, name `HF_TOKEN`, value = the token, switch **Notebook access** on.
4. Nothing else: the first upload creates a **private** repo called `YOUR_HF_USERNAME/ai-child-progress`. In Cell 2 below, replace `YOUR_HF_USERNAME` with your Hugging Face username.

What the backup does:
- At the start of a run it **restores** the backup if it has more progress than the local folder (`Backup: restored your progress (step N)`).
- During the run it refreshes the backup every 10 minutes and when a stage is passed, a part ends, you answer `n`, or you stop the cell.
- It **never overwrites** a backup that is further along than the current run, and a backup problem never stops training (you see `! Backup failed ...` and it retries).
- You can see the files at `huggingface.co/YOUR_HF_USERNAME/ai-child-progress` (it is private; every upload is also a saved version).

What is and is not backed up:
- **Backed up:** the model, optimizer and progress (`latest.pt`, `state.json`), the best version of the current stage, and a snapshot of every passed stage (`passed_*.pt`). About 20-100 MB in total.
- **Not backed up:** the downloaded lessons (they are public data) and your own `content/` files (keep those in a Kaggle Dataset or your Drive). A new session therefore downloads the lessons again, which takes a few minutes at the start of Cell 2. If Wikipedia or Hugging Face rate-limits the download (`! ... could not be downloaded`), answer `n`, wait a few minutes and run Cell 2 again.
- Your progress is restored after a session ends, but a part that is *in the middle of a lesson* resumes from the last backup (at most about 10 minutes of work is repeated).

---------------------------------------------------------------------------

# Kaggle

## Before you start (every session)
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
!pip -q install pypdf huggingface_hub
!ls
```
Expected: `content  KAGGLE_COLAB.md  README.md  school  tests`.
After any update that Cell 1 pulls: **Run -> Restart session**, then Cell 1 and Cell 2 again (Python keeps old code in memory otherwise).

## Cell 2: train one part (first session AND every later session)
```python
%cd /kaggle/working/school_repo
import os, glob
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")       # your token, from the secret
HF_REPO = "YOUR_HF_USERNAME/ai-child-progress"                            # <- put YOUR Hugging Face username here

OUT = "/kaggle/working/runs/school"                                       # working folder (erased when the session ends)
CONTENT = next(iter(glob.glob("/kaggle/input/*/content")), "content")     # your own PDFs/links dataset, if you added one

from school.train import main
main(["--out", OUT,
      "--content", CONTENT,
      "--hf-repo", HF_REPO,
      "--part-minutes", "60",
      "--size", "small"])          # small ~5M params; "base" ~14M learns more but is slower (size is fixed when a run is first created)
```
What you'll see:
- `Backup: nothing saved there yet; it will be created during training.` (first time) or `Backup: restored your progress (step N).` (later sessions).
- `model: 4.87M parameters on cuda` (if it says `cpu`, the GPU is off in Session options).
- A lesson summary like `values:1023, judgment:3215, mind:150 ...`. Very small numbers or a missing `judgment` mean a download problem: see Troubleshooting.
- One `exam` line every so often: see "How to read the exam line" below.
- On a pass: `*** Values PASSED ***` then `Move on to ...? [y/n]` -> type your answer in the box under the cell.
- After 60 minutes: `Continue with the next part? [y/n]`. `n` saves, backs up and stops. Then stop the session to save quota.

## Cell 3 (any time): report card and what it has learned
```python
%cd /kaggle/working/school_repo
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from school.chat import main
main(["--out", "/kaggle/working/runs/school", "--hf-repo", "YOUR_HF_USERNAME/ai-child-progress", "--learned"])
```

## Cell 4 (any time): ask it questions
Stop the training cell first (or wait for the end of a part): a notebook runs one cell at a time.
```python
%cd /kaggle/working/school_repo
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
from school.chat import main
main(["--out", "/kaggle/working/runs/school", "--hf-repo", "YOUR_HF_USERNAME/ai-child-progress", "--temperature", "0.5"])
```
It first prints **what it has learned so far** (stages passed with scores, the exact topics and books taught, weak subjects, what is not taught yet). Then type questions in the box under the cell:

| You type | What happens |
|---|---|
| `Humpty Dumpty sat on a` (or `/say Humpty Dumpty sat on a`) | It continues the text. Do not type the word `say` without the `/`. |
| `/feel i lost my toy` | Which of 6 feelings fits (sadness, joy, love, anger, fear, surprise), with percentages. |
| `/wrong I took his lunch` | Wrong or not wrong, with percentages. |
| `/word The dog ran to the ___ \| park \| tree \| moon` | Picks the word that fits best. |
| `/moral A fox saw grapes. He could not reach them.` | Writes the moral it draws. |
| `/learned` | Shows the learned-so-far report again. |
| `/quit` | Leave. |

Notes: it is a small model that only continues text; it was never taught to answer questions, so "what is a moon?" gets a story-like ramble. Ask about what it was taught (see the report); for later stages it can only guess, and `/feel` / `/wrong` say so. Percentages near 50% (2 choices) or 17% (6 choices) mean a guess. Lower `--temperature` (0.3) makes it repeat its lessons more faithfully; held-out lessons (20% of every lesson is kept for exams) it has never seen. To talk to the model as it was when it passed a stage, add `"--ckpt", "/kaggle/working/runs/school/passed_0_Values.pt"`.

## Your own material (PDFs, text, links)
1. On your computer make a folder named `content` with `grade1/`, `grade2/` ... inside (see `content/README.md`).
2. Kaggle -> **Datasets -> New Dataset** -> upload that `content` folder.
3. In the notebook: **Add Input -> your dataset**. Cell 2 finds it automatically (`/kaggle/input/<name>/content`).

## Tips for the 30 hours/week
- 1 part = 1 hour, so about 25 parts a week at most. Use `"--part-minutes", "30"` for shorter parts.
- Answer `n` and stop the session when you are done: an idle session still burns quota.
- If a grade does not pass: read the per-subject scores. A low subject needs more material in `content/<grade>/<subject>/`, or ease all marks with `"--pass-scale", "0.9"`.

---------------------------------------------------------------------------

# Google Colab

Setup: **Runtime -> Change runtime type -> T4 GPU**, and do the one-time Hugging Face setup above (secret named `HF_TOKEN`).

## Cell 1: get / update the code
```python
%cd /content
import os
if os.path.isdir("school_repo/.git"):
    !git -C school_repo pull origin claude/progressive-model-training-gates-33mehr
else:
    !rm -rf school_repo
    !git clone -b claude/progressive-model-training-gates-33mehr https://github.com/ajitpanday80/ai-child-toddler school_repo
%cd /content/school_repo
!pip -q install pypdf huggingface_hub
!ls
```

## Cell 2: train one part (first time and every later time)
```python
%cd /content/school_repo
import os
from google.colab import userdata, drive
drive.mount('/content/drive')                                   # optional second copy of your progress on Google Drive

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
HF_REPO = "YOUR_HF_USERNAME/ai-child-progress"                  # <- put YOUR Hugging Face username here
BASE = "/content/drive/MyDrive/ai-child"
os.makedirs(BASE + "/content", exist_ok=True)                   # put your grade1/, grade2/ ... folders in this "content" folder

from school.train import main
main(["--out", BASE + "/runs",
      "--data", BASE + "/data",
      "--content", BASE + "/content",
      "--hf-repo", HF_REPO,
      "--part-minutes", "60",
      "--size", "small"])
```
Answer the `[y/n]` questions in the box under the cell. After a disconnect: reconnect, run Cell 1 and Cell 2 again.

## Cell 3: ask it questions
```python
%cd /content/school_repo
import os
from google.colab import userdata, drive
drive.mount('/content/drive')
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
from school.chat import main
main(["--out", "/content/drive/MyDrive/ai-child/runs", "--data", "/content/drive/MyDrive/ai-child/data",
      "--hf-repo", "YOUR_HF_USERNAME/ai-child-progress", "--temperature", "0.5"])
```
Same questions as on Kaggle (`/say`, `/feel`, `/wrong`, `/word`, `/moral`, `/learned`, `/quit`).

---------------------------------------------------------------------------

# How to read the exam line
```
exam 12 | step 2400 | loss 1.9 | 15s | words[values=58% mind=41%] avg=49% judgment=57% moral=27% bpc=3.1 | need avg>=42% judgment>=54%
```
- `words[subject=..%, ...]`: pick the missing word out of 4, one score per subject (25% = guessing). `avg` is their average; no subject may lag far behind it.
- `judgment` / `emotion`: right-or-wrong and feeling tests on unseen examples with the same number of each answer, so guessing scores only 50% (judgment) or 17% (emotion). `moral`: fable test, shown for information only.
- `loss` is the error on text it is *studying*; `bpc` is the error on text it has *never seen* (lower is better, 2-3 is good). **If `loss` is near 0 but `bpc` stays high or rises, the model is memorising** and cannot pass; training longer makes it worse.
- Few steps per exam when the lessons are small, and stronger regularisation, limit memorising.

# The "Move on anyway?" question
```
Values: no improvement for 8 exams (best so far: ...)
Move on to Pre-Nursery anyway, using the best version so far? [y/n]
```
Appears when the exams stop improving (training longer would only memorise). It has already gone back to its best version.
- **`y`** (usually right): continue with the best version. Later stages re-test earlier ones.
- **`n`**: stop. Add material in `content/<grade>/` (PDFs, links) or ease the marks (`"--pass-scale", "0.9"`), then run Cell 2 again.

# Why is a subject small? (diagnose downloads)
```python
%cd /kaggle/working/school_repo
from school.diagnose import main
main(0, "mind")
```
(`0` = first stage, `"mind"` = the subject.) `FAILED` lines name the error (for example a rate limit): wait a few minutes and re-run Cell 2. `EMPTY` means the page does not exist.

---------------------------------------------------------------------------

# Troubleshooting
| Problem | Fix |
|---|---|
| `No Hugging Face token ... HF_TOKEN` | The secret is missing or not switched on for this notebook (Kaggle: Add-ons -> Secrets; Colab: key icon -> Notebook access). |
| `Hugging Face did not accept the token` | Wrong/expired token or no WRITE access. Create a new **Write** token and update the secret. |
| `! Backup failed (...)` | Network trouble. Training continues and retries. If it never succeeds, check the token and Internet = On. |
| `Backup NOT updated: the backup is further along ...` | This run is behind the backup (for example you started fresh by mistake). Nothing was overwritten. Restart the session and run Cell 2: it restores the backup. |
| `Starting fresh` in a session where you expected progress | Check that Cell 2 shows `Backup: restored ...`. If it says `nothing saved there yet`, the repo name differs from the one you trained with. |
| Clone asks `Username for 'https://github.com'` | The repo is private. Make it public (GitHub -> Settings -> General -> Change visibility). |
| `getcwd` / "folder no longer found" errors | You deleted the folder you were standing in. **Run -> Restart session**, then Cell 1. |
| `fatal: destination path ... already exists` | Use Cell 1 as written: it pulls if the folder exists. |
| Nothing changed after an update | **Run -> Restart session**, Cell 1, Cell 2. |
| `! N of M ... pages could not be downloaded`, or tiny counts like `mind:10` (about 150-200 is normal), or `judgment` missing | Wikipedia / Hugging Face rate-limited the notebook. Finished downloads are kept. Answer `n`, wait a few minutes, run Cell 2 again. Clear an old bad cache with `!rm -rf /kaggle/working/school_repo/data`. |
| `could not download ...` | Kaggle Internet is Off. Turn it on and re-run. |
| The question never appears | You ran it with `!python ...`. Use the Python `main([...])` call as in the cells. |
| Out of memory | Add `"--batch", "16"` to the list in Cell 2, or use `"--size", "tiny"`. |
| `loss` near 0 but scores low / `bpc` high | Memorising (see above). Pull the latest code and start over (below). |
| Stuck at a grade for many exams | More material in `content/<grade>/`, or `"--pass-scale", "0.9"`. |

---------------------------------------------------------------------------

# Clean the old model (start over from nothing)

**This deletes all progress.** Because progress is backed up, you must clear **both** the working folder **and** the backup, otherwise the next run restores the old model.

1. **Run -> Restart session**, then run Cell 1.
2. Clear the working folder, in a cell: `!rm -rf /kaggle/working/runs/school` (Colab: `!rm -rf /content/drive/MyDrive/ai-child/runs`).
3. Clear the backup. Easiest: use a **new repo name** in Cell 2, for example `YOUR_HF_USERNAME/ai-child-progress-2` (the old repo stays as a keepsake). Or on huggingface.co open the repo -> Files -> delete `state.json`, `latest.pt` and the other `.pt` files.
4. Run Cell 2. It should say `Backup: nothing saved there yet`.

Keep a copy first? The `passed_*.pt` files are snapshots of each passed grade (download them from the backup repo if you want them).
