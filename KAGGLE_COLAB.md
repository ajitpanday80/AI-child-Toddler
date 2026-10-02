# Running the school on Kaggle or Google Colab

Free GPUs have limits (Kaggle: ~30 GPU-hours/week; Colab: sessions end when idle or after a few hours).
The school is built for that: training runs in **parts** (default **60 minutes**). At the end of a part it saves
everything and asks *"Continue with the next part?"*. Say `n` (or just stop) and nothing is lost; the next session resumes
at exactly the same stage. Promotion to the next grade is a separate question, and it is always yours to answer.

What must survive between sessions: the `runs/school/` folder (model + progress). Downloads in `data/` are re-fetched if lost (it just takes a few minutes).

---------------------------------------------------------------------------

## Kaggle

### Setup (every new session)
1. kaggle.com -> **Code -> New Notebook**.
2. Right panel -> **Session options**: **Accelerator = GPU (T4 x2 / P100)** and **Internet = On** (needs a phone-verified account).
3. **Cell 1: get the code** (works the first time and every time after; the repo is public):
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
   `!ls` should show `content  KAGGLE_COLAB.md  README.md  school  tests`. (PyTorch is already installed on Kaggle.)
   - Always `%cd /kaggle/working` *before* deleting `school_repo`; deleting the folder you are standing in breaks every later command (`getcwd` errors). If that happens: **Run -> Restart session**, then run Cell 1 again.
   - If the clone asks for a GitHub username, the repo is private: make it public (GitHub -> Settings -> General -> Change visibility). Or use a read-only token kept in **Add-ons -> Secrets** (`GITHUB_TOKEN`) and clone with `https://{tok}@github.com/...`, then run `!git -C school_repo remote set-url origin https://github.com/ajitpanday80/ai-child-toddler`. Never paste the token in chat.
4. *(Optional)* your own material: create a Kaggle **Dataset** containing a `content/` folder (`grade1/`, `grade2/`, ... with PDFs, `.txt`, `links.txt`), then **Add Input**. It appears at `/kaggle/input/<dataset-name>/content`.

### Run a part
**Cell 2: first session**
```python
OUT = "/kaggle/working/runs/school"
from school.train import main
main(["--out", OUT, "--part-minutes", "60", "--size", "small"])
# own material? add:  "--content", "/kaggle/input/<dataset-name>/content"
```
**Later sessions: resume** (after adding last session's saved output as an Input, see below)
```python
import os, shutil
OUT = "/kaggle/working/runs/school"
PREV = "/kaggle/input/<your-notebook-or-dataset-name>/runs/school"
if os.path.exists(PREV) and not os.path.exists(OUT):
    shutil.copytree(PREV, OUT)
from school.train import main
main(["--out", OUT, "--part-minutes", "60"])
```
(Find the exact path with `!ls /kaggle/input`.) Use the Python call, **not** `!python ...`: a notebook cell can ask you questions, a `!` command cannot.

What you'll see:
- It prints the model size and `on cuda` (if it says `on cpu`, the GPU is not switched on).
- One `exam` line per 400 steps: `words[...]` = score per subject, `moral=`, `emotion=`, `judgment=`. About 25% is chance level for the 4-choice exams.
- On a pass: `*** Values PASSED ***` and `Move on to ...? [y/n]` -> type the answer in the box under the cell.
- After 60 minutes: `Continue with the next part? [y/n]`. `n` saves and stops (then stop the session to save quota).

Check progress any time: `main(["--out", OUT, "--status"])`

### Carrying progress over (Kaggle forgets `/kaggle/working` unless you save a version)
- At the end of a session click **Save Version -> Save & Run All (Commit)** (or *Quick Save*). The notebook's `/kaggle/working/runs` becomes that version's **Output**.
- Next session: **Add Input -> Your Work / Notebook Output -> pick your notebook**, name it `ai-child-progress` (or edit `PREV` above). The first cell copies it back and training resumes.
- Alternative: **Output tab -> New Dataset** from `runs/school`, and add that dataset as input. Keep the latest `latest.pt` + `state.json` (those two files are all that's needed to resume; the `passed_*.pt` files are keepsake snapshots of each grade).

### Tips for the 30 h/week budget
- 1 part = 1 hour, so ~25 parts a week at most. Use `--part-minutes 30` for shorter parts.
- Stop the session when you answer `n`, so the GPU isn't kept running idle (**Run -> Stop session**). Idle sessions still use quota.
- Unattended "Save & Run All" has no keyboard, so questions are answered as `n`: it trains one part, saves, and ends. Run it again next time. This makes it a good way to run one part without babysitting.
- If a grade isn't passing, loosen all marks a little with `--pass-scale 0.9`, or give it more lessons by adding material in `content/<grade>/`.

---------------------------------------------------------------------------

## Google Colab

1. colab.research.google.com -> **Runtime -> Change runtime type -> T4 GPU**.
2. Cell 1: save progress on Google Drive (survives disconnects):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone -b claude/progressive-model-training-gates-33mehr https://github.com/ajitpanday80/ai-child-toddler /content/school_repo
   %cd /content/school_repo
   !pip -q install pypdf
   ```
3. Cell 2 (run this each session; it resumes on its own because progress lives on Drive):
   ```python
   from school.train import main
   main(["--out", "/content/drive/MyDrive/ai-child/runs",
         "--data", "/content/drive/MyDrive/ai-child/data",
         "--content", "/content/drive/MyDrive/ai-child/content",   # put your grade1/, grade2/ ... folders here
         "--part-minutes", "60"])
   ```
4. Answer the `[y/n]` questions in the box under the cell.

Colab disconnects idle sessions; if that happens, reconnect and run Cell 1 and Cell 2 again. At worst you lose the minutes since the last exam (a save happens at every exam).

---------------------------------------------------------------------------

## Using the trained model
```python
!python -m school.ask "Once upon a time" --ckpt /kaggle/working/runs/school/latest.pt
```
Use `--ckpt .../passed_0_Values.pt` to hear the model as it was when it passed a particular stage.

## Troubleshooting
| Problem | Fix |
|---|---|
| Clone asks for `Username for 'https://github.com'` | The repo is private: see the step 3 note above. |
| `could not download ...` | Internet is Off (Kaggle) or the site is slow. Turn it On, re-run: finished downloads are cached. |
| The question never appears | You ran it with `!python`. Use `from school.train import main; main([...])`. |
| Session ended in the middle of a lesson | Re-run. It resumes from the last exam (saved each time). |
| `shape '[32, 256, 6, 42]' is invalid` | Old code (fixed). **Run -> Restart session**, run Cell 1 (it pulls the fix), then run again. |
| After any update (`git pull`) nothing changed | Python keeps old code in memory: **Run -> Restart session**, then Cell 1, then your run cell. |
| `! N of M ... pages could not be downloaded` | Wikipedia rate-limited the notebook. Safe: finished pages are kept. Stop (`n`), wait a few minutes, run again; missing pages are fetched then. Lines like `mind:10` in the lesson summary (vs. ~100-300) mean this happened. |
| Out of memory | `--batch 16`, or `--size tiny`/`small`. |
| Stuck at a stage for many exams | Read the per-subject scores in the log. A low subject needs more material in `content/<grade>/<subject>/`, or use `--pass-scale 0.9`. |
