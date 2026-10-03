"""Keep training progress safe outside the notebook (Kaggle/Colab erase the working folder when a session ends).

    --hf-repo you/ai-child-progress     back up to a PRIVATE Hugging Face repo (token in env HF_TOKEN)
    --sync-dir /some/folder             or copy to any folder (e.g. a mounted Google Drive)

At the start of a run the backup is restored if it has MORE progress than the local folder.
During the run it is refreshed every --sync-minutes, after each passed stage, at the end of each part, and when you stop.
It never overwrites a backup that is further along than what is local, and a backup problem never stops training.
"""
import glob
import json
import os
import shutil
import time

PATTERNS = ["state.json", "latest.pt", "best_current_stage.pt", "passed_*.pt"]


def local_step(folder):
    try:
        return json.load(open(os.path.join(folder, "state.json"))).get("step", 0)
    except Exception:
        return 0


class DirBackend:
    """Backup to a plain folder (also used by the tests)."""

    def __init__(self, path):
        self.path = path

    def remote_step(self):
        f = os.path.join(self.path, "state.json")
        return json.load(open(f)).get("step", 0) if os.path.exists(f) else None

    def download(self, dest):
        for pat in PATTERNS:
            for f in glob.glob(os.path.join(self.path, pat)):
                shutil.copy2(f, os.path.join(dest, os.path.basename(f)))

    def upload(self, src, message):
        os.makedirs(self.path, exist_ok=True)
        for pat in PATTERNS:
            for f in glob.glob(os.path.join(src, pat)):
                shutil.copy2(f, self.path + "/" + os.path.basename(f) + ".tmp")
                os.replace(self.path + "/" + os.path.basename(f) + ".tmp", os.path.join(self.path, os.path.basename(f)))


class HubBackend:
    """Backup to a private Hugging Face model repo."""

    def __init__(self, repo, token):
        try:
            import huggingface_hub as hf
        except ImportError:
            raise RuntimeError("Backup needs:  pip install huggingface_hub")
        if not token:
            raise RuntimeError("No Hugging Face token. Set the HF_TOKEN environment variable "
                               "(Kaggle: Add-ons -> Secrets; Colab: the key icon). See KAGGLE_COLAB.md.")
        self.hf, self.repo, self.token = hf, repo, token

    def remote_step(self):
        from huggingface_hub.utils import RepositoryNotFoundError
        api = self.hf.HfApi(token=self.token)
        try:
            api.whoami()                         # a wrong/expired token must stop the run, not look like "no backup"
        except Exception as e:
            raise RuntimeError(f"Hugging Face did not accept the token in HF_TOKEN ({type(e).__name__}). "
                               "Create a new token with WRITE access and store it as the secret HF_TOKEN.")
        try:
            files = api.list_repo_files(self.repo)
        except RepositoryNotFoundError:
            return None                          # repo not created yet (the first upload creates it)
        if "state.json" not in files:
            return None
        f = self.hf.hf_hub_download(self.repo, "state.json", token=self.token, force_download=True)
        return json.load(open(f)).get("step", 0)

    def download(self, dest):
        self.hf.snapshot_download(self.repo, local_dir=dest, token=self.token, allow_patterns=PATTERNS, force_download=True)

    def upload(self, src, message):
        api = self.hf.HfApi(token=self.token)
        api.create_repo(self.repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=src, repo_id=self.repo, allow_patterns=PATTERNS, commit_message=message)


class Syncer:
    def __init__(self, out, backend, minutes=10, log=print):
        self.out, self.backend, self.minutes, self.log = out, backend, minutes, log
        self.last, self.warned, self.ok = time.time(), False, True

    def pull(self):
        """Call BEFORE loading the model. Fails loudly if the backup cannot be read, so we never start fresh by mistake."""
        os.makedirs(self.out, exist_ok=True)
        remote, local = self.backend.remote_step(), local_step(self.out)
        if remote is None:
            self.log("Backup: nothing saved there yet; it will be created during training.")
        elif remote > local:
            self.backend.download(self.out)
            self.log(f"Backup: restored your progress (step {remote}).")
        else:
            self.log(f"Backup: local progress (step {local}) is as new as the backup (step {remote}).")

    def push(self, force=False, why="progress"):
        if not force and time.time() - self.last < self.minutes * 60:
            return
        try:
            remote = self.backend.remote_step()
            if remote is not None and remote > local_step(self.out):
                if not self.warned:
                    self.log(f"Backup NOT updated: the backup is further along (step {remote}) than this run. Nothing was overwritten.")
                self.warned = True
                return
            self.backend.upload(self.out, f"{why} (step {local_step(self.out)})")
            self.last = time.time()
        except Exception as e:                      # never let a backup problem stop the training
            if not self.warned:
                self.log(f"! Backup failed ({type(e).__name__}: {str(e)[:120]}). Training continues; it will retry.")
            self.warned = True


def make_syncer(args, log=print):
    if getattr(args, "hf_repo", None):
        backend = HubBackend(args.hf_repo, os.environ.get("HF_TOKEN"))
    elif getattr(args, "sync_dir", None):
        backend = DirBackend(args.sync_dir)
    else:
        return None
    return Syncer(args.out, backend, args.sync_minutes, log)
