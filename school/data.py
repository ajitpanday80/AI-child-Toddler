"""Fetch teaching material (internet + your own folders), clean it, and cut it into documents.

Everything downloaded is cached under data/, so a lesson is fetched only once.
"""
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import List, Optional

UA = {"User-Agent": "ai-child-school/1.0 (https://github.com/ajitpanday80/ai-child-toddler; educational model training)"}
SMART = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "}
SUBJECTS = ["values", "language", "math", "science", "history", "civics", "mind", "heart", "judgment", "extra"]
MATURE = re.compile(r"\b(sex\w*|rape\w*|porn\w*|naked|nude|kill\w*|murder\w*|suicide|drug\w*|cocaine|heroin|weed|gun\w*|shoot\w*|stab\w*|bomb\w*|drunk|alcohol\w*|beer|wine|vodka|cigarette\w*|stripper|prostitut\w*|whore|slut|bitch|fuck\w*|shit|damn|bastard|abuse\w*|molest\w*|terror\w*)\b", re.I)
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


@dataclass
class Doc:
    text: str                          # what the model reads while studying
    key: str                           # identity used to decide study-vs-exam (hash)
    subject: Optional[str] = None      # one of SUBJECTS; filled from the Source when not set
    story: Optional[str] = None        # fables: the story ...
    moral: Optional[str] = None        # ... and its moral (moral exam)
    task: Optional[str] = None         # "emotion" | "judgment": a labelled example (label exam)
    prompt: Optional[str] = None
    answer: Optional[str] = None
    choices: Optional[List[str]] = None

    def is_exam(self) -> bool:
        return int(hashlib.md5(self.key.encode()).hexdigest(), 16) % 5 == 0   # 20% held back, never studied


@dataclass
class Source:
    kind: str                          # gutenberg | tinystories | wiki | emotion | ethics | text
    ref: object = None                 # book id | (start, nbytes) | (lang, [titles]) | n rows | raw text
    subject: str = "language"
    verse: bool = False                # keep line breaks (rhymes)
    fables: bool = False               # parse "TITLE / story / _moral_" structure


# ---------------------------------------------------------------- low-level helpers
def _download(url: str, headers=None, tries=5) -> bytes:
    err = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=90).read()
        except Exception as e:  # the network is flaky; back off and retry
            err = e
            code = getattr(e, "code", 0)
            if code in (404, 403):
                break
            time.sleep(6 * (i + 1) if code in (429, 503) else 2 ** i)      # rate-limited: wait longer
    raise RuntimeError(f"could not download {url}: {err}")


MISSING = b"\x00missing"


def _cached(cache_dir: str, name: str, getter) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9._-]", "_", name))
    if not os.path.exists(path) or os.path.getsize(path) == 0:       # an empty file is never trusted (old failed downloads)
        data = getter()
        if data:
            open(path, "wb").write(data)
        return "" if data in (b"", MISSING) else data.decode("utf8", "ignore")
    raw = open(path, "rb").read()
    return "" if raw == MISSING else raw.decode("utf8", "ignore")


def to_ascii(t: str) -> str:
    for a, b in SMART.items():
        t = t.replace(a, b)
    return unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()


def strip_gutenberg(t: str) -> str:
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", t)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", t)
    return t[s.end(): e.start() if e else None] if s else t


def chunk(p: str, maxlen: int = 900) -> List[str]:
    """Split a long paragraph into sentence-aligned pieces (PDF/web text is often one giant block)."""
    if len(p) <= maxlen:
        return [p]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", p):
        if cur and len(cur) + len(s) > maxlen:
            out.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
    return out + [cur] if cur else out


def paragraphs(t: str, verse: bool = False) -> List[str]:
    out = []
    for p in re.split(r"\n\s*\n", t.replace("\r", "")):
        lines = [l.strip() for l in p.split("\n") if l.strip()]
        p = ("\n" if verse else " ").join(lines)
        p = re.sub(r"\[[^\]]*\]", "", p)           # [Illustration: ...]
        p = re.sub(r"[ \t]+", " ", p.replace("_", "")).strip()
        if len(p) >= 25 and not re.fullmatch(r"[A-Z0-9 ,.'-]+", p):   # drop headings / page furniture
            out += chunk(p) if not verse else [p]
    return out


# ---------------------------------------------------------------- source kinds
def parse_fables(t: str) -> List[Doc]:
    """Aesop-for-Children layout: ALL-CAPS title, blank, story paragraphs, then _moral_ paragraph(s)."""
    t = re.sub(r"\[[^\]]*\]", "", t)
    blocks = re.split(r"\n\s*\n", t.replace("\r", ""))
    is_title = lambda b: re.fullmatch(r"[A-Z][A-Z ,'-]{4,60}", b.strip())
    docs, i = [], 0
    while i < len(blocks):
        if is_title(blocks[i]):
            title, j, story, morals = blocks[i].strip(), i + 1, [], []
            while j < len(blocks) and not is_title(blocks[j]):
                b = " ".join(blocks[j].split())
                if b:
                    (morals if b.startswith("_") and b.endswith("_") else story).append(b.replace("_", ""))
                j += 1
            story_t = " ".join(story)
            if morals and len(story_t) > 200:
                moral = " ".join(morals)
                docs.append(Doc(f"{title.title()}\n\n{story_t}\n\nMoral: {moral}", "fable:" + title, "values", story_t, moral))
            i = j
        else:
            i += 1
    return docs


WIKI_FAILED: List[str] = []
WIKI_ERRORS: dict = {}          # title -> last error (for school.diagnose)


def wiki_cached(cache_dir: str, lang: str, title: str) -> bool:
    """True if this page was really downloaded (Simple English, or the English fallback)."""
    for lg in {lang, "en"}:
        f = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9._-]", "_", f"wiki_{lg}_{title}.txt"))
        if os.path.exists(f) and os.path.getsize(f) > 200:
            return True
    return False


def _wiki_page(lang: str, title: str, cache_dir: str) -> str:
    def fetch(lg):
        url = (f"https://{lg}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles="
               + urllib.parse.quote(title))
        def get():
            pages = json.loads(_download(url))["query"]["pages"].values()
            text = "\n".join(p.get("extract", "") for p in pages).encode()
            return text or (MISSING if any("missing" in p for p in pages) else b"")   # remember only pages Wikipedia says do not exist
        try:
            return _cached(cache_dir, f"wiki_{lg}_{title}.txt", get)
        except Exception as e:
            WIKI_ERRORS[title] = f"{type(e).__name__}: {e}"[:160]
            WIKI_FAILED.append(title)          # network trouble (not "page missing"): not cached, retried next run
            return ""
    text = fetch(lang)
    return text if len(text) > 200 or lang == "en" else fetch("en")   # not in Simple English? use the full article


def _clean_wiki(t: str) -> str:
    t = re.split(r"\n==+ ?(References|Other websites|Related pages|External links|See also|Notes|Sources|Further reading)", t)[0]
    return re.sub(r"\n==+[^\n]*==+", "\n", t)


def hf_rows(dataset: str, config: str, split: str, n: int, cache_dir: str) -> List[dict]:
    """Rows from the Hugging Face datasets server, 100 per request. Every batch is cached the moment it arrives,
    so a rate limit (HTTP 429) in the middle costs nothing: the next run continues where this one stopped."""
    rows = []
    for off in range(0, n, 100):
        def get(off=off):
            u = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset)}&config={config}"
                 f"&split={split}&offset={off}&length=100")
            return json.dumps([r["row"] for r in json.loads(_download(u, tries=7))["rows"]]).encode()
        t0 = time.time()
        try:
            rows += json.loads(_cached(cache_dir, f"hf_{dataset}_{config}_{split}_{off}.json", get))
        except RuntimeError as e:
            print(f"  ! {dataset}: stopped at row {off} of {n} ({str(e)[-60:]}). Using the {len(rows)} rows loaded so far; "
                  "run again later to fetch the rest.", flush=True)
            break
        if time.time() - t0 > 0.05:
            time.sleep(0.3)                             # a real download happened: be gentle with the server
    return rows


def load_source(src: Source, cache_dir: str = "data") -> List[Doc]:
    if src.kind == "gutenberg":
        i = src.ref
        raw = _cached(cache_dir, f"pg{i}.txt", lambda: _download(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"))
        text = to_ascii(strip_gutenberg(raw))
        docs = parse_fables(text) if src.fables else [Doc(p, p) for p in paragraphs(text, src.verse)]
    elif src.kind == "tinystories":
        start, n = src.ref
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
        raw = _cached(cache_dir, f"tinystories_{start}_{n}.txt", lambda: _download(url, {"Range": f"bytes={start}-{start + n - 1}"}))
        docs = [Doc(to_ascii(s), s) for s in (x.strip() for x in raw.split("<|endoftext|>")[1:-1]) if len(s) > 80]
    elif src.kind == "wiki":
        lang, titles = src.ref
        WIKI_FAILED.clear()
        with ThreadPoolExecutor(3) as ex:                      # gentle: Wikipedia rate-limits busy addresses
            pages = list(ex.map(lambda t: _wiki_page(lang, t, cache_dir), titles))
        for _ in range(2):                                      # second chance for pages that failed on the network
            retry = sorted(set(WIKI_FAILED))
            if not retry:
                break
            WIKI_FAILED.clear()
            time.sleep(10)
            for t in retry:
                pages[titles.index(t)] = _wiki_page(lang, t, cache_dir)
        if WIKI_FAILED:
            print(f"  ! {len(set(WIKI_FAILED))} of {len(titles)} {src.subject} pages could not be downloaded "
                  "(network/rate limit); re-run later to fetch them", flush=True)
        docs = [Doc(p, p) for pg in pages for p in paragraphs(to_ascii(_clean_wiki(pg)))]
    elif src.kind == "emotion":     # tweets labelled with one of 6 emotions
        docs = []
        for r in hf_rows("dair-ai/emotion", "split", "train", src.ref, cache_dir):
            pr, ans = f"Feeling: {r['text']}\nEmotion:", " " + EMOTIONS[r["label"]]
            docs.append(Doc(pr + ans, r["text"], "heart", task="emotion", prompt=pr, answer=ans, choices=[" " + e for e in EMOTIONS]))
    elif src.kind == "ethics":      # short everyday scenarios: is the action wrong?
        docs = []
        for r in hf_rows("hendrycks/ethics", "commonsense", "train", src.ref, cache_dir):
            if len(r["input"]) <= 160 and not MATURE.search(r["input"]):
                pr, ans = f"Situation: {to_ascii(r['input'])}\nIs it wrong?", " yes" if r["label"] else " no"
                docs.append(Doc(pr + ans, r["input"], "judgment", task="judgment", prompt=pr, answer=ans, choices=[" yes", " no"]))
    elif src.kind == "text":
        docs = [Doc(p, p) for p in paragraphs(to_ascii(src.ref), src.verse)]
    else:
        raise ValueError(src.kind)
    for d in docs:
        d.subject = d.subject or src.subject
    return docs


# ---------------------------------------------------------------- your own content: content/<stage>/[subject/]files
class _Text(HTMLParser):
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "div", "br", "tr"}

    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer", "header"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, d):
        if not self.skip:
            self.parts.append(d)


def html_to_text(h: str) -> str:
    p = _Text()
    p.feed(h)
    return "".join(p.parts)


def pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("PDF support needs:  pip install pypdf")
    return "\n\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)


def fetch_url_text(url: str, cache_dir: str) -> str:
    key = hashlib.md5(url.encode()).hexdigest()
    path = os.path.join(cache_dir, f"url_{key}.txt")
    if not os.path.exists(path):
        raw = _download(url)
        text = pdf_to_text(raw) if raw[:5] == b"%PDF-" else html_to_text(raw.decode("utf8", "ignore"))
        os.makedirs(cache_dir, exist_ok=True)
        open(path, "w").write(text)
    return open(path).read()


def load_user_content(content_dir: str, folder: str, cache_dir: str = "data", log=print) -> List[Doc]:
    root = os.path.join(content_dir, folder)
    docs = []
    if not os.path.isdir(root):
        return docs
    for dp, _, files in sorted(os.walk(root)):
        rel = os.path.relpath(dp, root)
        subject = rel.split(os.sep)[0] if rel.split(os.sep)[0] in SUBJECTS else "extra"
        for f in sorted(files):
            path, ext = os.path.join(dp, f), f.lower().rsplit(".", 1)[-1]
            texts = []
            try:
                if ext == "pdf":
                    texts.append(pdf_to_text(open(path, "rb").read()))
                elif ext in ("txt", "md", "urls", "url"):
                    body = open(path, encoding="utf8", errors="ignore").read()
                    lines = [l.strip() for l in body.splitlines() if l.strip() and not l.strip().startswith("#")]
                    if lines and all(re.match(r"https?://", l) for l in lines):     # a list of web links / PDF links
                        for u in lines:
                            try:
                                texts.append(fetch_url_text(u, cache_dir))
                            except Exception as e:
                                log(f"  ! could not read {u}: {e}")
                    else:
                        texts.append(body)
            except Exception as e:
                log(f"  ! skipped {path}: {e}")
            for t in texts:
                docs += [Doc(p, p, subject) for p in paragraphs(to_ascii(t))]
    return docs
