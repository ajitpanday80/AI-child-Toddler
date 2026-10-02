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

UA = {"User-Agent": "Mozilla/5.0 (ai-child-school; educational model training)"}
SMART = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " "}
SUBJECTS = ["values", "language", "math", "science", "history", "civics", "mind", "heart", "judgment", "extra"]
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
def _download(url: str, headers=None, tries=4) -> bytes:
    err = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=90).read()
        except Exception as e:  # the network is flaky; back off and retry
            err = e
            if getattr(e, "code", 0) in (404, 403):
                break
            time.sleep(2 ** i)
    raise RuntimeError(f"could not download {url}: {err}")


def _cached(cache_dir: str, name: str, getter) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9._-]", "_", name))
    if not os.path.exists(path):
        open(path, "wb").write(getter())
    return open(path, "rb").read().decode("utf8", "ignore")


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


def _wiki_page(lang: str, title: str, cache_dir: str) -> str:
    def fetch(lg):
        url = (f"https://{lg}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&redirects=1&format=json&titles="
               + urllib.parse.quote(title))
        def get():
            d = json.loads(_download(url))
            return "\n".join(p.get("extract", "") for p in d["query"]["pages"].values()).encode()
        try:
            return _cached(cache_dir, f"wiki_{lg}_{title}.txt", get)
        except Exception:
            return ""
    text = fetch(lang)
    return text if len(text) > 200 or lang == "en" else fetch("en")   # not in Simple English? use the full article


def _clean_wiki(t: str) -> str:
    t = re.split(r"\n==+ ?(References|Other websites|Related pages|External links|See also|Notes|Sources|Further reading)", t)[0]
    return re.sub(r"\n==+[^\n]*==+", "\n", t)


def hf_rows(dataset: str, config: str, split: str, n: int, cache_dir: str) -> List[dict]:
    def get():
        rows = []
        for off in range(0, n, 100):
            u = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset)}&config={config}"
                 f"&split={split}&offset={off}&length=100")
            rows += [r["row"] for r in json.loads(_download(u))["rows"]]
        return json.dumps(rows).encode()
    return json.loads(_cached(cache_dir, f"hf_{dataset}_{config}_{split}_{n}.json", get))


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
        with ThreadPoolExecutor(8) as ex:
            pages = list(ex.map(lambda t: _wiki_page(lang, t, cache_dir), titles))
        docs = [Doc(p, p) for pg in pages for p in paragraphs(to_ascii(_clean_wiki(pg)))]
    elif src.kind == "emotion":     # tweets labelled with one of 6 emotions
        docs = []
        for r in hf_rows("dair-ai/emotion", "split", "train", src.ref, cache_dir):
            pr, ans = f"Feeling: {r['text']}\nEmotion:", " " + EMOTIONS[r["label"]]
            docs.append(Doc(pr + ans, r["text"], "heart", task="emotion", prompt=pr, answer=ans, choices=[" " + e for e in EMOTIONS]))
    elif src.kind == "ethics":      # short everyday scenarios: is the action wrong?
        docs = []
        for r in hf_rows("hendrycks/ethics", "commonsense", "train", src.ref, cache_dir):
            if len(r["input"]) <= 160:
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
