"""Why are my lessons small?   python -m school.diagnose   (or: from school.diagnose import main; main())
Downloads every Wikipedia page of the first stages WITHOUT using the cache and says what happened to each."""
import tempfile

from . import data
from .stages import STAGES


def main(stage_index=0, subject=None):
    st = STAGES[stage_index]
    tmp = tempfile.mkdtemp()
    print(f"Stage: {st.name}")
    for src in st.sources:
        if src.kind != "wiki" or (subject and src.subject != subject):
            continue
        lang, titles = src.ref
        ok = miss = fail = 0
        for t in titles:
            data.WIKI_ERRORS.pop(t, None)
            n = len(data._wiki_page(lang, t, tmp))
            print(f"    {t:<22} {n:>6} chars", flush=True)
            if t in data.WIKI_ERRORS:
                fail += 1
                print(f"  FAILED  {src.subject:<8} {t:<22} {data.WIKI_ERRORS[t]}")
            elif n < 200:
                miss += 1
                print(f"  EMPTY   {src.subject:<8} {t:<22} ({n} chars: page does not exist, even in full English Wikipedia)")
            else:
                ok += 1
        print(f"{src.subject}: {ok} downloaded, {miss} empty, {fail} failed (of {len(titles)})")


if __name__ == "__main__":
    main()
