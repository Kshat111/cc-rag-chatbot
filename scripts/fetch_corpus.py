"""Download the Wikipedia corpus into data/raw/ (see docs/DECISIONS.md D-002).

Usage:
    uv run python scripts/fetch_corpus.py --check   # resolve titles only, write nothing
    uv run python scripts/fetch_corpus.py           # fetch, clean, write files + manifest

Re-runs always overwrite: every title is fetched again, each file is replaced
atomically, and manifest.json is rebuilt from scratch. Nothing is ever deleted;
.txt files no longer produced by the title list are reported as stale.
Exits 1 if any title was skipped or failed.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from ragbot import config
from ragbot.ingest import clean_wikipedia_extract

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
_last_request_at = 0.0


class FetchError(Exception):
    pass


def api_get(params):
    """GET the Action API with politeness delay, retries and backoff."""
    global _last_request_at
    query = {"format": "json", "formatversion": "2", "maxlag": config.WIKI_MAXLAG_S, **params}
    url = config.WIKI_API_URL + "?" + urllib.parse.urlencode(query)
    req = urllib.request.Request(url, headers={"User-Agent": config.HTTP_USER_AGENT})

    for attempt in range(config.FETCH_MAX_RETRIES + 1):
        wait = config.FETCH_DELAY_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()
        backoff = 2 ** (attempt + 1)
        try:
            with urllib.request.urlopen(req, timeout=config.FETCH_TIMEOUT_S) as resp:
                data = json.load(resp)
                retry_after = resp.headers.get("Retry-After")
        except urllib.error.HTTPError as e:
            if e.code not in RETRYABLE_HTTP or attempt == config.FETCH_MAX_RETRIES:
                raise FetchError(f"HTTP {e.code}") from e
            time.sleep(_retry_after(e.headers.get("Retry-After"), backoff))
            continue
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == config.FETCH_MAX_RETRIES:
                raise FetchError(f"network error: {e}") from e
            time.sleep(backoff)
            continue

        # maxlag comes back as HTTP 200 with an error body, not as a status code.
        err = data.get("error")
        if err and err.get("code") == "maxlag" and attempt < config.FETCH_MAX_RETRIES:
            time.sleep(_retry_after(retry_after, backoff))
            continue
        if err:
            raise FetchError(f"API error {err.get('code')}: {err.get('info')}")
        return data
    raise FetchError("retries exhausted")


def _retry_after(header, default):
    try:
        return max(float(header), 1.0)
    except (TypeError, ValueError):
        return default


def resolve(title, with_text):
    """Query one title. Returns (page_or_None, notes, problem_or_None)."""
    props = "info|pageprops" + ("|extracts" if with_text else "")
    params = {
        "action": "query",
        "titles": title,
        "redirects": "1",
        "prop": props,
        "inprop": "url",
        "ppprop": "disambiguation",
    }
    if with_text:
        params.update({"explaintext": "1", "exsectionformat": "wiki"})
    q = api_get(params).get("query", {})

    notes = [f"normalized '{n['from']}' -> '{n['to']}'" for n in q.get("normalized", [])]
    for r in q.get("redirects", []):
        notes.append(f"redirect '{r['from']}' -> '{r['to']}'")
        # A redirect to a section of another article is not the topic itself;
        # report it instead of silently fetching the whole target.
        if r.get("tofragment"):
            return None, notes, f"redirects to a section: '{r['to']}#{r['tofragment']}'"

    pages = q.get("pages", [])
    if not pages:
        return None, notes, "no page in response"
    page = pages[0]
    if page.get("missing"):
        return None, notes, "page does not exist"
    if page.get("invalid"):
        return None, notes, f"invalid title: {page.get('invalidreason')}"
    if "disambiguation" in page.get("pageprops", {}):
        return None, notes, f"disambiguation page: '{page['title']}'"
    return page, notes, None


def slugify(title):
    # Strip diacritics so "Chūō" becomes "chuo"; keeps filenames ASCII and stable.
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    # Apostrophes join rather than split, so "San'yō" -> "sanyo", not "san-yo".
    ascii_title = re.sub(r"['’]", "", ascii_title)
    return re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")


def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    # mkstemp creates files as 0600; corpus files are ordinary readable data.
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="resolve titles only; write nothing")
    args = ap.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    docs, problems, seen_ids = [], [], {}

    for requested in config.CORPUS_TITLES:
        try:
            page, notes, problem = resolve(requested, with_text=not args.check)
        except FetchError as e:
            page, notes, problem = None, [], f"FAILED: {e}"

        for n in notes:
            print(f"  note   {requested}: {n}")
        if problem is None and page["pageid"] in seen_ids:
            problem = f"duplicate of '{seen_ids[page['pageid']]}' (same page id)"
        if problem:
            print(f"  SKIP   {requested}: {problem}")
            problems.append((requested, problem))
            continue
        seen_ids[page["pageid"]] = requested

        if args.check:
            print(f"  ok     {requested} -> '{page['title']}' (page {page['pageid']}, rev {page['lastrevid']})")
            continue

        text = clean_wikipedia_extract(
            page["title"],
            page.get("extract", ""),
            config.WIKI_DROP_SECTIONS,
            bare_label_max_words=config.WIKI_BARE_LABEL_MAX_WORDS,
            footnote_markers=config.WIKI_FOOTNOTE_MARKERS,
        )
        body_chars = len(text) - len(f"# {page['title']}\n")
        if body_chars <= 1:
            print(f"  SKIP   {requested}: empty extract after cleaning")
            problems.append((requested, "empty extract after cleaning"))
            continue

        filename = slugify(page["title"]) + ".txt"
        atomic_write(config.RAW_DIR / filename, text)
        doc = {
            "title": page["title"],
            "requested_title": requested,
            "page_id": page["pageid"],
            "revision_id": page["lastrevid"],
            "source_url": page["fullurl"],
            "permalink": f"{page['fullurl'].split('/wiki/')[0]}/w/index.php?oldid={page['lastrevid']}",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "char_count": len(text),
            "file": filename,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "license": config.WIKI_LICENSE,
        }
        docs.append(doc)
        print(f"  ok     {requested} -> {filename} ({len(text):,} chars)")

    if not args.check:
        manifest = {
            "source": config.WIKI_API_URL,
            "license": config.WIKI_LICENSE,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "documents": docs,
        }
        atomic_write(config.MANIFEST_PATH, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        _summary(docs)

        written = {d["file"] for d in docs}
        stale = sorted(p.name for p in config.RAW_DIR.glob("*.txt") if p.name not in written)
        if stale:
            print(f"\nStale files (not deleted, not in manifest): {', '.join(stale)}")

    ok = len(config.CORPUS_TITLES) - len(problems)
    print(f"\n{ok} ok, {len(problems)} skipped/failed of {len(config.CORPUS_TITLES)} titles")
    for requested, problem in problems:
        print(f"  - {requested}: {problem}")
    return 1 if problems else 0


def _summary(docs):
    if not docs:
        return
    by_size = sorted(docs, key=lambda d: d["char_count"])
    total = sum(d["char_count"] for d in docs)
    print(f"\nDocuments: {len(docs)}   Total characters: {total:,}")
    print("Smallest:")
    for d in by_size[:3]:
        print(f"  {d['char_count']:>8,}  {d['title']}")
    print("Largest:")
    for d in reversed(by_size[-3:]):
        print(f"  {d['char_count']:>8,}  {d['title']}")


if __name__ == "__main__":
    sys.exit(main())
