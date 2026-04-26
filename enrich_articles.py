#!/usr/bin/env python3
"""
Fetch the linked article for each bookmark and store it as markdown.

Phase 1: external links only (trafilatura). x.com/i/article/* is deferred to
Phase 2 because it requires authenticated GraphQL.

Output:
    ~/.ft-bookmarks/articles/<tweetId>/body.md
    ~/.ft-bookmarks/articles/<tweetId>/images/<hash>.<ext>
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

BOOKMARKS_PATH = Path.home() / ".ft-bookmarks" / "bookmarks.jsonl"
ARTICLES_DIR = Path.home() / ".ft-bookmarks" / "articles"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Hosts where the link is not a readable article. Skip silently.
SKIP_HOSTS = {
    "youtube.com", "m.youtube.com", "youtu.be",
    "drive.google.com",          # PDFs — separate path later
    "claude.ai", "claude.com",   # chat URLs, not articles
    "x.com", "twitter.com",      # bare tweet links — not what we're after here
}

MAX_IMAGE_BYTES = 5 * 1024 * 1024


def is_x_article(url: str) -> bool:
    return "/i/article/" in url and ("x.com" in url or "twitter.com" in url)


def normalize_host(url: str) -> str:
    h = (urlparse(url).netloc or "").lower()
    return h[4:] if h.startswith("www.") else h


def classify(url: str):
    if is_x_article(url):
        return "x_article"
    if normalize_host(url) in SKIP_HOSTS:
        return "skip"
    return "external"


def image_filename(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:12]
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if not ext or len(ext) > 5 or not re.match(r"^\.[a-z0-9]+$", ext):
        ext = ".img"
    return f"{h}{ext}"


def download_image(url: str, dest_dir: Path) -> str | None:
    fname = image_filename(url)
    dest = dest_dir / fname
    if dest.exists():
        return fname
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15, stream=True)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("content-type", "").lower()
        if "image" not in ctype and not fname.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg")):
            return None
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    f.close()
                    dest.unlink(missing_ok=True)
                    return None
                f.write(chunk)
        return fname
    except requests.RequestException:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return None


IMG_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def localize_images(markdown: str, source_url: str, images_dir: Path) -> tuple[str, int]:
    images_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        alt, src = m.group(1), m.group(2)
        abs_src = urljoin(source_url, src)
        if urlparse(abs_src).scheme not in ("http", "https"):
            return m.group(0)
        fname = download_image(abs_src, images_dir)
        if fname:
            count += 1
            return f"![{alt}](./images/{fname})"
        return m.group(0)

    return IMG_PATTERN.sub(repl, markdown), count


def fetch_external(url: str) -> dict:
    html = trafilatura.fetch_url(url, no_ssl=True)
    if not html:
        raise RuntimeError("fetch failed (no html)")
    body = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_images=True,
        include_links=True,
        favor_recall=True,
    )
    if not body or len(body.strip()) < 80:
        raise RuntimeError(f"extract returned thin content ({len(body or '')} chars)")
    meta = trafilatura.extract_metadata(html, default_url=url)
    return {
        "title": (getattr(meta, "title", None) or "").strip() or None,
        "site": (getattr(meta, "sitename", None) or normalize_host(url)) or None,
        "author": getattr(meta, "author", None),
        "published": getattr(meta, "date", None),
        "body": body,
    }


def yaml_value(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ":#\n\"'") or s.strip() != s:
        return json.dumps(s, ensure_ascii=False)
    return s


def write_article(out_dir: Path, frontmatter: dict, body: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for k, v in frontmatter.items():
        if v is None:
            continue
        lines.append(f"{k}: {yaml_value(v)}")
    lines.append("---")
    lines.append("")
    title = frontmatter.get("title") or "(untitled)"
    lines.append(f"# {title}")
    lines.append("")
    (out_dir / "body.md").write_text("\n".join(lines) + body + "\n")


def process_bookmark(b: dict, force: bool, only_kind: str | None):
    links = b.get("links") or []
    if not links:
        return None
    url = links[0]
    kind = classify(url)
    if only_kind and kind != only_kind:
        return None  # silently skip — wrong kind for this run

    tweet_id = b["id"]
    out_dir = ARTICLES_DIR / tweet_id
    body_path = out_dir / "body.md"

    if kind == "skip":
        return ("skip", normalize_host(url), url)
    if kind == "x_article":
        return ("deferred", "phase 2", url)
    if body_path.exists() and not force:
        return ("cached", "exists", url)

    try:
        article = fetch_external(url)
    except Exception as e:
        return ("error", str(e)[:80], url)

    body_md, image_count = localize_images(article["body"], url, out_dir / "images")
    fm = {
        "tweet_id": tweet_id,
        "tweet_url": b["url"],
        "source_url": url,
        "source_kind": "external",
        "title": article["title"],
        "site": article["site"],
        "article_author": article["author"],
        "article_published": article["published"],
        "tweet_author_handle": b.get("authorHandle"),
        "tweet_posted_at": b.get("postedAt"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_count": image_count,
        "body_chars": len(body_md),
    }
    write_article(out_dir, fm, body_md)
    return ("ok", f"{image_count} imgs, {len(body_md)} chars", url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument("--tweet-id", help="process a single bookmark by id")
    ap.add_argument("--limit", type=int, help="cap number of bookmarks processed")
    ap.add_argument("--kind", choices=["external", "x_article"], default="external",
                    help="which link kind to process (default: external)")
    args = ap.parse_args()

    bookmarks = [json.loads(line) for line in BOOKMARKS_PATH.open()]
    bookmarks = [b for b in bookmarks if b.get("links")]
    if args.tweet_id:
        bookmarks = [b for b in bookmarks if b["id"] == args.tweet_id]
    if args.limit:
        bookmarks = bookmarks[: args.limit]

    sym = {"ok": "[ok]", "cached": "[--]", "skip": "[skip]",
           "error": "[ERR]", "deferred": "[def]"}
    counts = {k: 0 for k in sym}

    for b in bookmarks:
        result = process_bookmark(b, force=args.force, only_kind=args.kind)
        if result is None:
            continue
        status, info, url = result
        counts[status] += 1
        print(f"{sym[status]} {b['id']} @{b.get('authorHandle','?'):20} {url[:55]:55}  {info}")

    print()
    print("summary:", " ".join(f"{k}={v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
