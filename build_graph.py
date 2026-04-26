#!/usr/bin/env python3
"""
Embed every fetched article and link similar ones with [[wiki-links]] so the
articles dir becomes an Obsidian-compatible vault.

Steps:
  1. Find all body.md files under ~/.ft-bookmarks/articles/<tweetId>/
  2. Parse frontmatter + body, build a slug from author + title
  3. Chunk body and embed with sentence-transformers (all-MiniLM-L6-v2),
     mean-pool to one vector per article
  4. Compute pairwise cosine similarity, pick top-K neighbors above threshold
  5. Rename body.md -> <slug>.md and append a "## Related" section with [[links]]
  6. Write INDEX.md listing all articles by date

Open ~/.ft-bookmarks/articles in Obsidian afterwards. The graph view shows
emergent clusters; no rigid categories.

Idempotent: re-running just regenerates the related section + INDEX.
"""

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

ARTICLES_DIR = Path.home() / ".ft-bookmarks" / "articles"
EMBED_CACHE = ARTICLES_DIR / ".embeddings.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 400               # rough; the model truncates to 256 anyway
TOP_K = 6                        # neighbors per article
SIM_THRESHOLD = 0.35             # below this, skip the link

RELATED_MARK_BEGIN = "<!-- related-begin -->"
RELATED_MARK_END = "<!-- related-end -->"


# ── Frontmatter & slug ──────────────────────────────────────────────────────

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n(.*)", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str, str]:
    """Returns (parsed_dict, raw_frontmatter_block_with_delimiters, body)."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, "", text
    raw_block = f"---\n{m.group(1)}\n---\n"
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        # only top-level keys; don't try to parse nested/multi-line values
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*:", line):
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            try:
                v = json.loads(v) if v.startswith('"') else v[1:-1]
            except json.JSONDecodeError:
                v = v[1:-1]
        fm[k.strip()] = v
    return fm, raw_block, m.group(2)


SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(title: str, max_len: int = 60) -> str:
    s = SLUG_RE.sub("-", (title or "").lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0]
    return s or "untitled"


def make_slug(fm: dict, tweet_id: str) -> str:
    handle = (fm.get("tweet_author_handle") or "anon").strip().lower()
    title = fm.get("title") or fm.get("source_url") or tweet_id
    return f"{handle}-{slugify(title)}"


# ── Article model ───────────────────────────────────────────────────────────

class Article:
    __slots__ = ("tweet_id", "folder", "fm", "fm_raw", "body", "slug", "vec")

    def __init__(self, tweet_id: str, folder: Path, fm: dict, fm_raw: str, body: str):
        self.tweet_id = tweet_id
        self.folder = folder
        self.fm = fm
        self.fm_raw = fm_raw  # original frontmatter block, preserved verbatim
        self.body = body
        self.slug = make_slug(fm, tweet_id)
        self.vec: np.ndarray | None = None

    def display_title(self) -> str:
        return self.fm.get("title") or self.slug


# Pick up both freshly-fetched (body.md) and already-renamed (<slug>.md) articles
def load_articles_any() -> list[Article]:
    seen_dirs = set()
    arts: list[Article] = []
    for md in sorted(ARTICLES_DIR.glob("*/*.md")):
        if md.parent in seen_dirs:
            continue
        seen_dirs.add(md.parent)
        text = md.read_text()
        fm, fm_raw, body = parse_frontmatter(text)
        tweet_id = md.parent.name
        a = Article(tweet_id, md.parent, fm, fm_raw, body)
        # If already-renamed, file name is <slug>.md not body.md
        if md.stem != "body":
            a.slug = md.stem
        arts.append(a)
    return arts


# Resolve unique slugs across the corpus (handles author posting many articles)
def dedupe_slugs(arts: list[Article]) -> None:
    counts = defaultdict(int)
    used: dict[str, int] = {}
    for a in arts:
        used[a.slug] = used.get(a.slug, 0) + 1
    seen: dict[str, int] = {}
    for a in arts:
        if used[a.slug] > 1:
            seen[a.slug] = seen.get(a.slug, 0) + 1
            a.slug = f"{a.slug}-{seen[a.slug]}"


# ── Embedding ──────────────────────────────────────────────────────────────

WORD_RE = re.compile(r"\S+")


def chunk_text(text: str, chunk_tokens: int = CHUNK_TOKENS,
               overlap: int = 50) -> list[str]:
    words = WORD_RE.findall(text)
    if not words:
        return []
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_tokens])
        chunks.append(chunk)
        if i + chunk_tokens >= len(words):
            break
        i += chunk_tokens - overlap
    return chunks


def strip_related_section(body: str) -> str:
    """Remove a previous '## Related' block when re-running."""
    i = body.find(RELATED_MARK_BEGIN)
    if i == -1:
        # backwards compat — strip any trailing "## Related" section
        m = re.search(r"\n## Related\n", body)
        return body[: m.start()] if m else body
    return body[:i].rstrip() + "\n"


def embed_articles(arts: list[Article], force: bool) -> None:
    if EMBED_CACHE.exists() and not force:
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        ids = cache["tweet_ids"]
        vecs = cache["vectors"]
        cached = {tid: vecs[i] for i, tid in enumerate(ids.tolist())}
        if all(a.tweet_id in cached for a in arts):
            for a in arts:
                a.vec = cached[a.tweet_id]
            print(f"loaded {len(arts)} cached embeddings from {EMBED_CACHE.name}")
            return
        print("cache stale, recomputing")

    print(f"loading model {MODEL_NAME} (first run downloads ~80MB)...")
    model = SentenceTransformer(MODEL_NAME)

    # Build chunks per article and remember offsets so we can mean-pool
    all_chunks: list[str] = []
    offsets: list[tuple[int, int]] = []
    for a in arts:
        body_for_embed = strip_related_section(a.body)
        chunks = chunk_text(body_for_embed)
        if not chunks:
            chunks = [a.display_title()]
        start = len(all_chunks)
        all_chunks.extend(chunks)
        offsets.append((start, len(all_chunks)))

    print(f"embedding {len(all_chunks)} chunks across {len(arts)} articles...")
    vecs = model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for a, (s, e) in zip(arts, offsets):
        pooled = vecs[s:e].mean(axis=0)
        # re-normalize pooled vector (mean of unit vectors isn't unit)
        n = np.linalg.norm(pooled)
        a.vec = pooled / n if n > 0 else pooled

    # Cache
    np.savez(
        EMBED_CACHE,
        tweet_ids=np.array([a.tweet_id for a in arts]),
        vectors=np.array([a.vec for a in arts], dtype=np.float32),
    )
    print(f"cached embeddings to {EMBED_CACHE.name}")


# ── Similarity ──────────────────────────────────────────────────────────────

def find_neighbors(arts: list[Article], top_k: int, threshold: float):
    M = np.array([a.vec for a in arts], dtype=np.float32)
    sims = M @ M.T  # cosine since vectors are normalized
    np.fill_diagonal(sims, -1.0)

    neighbors: dict[str, list[tuple[str, float]]] = {}
    for i, a in enumerate(arts):
        order = np.argsort(-sims[i])
        picks: list[tuple[str, float]] = []
        for j in order:
            score = float(sims[i, j])
            if score < threshold:
                break
            if len(picks) >= top_k:
                break
            picks.append((arts[j].tweet_id, score))
        neighbors[a.tweet_id] = picks
    return neighbors


# ── Vault writer ────────────────────────────────────────────────────────────

def humanize_delta(days: int) -> str:
    """Render a signed day count as 'Nmo earlier' / '3wk later' / 'same day'."""
    if days == 0:
        return "same day"
    suffix = "earlier" if days < 0 else "later"
    n = abs(days)
    if n < 7:
        return f"{n}d {suffix}"
    if n < 60:
        return f"{n // 7}wk {suffix}"
    if n < 365 * 2:
        return f"{n // 30}mo {suffix}"
    return f"{n // 365}y {suffix}"


def render_related(this: Article, picks: list[tuple[str, float]],
                   by_id: dict[str, Article]) -> str:
    if not picks:
        return ""
    this_when = parse_when(this)
    lines = [RELATED_MARK_BEGIN, "## Related", ""]
    for tid, score in picks:
        other = by_id[tid]
        other_when = parse_when(other)
        if this_when != datetime.min and other_when != datetime.min:
            delta_days = (other_when - this_when).days
            time_str = humanize_delta(delta_days)
        else:
            time_str = "?"
        lines.append(
            f"- [[{other.slug}]] — {other.display_title()}  ·  {time_str}  ·  ({score:.2f})"
        )
    lines.append(RELATED_MARK_END)
    return "\n".join(lines) + "\n"


def write_vault(arts: list[Article], neighbors) -> None:
    by_id = {a.tweet_id: a for a in arts}
    for a in arts:
        body = strip_related_section(a.body).rstrip() + "\n\n"
        related = render_related(a, neighbors[a.tweet_id], by_id)
        new_text = a.fm_raw + body + related

        target = a.folder / f"{a.slug}.md"
        target.write_text(new_text)
        # Remove the old body.md if we renamed
        old = a.folder / "body.md"
        if old.exists() and old != target:
            old.unlink()


def parse_when(a: Article) -> datetime:
    """Parse a posting timestamp into a sortable datetime."""
    raw = a.fm.get("tweet_posted_at") or ""
    # Twitter format: "Tue Apr 07 17:28:55 +0000 2026"
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")
    except ValueError:
        pass
    raw2 = a.fm.get("article_published") or ""
    # ISO date "2026-04-26"
    try:
        return datetime.fromisoformat(raw2)
    except ValueError:
        return datetime.min


def write_index(arts: list[Article]) -> None:
    by_date = sorted(arts, key=parse_when, reverse=True)
    lines = ["# Articles", "",
             f"_{len(arts)} articles. Open this folder in Obsidian to navigate the graph._", ""]
    for a in by_date:
        when = parse_when(a)
        date_str = when.strftime("%Y-%m-%d") if when != datetime.min else "----------"
        title = a.display_title()
        author = a.fm.get("tweet_author_handle") or "?"
        lines.append(f"- {date_str} · @{author} · [[{a.slug}]] — {title}")
    (ARTICLES_DIR / "INDEX.md").write_text("\n".join(lines) + "\n")


# ── Driver ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-embed", action="store_true",
                    help="recompute embeddings ignoring cache")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--threshold", type=float, default=SIM_THRESHOLD)
    args = ap.parse_args()

    arts = load_articles_any()
    if not arts:
        print("no articles found at", ARTICLES_DIR)
        return

    print(f"found {len(arts)} articles")
    dedupe_slugs(arts)
    embed_articles(arts, force=args.force_embed)
    print(f"computing top-{args.top_k} neighbors (threshold={args.threshold})...")
    neighbors = find_neighbors(arts, args.top_k, args.threshold)

    edge_count = sum(len(v) for v in neighbors.values())
    isolated = sum(1 for v in neighbors.values() if not v)
    print(f"{edge_count} directed edges, {isolated} isolated articles")

    print("writing vault...")
    write_vault(arts, neighbors)
    write_index(arts)
    print(f"done. open {ARTICLES_DIR} in Obsidian.")


if __name__ == "__main__":
    main()
