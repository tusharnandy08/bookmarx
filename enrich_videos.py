#!/usr/bin/env python3
"""
Trace Twitter video clips back to their YouTube originals (for lectures).

Pipeline (per video bookmark):
  1. Classify & extract — Haiku decides is_lecture and pulls
     {speaker, institution, topic, year_hint} from the tweet text.
  2. yt-dlp search — ytsearch10 with a query built from the extracted fields.
     Filtered to candidates ≥ 20 min.
  3. Rank — Haiku picks the best candidate (or "none confident") given the
     original tweet text.

Output: ~/.ft-bookmarks/videos/traces.jsonl
  One newline-delimited JSON record per attempted bookmark. Idempotent —
  re-runs skip rows already traced.

Usage:
  python enrich_videos.py                  # full corpus
  python enrich_videos.py --tweet-id ...   # one bookmark
  python enrich_videos.py --limit 10       # first N video bookmarks
  python enrich_videos.py --force          # re-trace already-traced rows
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import anthropic
from yt_dlp import YoutubeDL

BOOKMARKS_PATH = Path.home() / ".ft-bookmarks" / "bookmarks.jsonl"
VIDEOS_DIR = Path.home() / ".ft-bookmarks" / "videos"
TRACES_PATH = VIDEOS_DIR / "traces.jsonl"

MODEL = "claude-haiku-4-5"
MIN_LECTURE_DURATION_SEC = 20 * 60  # 20 min — anything shorter isn't a lecture
SEARCH_K = 10
RANK_K = 5
CONFIDENCE_THRESHOLD = 0.7  # below this, status -> low_confidence

client = anthropic.Anthropic()
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


# ── Bookmark filtering ──────────────────────────────────────────────────────

def is_video_bookmark(b: dict) -> bool:
    for m in b.get("mediaObjects") or []:
        if m.get("type") == "video" and m.get("videoVariants"):
            return True
    return False


# ── Stage 1: classify + extract ─────────────────────────────────────────────

CLASSIFY_TOOL = {
    "name": "submit_classification",
    "description": "Submit your classification of this tweet's attached video.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_lecture": {
                "type": "boolean",
                "description": (
                    "True if the video is a clip of a long-form lecture, talk, seminar, "
                    "course session, conference keynote, or academic colloquium that likely "
                    "exists as a full recording on YouTube. False for memes, product demos, "
                    "interviews unless framed as a talk, podcast clips unless from a recorded "
                    "lecture series, news clips, sketches, etc."
                ),
            },
            "speaker": {
                "type": ["string", "null"],
                "description": "Name of the speaker if mentioned in the tweet, else null.",
            },
            "institution": {
                "type": ["string", "null"],
                "description": "Affiliated university/lab/conference if mentioned, else null.",
            },
            "topic": {
                "type": ["string", "null"],
                "description": "Short noun phrase for the lecture topic, e.g. 'world models', 'monte carlo simulation', 'KV cache'. Null if unclear.",
            },
            "year_hint": {
                "type": ["string", "null"],
                "description": "Year (e.g. '2024') if mentioned or inferable from tweet date.",
            },
            "duration_minutes_hint": {
                "type": ["integer", "null"],
                "description": (
                    "If the tweet explicitly mentions the lecture's length (e.g., '2 hour lecture' "
                    "→ 120, '45 min talk' → 45, '90-minute seminar' → 90), return integer minutes. "
                    "Else null. Be precise; don't guess."
                ),
            },
            "search_queries": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": (
                    "If is_lecture is true, return 1-3 YouTube search queries you'd type to find "
                    "the original lecture. Use REAL lecture vocabulary, not marketing prose from "
                    "the tweet. Good: 'Yann LeCun world models lecture', 'Karpathy zero to hero "
                    "neural networks', 'MIT 18.06 linear algebra Strang'. Bad: 'AI beyond LLMs, "
                    "trillion dollar future' (that's tweet pitch, not how lectures get titled). "
                    "Include speaker name in every query. Don't quote terms. Return null if not a lecture."
                ),
            },
        },
        "required": ["is_lecture"],
    },
}


def classify_and_extract(b: dict) -> dict:
    text = b.get("text") or ""
    quoted = ((b.get("quotedTweet") or {}).get("text")) or ""
    handle = b.get("authorHandle") or ""
    posted = b.get("postedAt") or ""

    prompt = (
        "A user bookmarked this tweet which contains an attached video clip. "
        "Decide whether the video is a clip from a long-form lecture/talk that likely "
        "has a full version on YouTube, and extract any signals about speaker, institution, "
        "and topic from the tweet text.\n\n"
        f"Tweet author: @{handle}\n"
        f"Posted: {posted}\n"
        f"Tweet text:\n{text}\n\n"
        + (f"Quoted tweet text:\n{quoted}\n\n" if quoted else "")
        + "Submit your classification via the tool."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        tools=[CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "submit_classification"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return dict(block.input)
    return {"is_lecture": False}


# ── Stage 2: yt-dlp search ──────────────────────────────────────────────────

def build_queries(extracted: dict) -> list[str]:
    """Prefer model-supplied queries; fall back to a synthesized one."""
    q = extracted.get("search_queries") or []
    q = [s for s in q if isinstance(s, str) and s.strip()]
    if q:
        return q[:3]
    # Fallback (rare — model failed to supply queries)
    parts = []
    if extracted.get("speaker"):
        parts.append(extracted["speaker"])
    if extracted.get("topic"):
        parts.append(extracted["topic"])
    parts.append("lecture")
    return [" ".join(parts).strip()]


_YDL_OPTS = {
    "quiet": True,
    "skip_download": True,
    "extract_flat": "in_playlist",
    "no_warnings": True,
    "ignoreerrors": True,
}


def search_youtube(query: str, k: int = SEARCH_K) -> list[dict]:
    with YoutubeDL(_YDL_OPTS) as ydl:
        info = ydl.extract_info(f"ytsearch{k}:{query}", download=False)
    out = []
    for e in (info or {}).get("entries") or []:
        if not e:
            continue
        dur = e.get("duration") or 0
        if dur < MIN_LECTURE_DURATION_SEC:
            continue
        out.append({
            "id": e.get("id"),
            "url": f"https://www.youtube.com/watch?v={e.get('id')}",
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "duration": dur,
            "description": (e.get("description") or "")[:300],
        })
    return out


# ── Stage 3: rank candidates ────────────────────────────────────────────────

RANK_TOOL = {
    "name": "submit_match",
    "description": "Submit your best-match decision for this lecture.",
    "input_schema": {
        "type": "object",
        "properties": {
            "best_index": {
                "type": ["integer", "null"],
                "description": "0-based index of the best matching candidate. Null if no candidate is a confident match.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0–1.0. How confident you are this is the same lecture clipped in the tweet. Use 0.0 if best_index is null.",
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence justifying the pick (or non-pick).",
            },
        },
        "required": ["best_index", "confidence", "reasoning"],
    },
}


def rank_candidates(b: dict, extracted: dict, candidates: list[dict]) -> dict:
    if not candidates:
        return {"best_index": None, "confidence": 0.0, "reasoning": "no candidates"}
    # When multiple queries produced unique hits, allow a few more candidates through.
    candidates = candidates[: max(RANK_K, 8)]
    cand_lines = []
    for i, c in enumerate(candidates):
        mins = c["duration"] // 60
        cand_lines.append(
            f"[{i}] {c['title']}\n"
            f"    channel: {c['channel']} | duration: {mins} min\n"
            f"    desc: {c['description'][:200]}"
        )

    duration_hint = extracted.get("duration_minutes_hint")
    duration_rule = ""
    if duration_hint:
        duration_rule = (
            f"\n\nDURATION CONSTRAINT: The tweet says the lecture is ~{duration_hint} minutes. "
            f"Reject any candidate whose duration is off by more than ~25%. The Twitter clip is a "
            f"slice of the full lecture, so the YouTube video must be AT LEAST as long as stated, "
            f"and not absurdly longer. If no candidate's duration fits, return null."
        )

    prompt = (
        "You're matching a Twitter lecture clip to its likely YouTube original.\n\n"
        f"Tweet text:\n{b.get('text', '')}\n\n"
        f"Extracted hints: speaker={extracted.get('speaker')} institution={extracted.get('institution')} "
        f"topic={extracted.get('topic')} year_hint={extracted.get('year_hint')} "
        f"duration_minutes_hint={duration_hint}\n\n"
        "YouTube candidates:\n" + "\n\n".join(cand_lines) + "\n\n"
        "Pick the candidate that's most likely the source lecture, or null if none is a confident match. "
        "Channel name matching the speaker or hosting institution is a strong signal. Topic alignment "
        "matters more than exact title match."
        + duration_rule +
        "\n\nBe strict: false positives waste the user's time more than misses do. Confidence should "
        "reflect speaker/institution match AND topic match AND duration plausibility — if any of "
        "these fail, confidence drops sharply. Submit via the tool."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        tools=[RANK_TOOL],
        tool_choice={"type": "tool", "name": "submit_match"},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in resp.content:
        if block.type == "tool_use":
            return dict(block.input)
    return {"best_index": None, "confidence": 0.0, "reasoning": "no tool call returned"}


# ── Per-bookmark trace ──────────────────────────────────────────────────────

def trace_one(b: dict) -> dict:
    tweet_id = b["id"]
    base = {
        "tweet_id": tweet_id,
        "tweet_url": b["url"],
        "tweet_text": (b.get("text") or "")[:500],
        "tweet_author_handle": b.get("authorHandle"),
        "tweet_posted_at": b.get("postedAt"),
        "traced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "text_only",
    }

    extracted = classify_and_extract(b)
    base["is_lecture"] = bool(extracted.get("is_lecture"))
    base["extracted"] = {
        k: extracted.get(k) for k in ("speaker", "institution", "topic", "year_hint")
    }

    if not base["is_lecture"]:
        return {**base, "status": "not_lecture"}

    queries = build_queries(extracted)
    base["queries"] = queries
    seen = {}
    try:
        for q in queries:
            for c in search_youtube(q, k=SEARCH_K):
                # de-dupe by video id, keep first occurrence
                if c["id"] not in seen:
                    seen[c["id"]] = c
    except Exception as e:
        return {**base, "status": "search_error", "error": str(e)[:200]}
    candidates = list(seen.values())

    if not candidates:
        return {**base, "status": "no_candidates"}

    decision = rank_candidates(b, extracted, candidates)
    base["rank_reasoning"] = decision.get("reasoning", "")
    confidence = float(decision.get("confidence") or 0.0)
    best = decision.get("best_index")

    if best is None or confidence < CONFIDENCE_THRESHOLD:
        return {**base, "status": "low_confidence", "confidence": confidence}

    pick = candidates[best]
    return {
        **base,
        "status": "traced",
        "youtube_url": pick["url"],
        "youtube_title": pick["title"],
        "youtube_channel": pick["channel"],
        "youtube_duration_sec": pick["duration"],
        "confidence": confidence,
    }


# ── Driver ──────────────────────────────────────────────────────────────────

def load_traces() -> dict[str, dict]:
    if not TRACES_PATH.exists():
        return {}
    out = {}
    with open(TRACES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[row["tweet_id"]] = row
            except json.JSONDecodeError:
                pass
    return out


def append_trace(row: dict) -> None:
    with open(TRACES_PATH, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_traces(rows_by_id: dict[str, dict]) -> None:
    """Used after --force re-traces; rewrite the file deduped by tweet_id."""
    tmp = TRACES_PATH.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for row in rows_by_id.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(TRACES_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweet-id", help="Process a single bookmark by id")
    ap.add_argument("--limit", type=int, help="Cap number of video bookmarks")
    ap.add_argument("--force", action="store_true", help="Re-trace already-traced rows")
    args = ap.parse_args()

    bookmarks = [json.loads(line) for line in BOOKMARKS_PATH.open()]
    bookmarks = [b for b in bookmarks if is_video_bookmark(b)]
    if args.tweet_id:
        bookmarks = [b for b in bookmarks if b["id"] == args.tweet_id]
    if args.limit:
        bookmarks = bookmarks[: args.limit]

    cached = load_traces()
    sym = {
        "traced": "[ok]",
        "not_lecture": "[skip]",
        "no_candidates": "[na ]",
        "low_confidence": "[low]",
        "search_error": "[ERR]",
        "cached": "[--]",
    }
    counts = {k: 0 for k in sym}

    rewrite_needed = False
    for b in bookmarks:
        tid = b["id"]
        prev = cached.get(tid)
        if prev and not args.force and prev.get("status") == "traced":
            counts["cached"] += 1
            print(f"{sym['cached']} {tid} @{b.get('authorHandle','?'):20} cached: {prev.get('youtube_url','?')}")
            continue

        try:
            row = trace_one(b)
        except Exception as e:
            row = {
                "tweet_id": tid,
                "tweet_url": b["url"],
                "tweet_author_handle": b.get("authorHandle"),
                "status": "search_error",
                "error": f"unhandled: {e}",
                "traced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

        status = row.get("status", "search_error")
        counts[status] = counts.get(status, 0) + 1

        if args.force and prev:
            cached[tid] = row
            rewrite_needed = True
        else:
            append_trace(row)
            cached[tid] = row

        info = ""
        if status == "traced":
            info = f"{row.get('confidence', 0):.2f} → {row.get('youtube_url')}"
        elif status == "low_confidence":
            info = f"conf={row.get('confidence', 0):.2f} ({row.get('rank_reasoning','')[:60]})"
        elif status == "not_lecture":
            info = ""
        elif status == "no_candidates":
            info = f"q={row.get('query','')[:60]}"
        elif status == "search_error":
            info = row.get("error", "")[:80]

        print(f"{sym.get(status,'[?]')} {tid} @{b.get('authorHandle','?'):20} {info}")

    if rewrite_needed:
        rewrite_traces(cached)

    print()
    print("summary:", " ".join(f"{k}={v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
