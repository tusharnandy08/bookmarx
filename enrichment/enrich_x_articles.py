#!/usr/bin/env python3
"""
Fetch X-native articles (x.com/i/article/<id>) via authenticated GraphQL and
render them to markdown alongside external articles.

Articles are owned by tweets — the bookmark's tweet_id IS the tweet that owns
the article. So we just call TweetResultByRestId(tweetId=<bookmark.id>).

Auth: cookies extracted from the user's Comet profile via ../tools/x_cookies.mjs.
Same output layout as enrich_articles.py: ~/.ft-bookmarks/articles/<tweetId>/.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

from enrich_articles import (
    ARTICLES_DIR,
    BOOKMARKS_PATH,
    download_image,
    is_x_article,
    write_article,
)

X_PUBLIC_BEARER = (
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
)

# Captured from a live page load (tools/probe_x_article.py). Updates if X
# rotates query hashes — re-run the probe to get a fresh value.
TWEET_QUERY_HASH = "fHLDP3qFEjnTqhWBVvsREg"

GRAPHQL_FEATURES = {
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "premium_content_api_read_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
    "responsive_web_grok_analyze_post_followups_enabled": True,
    "responsive_web_jetfuel_frame": True,
    "responsive_web_grok_share_attachment_enabled": True,
    "responsive_web_grok_annotations_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "content_disclosure_indicator_enabled": True,
    "content_disclosure_ai_generated_indicator_enabled": True,
    "responsive_web_grok_show_grok_translated_post": True,
    "responsive_web_grok_analysis_button_from_backend": True,
    "post_ctas_fetch_enabled": True,
    "rweb_cashtags_enabled": True,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": False,
    "profile_label_improvements_pcf_label_in_post_enabled": True,
    "responsive_web_profile_redirect_enabled": False,
    "rweb_tipjar_consumption_enabled": False,
    "verified_phone_label_enabled": False,
    "responsive_web_grok_image_annotation_enabled": True,
    "responsive_web_grok_imagine_annotation_enabled": True,
    "responsive_web_grok_community_note_auto_translation_is_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
}

FIELD_TOGGLES = {
    "withArticleRichContentState": True,
    "withArticlePlainText": False,
    "withArticleSummaryText": True,
    "withArticleVoiceOver": True,
}


# ── Cookie bridge ────────────────────────────────────────────────────────────

def get_cookies() -> dict:
    raw = subprocess.check_output(
        ["node", str(Path(__file__).parent.parent / "tools" / "x_cookies.mjs")],
        stderr=subprocess.PIPE,
    )
    return json.loads(raw)


def auth_headers(cookies: dict) -> dict:
    return {
        "Authorization": f"Bearer {X_PUBLIC_BEARER}",
        "Cookie": cookies["cookieHeader"],
        "x-csrf-token": cookies["csrfToken"],
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "x-twitter-active-user": "yes",
        "x-twitter-auth-type": "OAuth2Session",
        "x-twitter-client-language": "en",
    }


def fetch_tweet_with_article(tweet_id: str, headers: dict) -> dict:
    variables = {
        "tweetId": tweet_id,
        "includePromotedContent": True,
        "withBirdwatchNotes": True,
        "withVoice": True,
        "withCommunity": True,
    }
    params = {
        "variables": json.dumps(variables, separators=(",", ":")),
        "features": json.dumps(GRAPHQL_FEATURES, separators=(",", ":")),
        "fieldToggles": json.dumps(FIELD_TOGGLES, separators=(",", ":")),
    }
    url = (
        f"https://x.com/i/api/graphql/{TWEET_QUERY_HASH}"
        f"/TweetResultByRestId?" + urllib.parse.urlencode(params)
    )
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.json()


# ── Renderer ────────────────────────────────────────────────────────────────

INLINE_STYLE_MARKERS = {"Bold": "**", "Italic": "*"}


def apply_inline(text: str, inline_ranges: list, entity_ranges: list,
                 entity_map: list) -> str:
    """Apply bold/italic and inline LINK entities to a single block's text."""
    open_at: dict[int, list[str]] = {}
    close_at: dict[int, list[str]] = {}

    for r in inline_ranges:
        marker = INLINE_STYLE_MARKERS.get(r.get("style"))
        if not marker:
            continue
        start = r["offset"]
        end = start + r["length"]
        open_at.setdefault(start, []).append(marker)
        close_at.setdefault(end, []).insert(0, marker)

    for r in entity_ranges:
        ent = entity_map[r["key"]].get("value") or {}
        if ent.get("type") == "LINK":
            url = (ent.get("data") or {}).get("url", "")
            start = r["offset"]
            end = start + r["length"]
            open_at.setdefault(start, []).append("[")
            close_at.setdefault(end, []).insert(0, f"]({url})")

    out: list[str] = []
    for i, ch in enumerate(text):
        for m in close_at.get(i, []):
            out.append(m)
        for m in open_at.get(i, []):
            out.append(m)
        out.append(ch)
    for m in close_at.get(len(text), []):
        out.append(m)
    return "".join(out)


def media_lookup(media_entities: list) -> dict:
    """Index media_entities by media_id (string)."""
    out = {}
    for m in media_entities or []:
        mid = str(m.get("media_id", ""))
        if mid:
            out[mid] = m
    return out


def render_atomic(entity: dict, media_by_id: dict, images_dir: Path,
                  source_url: str) -> str:
    v = entity.get("value") or {}
    t = v.get("type")
    data = v.get("data") or {}

    if t == "DIVIDER":
        return "---"

    if t == "MARKDOWN":
        return data.get("markdown", "").rstrip()

    if t == "MEDIA":
        items = data.get("mediaItems") or []
        chunks = []
        for it in items:
            mid = str(it.get("mediaId", ""))
            m = media_by_id.get(mid)
            if not m:
                continue
            info = m.get("media_info") or {}
            img_url = info.get("original_img_url")
            if not img_url:
                continue
            fname = download_image(img_url, images_dir)
            if fname:
                chunks.append(f"![](./images/{fname})")
            else:
                chunks.append(f"![]({img_url})")
        return "\n\n".join(chunks)

    if t == "TWEET":
        tid = data.get("tweetId") or data.get("rest_id") or ""
        # We deliberately don't fetch the embedded tweet to keep the pipeline
        # one request per article. Leave a pointer the agent can resolve later.
        return f"> embedded tweet: https://x.com/i/web/status/{tid}".rstrip(": ")

    return ""


def render_article(article: dict, images_dir: Path, source_url: str) -> tuple[str, int]:
    cs = article.get("content_state") or {}
    blocks = cs.get("blocks") or []
    entity_map = cs.get("entityMap") or []
    media_by_id = media_lookup(article.get("media_entities") or [])

    # cover image up top
    out_lines: list[str] = []
    image_count = 0
    cover = article.get("cover_media") or {}
    cover_url = ((cover.get("media_info") or {}).get("original_img_url"))
    if cover_url:
        fname = download_image(cover_url, images_dir)
        if fname:
            out_lines.append(f"![cover](./images/{fname})")
            image_count += 1
            out_lines.append("")

    block_prefix = {
        "unstyled": "",
        "header-one": "# ",
        "header-two": "## ",
        "header-three": "### ",
        "blockquote": "> ",
    }

    # State for list grouping
    ordered_counter = 0
    prev_type = None

    for b in blocks:
        btype = b.get("type", "unstyled")
        text = b.get("text", "")
        inline = b.get("inlineStyleRanges") or []
        ents = b.get("entityRanges") or []

        # Reset ordered list counter when leaving an ordered list
        if btype != "ordered-list-item":
            ordered_counter = 0

        if btype == "atomic":
            if not ents:
                continue
            ent = entity_map[ents[0]["key"]]
            rendered = render_atomic(ent, media_by_id, images_dir, source_url)
            if rendered:
                if rendered.startswith("![]") and "(./images/" in rendered:
                    image_count += rendered.count("![]")
                out_lines.append(rendered)
                out_lines.append("")
            prev_type = btype
            continue

        rendered_text = apply_inline(text, inline, ents, entity_map)

        if btype == "unordered-list-item":
            out_lines.append(f"- {rendered_text}")
        elif btype == "ordered-list-item":
            ordered_counter += 1
            out_lines.append(f"{ordered_counter}. {rendered_text}")
        else:
            prefix = block_prefix.get(btype, "")
            line = f"{prefix}{rendered_text}".rstrip()
            out_lines.append(line)
            out_lines.append("")
        prev_type = btype

    body = "\n".join(out_lines).strip() + "\n"
    return body, image_count


# ── Driver ──────────────────────────────────────────────────────────────────

def process(b: dict, headers: dict, force: bool):
    links = b.get("links") or []
    if not links or not is_x_article(links[0]):
        return None
    tweet_id = b["id"]
    out_dir = ARTICLES_DIR / tweet_id
    # Cached if any *.md exists in the folder. body.md gets renamed to
    # <author-slug>.md by build_graph; checking only body.md would re-fetch
    # every previously-enriched article on each sync cycle.
    if not force and out_dir.is_dir() and any(out_dir.glob("*.md")):
        return ("cached", "exists", links[0])

    try:
        data = fetch_tweet_with_article(tweet_id, headers)
    except requests.HTTPError as e:
        return ("error", f"HTTP {e.response.status_code}", links[0])
    except Exception as e:
        return ("error", str(e)[:80], links[0])

    result = (data.get("data") or {}).get("tweetResult", {}).get("result") or {}
    article_wrapper = result.get("article") or {}
    article = (article_wrapper.get("article_results") or {}).get("result")

    if not article:
        return ("no_article", "tweet has no article payload", links[0])

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    body, image_count = render_article(article, images_dir, links[0])

    fm = {
        "tweet_id": tweet_id,
        "tweet_url": b["url"],
        "source_url": links[0],
        "source_kind": "x_article",
        "title": article.get("title"),
        "site": "x.com",
        "tweet_author_handle": b.get("authorHandle"),
        "tweet_posted_at": b.get("postedAt"),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_count": image_count,
        "body_chars": len(body),
        "summary_text": article.get("summary_text") or article.get("preview_text"),
    }
    write_article(out_dir, fm, body)
    return ("ok", f"{image_count} imgs, {len(body)} chars", links[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tweet-id")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between requests (default 0.5)")
    args = ap.parse_args()

    cookies = get_cookies()
    headers = auth_headers(cookies)

    bookmarks = [json.loads(line) for line in BOOKMARKS_PATH.open()]
    bookmarks = [
        b for b in bookmarks
        if b.get("links") and is_x_article(b["links"][0])
    ]
    if args.tweet_id:
        bookmarks = [b for b in bookmarks if b["id"] == args.tweet_id]
    if args.limit:
        bookmarks = bookmarks[: args.limit]

    sym = {"ok": "[ok]", "cached": "[--]", "error": "[ERR]", "no_article": "[na ]"}
    counts = {k: 0 for k in sym}

    for i, b in enumerate(bookmarks):
        result = process(b, headers, force=args.force)
        if not result:
            continue
        status, info, url = result
        counts[status] += 1
        print(f"{sym[status]} {b['id']} @{b.get('authorHandle','?'):20} {info}")
        if status not in ("cached",) and i < len(bookmarks) - 1:
            time.sleep(args.sleep)

    print()
    print("summary:", " ".join(f"{k}={v}" for k, v in counts.items() if v))


if __name__ == "__main__":
    main()
