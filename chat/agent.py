#!/usr/bin/env python3
"""
Bookmark agent — chat interface over your X bookmarks, built on the
Claude Agent SDK with a context-engineering layer inspired by Xu et al.,
"Everything is Context: Agentic File System Abstraction for Context
Engineering" (arXiv:2512.05470, 2025).

Memory layout (~/.ft-bookmarks/agent/):
  history/<session-id>.jsonl      raw per-turn log (user, tool, assistant)
  memory/
    user.md                       WHO the user is, always in system prompt
    facts.md                      atomic durable facts, always in system prompt
    episodic/<session-id>.md      per-session summaries, lazy-loaded via recall()
  scratchpad/current.md           transient task notes, cleared per session

Across sessions: continue_conversation=True lets the SDK resume the prior
session's full message history. Within-session compaction is handled by
Claude Code itself. Long-term context flow is our concern: at session end,
if the conversation crossed COMPRESSION_TURN_THRESHOLD turns, a Haiku
summariser writes an episodic memory file that becomes searchable via
the recall() tool from future sessions.

Tools (4):
  update_user(content)   replace user.md
  update_facts(content)  replace facts.md
  pin_fact(text)         append one bullet to facts.md
  recall(query, limit)   grep facts.md + memory/episodic/*.md

Usage:
  python agent.py                    # interactive REPL
  python agent.py "your question"    # one-shot
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

# ── Paths ────────────────────────────────────────────────────────────────────

BOOKMARKS_PATH = Path.home() / ".ft-bookmarks" / "bookmarks.jsonl"
ARTICLES_DIR = Path.home() / ".ft-bookmarks" / "articles"
AGENT_DIR = Path.home() / ".ft-bookmarks" / "agent"
HISTORY_DIR = AGENT_DIR / "history"
MEMORY_DIR = AGENT_DIR / "memory"
EPISODIC_DIR = MEMORY_DIR / "episodic"
SCRATCHPAD_DIR = AGENT_DIR / "scratchpad"

USER_PATH = MEMORY_DIR / "user.md"
FACTS_PATH = MEMORY_DIR / "facts.md"
LEGACY_MEMORY_PATH = AGENT_DIR / "memory.md"

USER_LIMIT = 1500    # chars
FACTS_LIMIT = 3000   # chars
COMPRESSION_TURN_THRESHOLD = 30
SUMMARISER_MODEL = "claude-haiku-4-5"
MODEL = "claude-sonnet-4-5"


# ── First-run setup + migration ─────────────────────────────────────────────

def _ensure_dirs() -> None:
    for p in (HISTORY_DIR, MEMORY_DIR, EPISODIC_DIR, SCRATCHPAD_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_memory() -> None:
    """If memory/user.md and memory/facts.md don't exist, split the legacy
    monolithic memory.md into them. Old file is preserved as memory.md.bak."""
    if USER_PATH.exists() or FACTS_PATH.exists():
        return
    if not LEGACY_MEMORY_PATH.exists():
        # Cold start — write minimal stubs.
        USER_PATH.write_text(
            "# user.md — who the user is\n\n"
            "Identity, communication style, decisions about how to work together. "
            f"Cap ~{USER_LIMIT} chars. Always loaded into the system prompt.\n\n"
            "## Identity\n- (unknown — fill in via update_user as you learn)\n\n"
            "## Working style\n- (unknown)\n"
        )
        FACTS_PATH.write_text(
            "# facts.md — durable atomic facts\n\n"
            f"Environment, data layout, conventions. Cap ~{FACTS_LIMIT} chars. "
            "Pin via pin_fact (append) or rewrite via update_facts.\n\n"
            "## Environment\n"
            "- Bookmarks at ~/.ft-bookmarks/bookmarks.jsonl, synced via `ft sync` from Comet.\n"
            "- Enriched articles at ~/.ft-bookmarks/articles/<tweetId>/<author-slug>.md.\n"
            "- Traced video lectures at ~/.ft-bookmarks/videos/<tweetId>/source.md.\n"
        )
        return
    text = LEGACY_MEMORY_PATH.read_text()
    # Split on "## " sections; collect Environment → facts, User → user.
    user_lines: list[str] = []
    facts_lines: list[str] = []
    current = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            head = m.group(1).strip().lower()
            if head == "user":
                current = user_lines
            elif head in ("environment", "conventions", "facts"):
                current = facts_lines
            else:
                current = None
            continue
        if current is not None and line.strip():
            current.append(line)
    USER_PATH.write_text(
        "# user.md\n\n## Identity\n" + "\n".join(user_lines).strip() + "\n"
        if user_lines else "# user.md\n\n## Identity\n- (unknown)\n"
    )
    FACTS_PATH.write_text(
        "# facts.md\n\n## Environment\n" + "\n".join(facts_lines).strip() + "\n"
        if facts_lines else "# facts.md\n\n## Environment\n- (none yet)\n"
    )
    LEGACY_MEMORY_PATH.rename(LEGACY_MEMORY_PATH.with_suffix(".md.bak"))


_ensure_dirs()
_migrate_legacy_memory()

# Per-process session id, used by tools and the REPL for history + summary file paths.
SESSION_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


# ── History log ─────────────────────────────────────────────────────────────

def history_path() -> Path:
    return HISTORY_DIR / f"{SESSION_ID}.jsonl"


def log_event(role: str, content: Any) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": role,
        "content": content,
    }
    with open(history_path(), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


# ── Tool helpers ─────────────────────────────────────────────────────────────

def _text_result(s: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": s}]}


def _run_ft(*args: str) -> str:
    r = subprocess.run(["ft", *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


# ── Bookmark tools ───────────────────────────────────────────────────────────

@tool(
    "search_bookmarks",
    "Full-text BM25 search across all bookmarked tweets. Use for keyword/topic queries. "
    "Returns ranked results with tweet text, author, URL.",
    {"query": str, "limit": int},
)
async def search_bookmarks(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(_run_ft("search", args["query"], "--limit", str(args.get("limit", 10))))


@tool("get_stats", "Overall bookmark stats: counts, top authors, date range.", {})
async def get_stats(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(_run_ft("stats"))


@tool("list_categories", "Show how bookmarks are distributed across categories.", {})
async def list_categories(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(_run_ft("categories"))


@tool(
    "list_bookmarks",
    "List bookmarks with optional author filter. Good for browsing.",
    {"author": str, "limit": int},
)
async def list_bookmarks(args: dict[str, Any]) -> dict[str, Any]:
    cli = ["list", "--limit", str(args.get("limit", 20))]
    if args.get("author"):
        cli += ["--author", args["author"]]
    return _text_result(_run_ft(*cli))


@tool(
    "show_bookmark",
    "Get full details for a specific bookmark by its tweet ID.",
    {"tweet_id": str},
)
async def show_bookmark(args: dict[str, Any]) -> dict[str, Any]:
    return _text_result(_run_ft("show", args["tweet_id"]))


@tool(
    "read_article",
    "Read the full body of the article linked from a bookmark, when one was enriched. "
    "Many bookmarks point at long-form articles; the body is on disk as markdown with images. "
    "Use whenever the user asks to summarize/explain/quote — search returns the tweet snippet, "
    "this returns the article.",
    {"tweet_id": str},
)
async def read_article(args: dict[str, Any]) -> dict[str, Any]:
    folder = ARTICLES_DIR / args["tweet_id"]
    if not folder.is_dir():
        return _text_result(f"No enriched article on disk for tweet {args['tweet_id']}.")
    candidates = sorted(p for p in folder.iterdir() if p.suffix == ".md")
    if not candidates:
        return _text_result(f"No markdown body found in {folder}")
    return _text_result(candidates[0].read_text())


# ── Memory tools (4) ────────────────────────────────────────────────────────

@tool(
    "update_user",
    f"Rewrite the entire user.md file (cap {USER_LIMIT} chars). Pass the FULL new content. "
    "Use for the user's identity, communication style, decisions about how to work together. "
    "user.md is always loaded into the system prompt — keep it concise and high-signal.",
    {"content": str},
)
async def update_user(args: dict[str, Any]) -> dict[str, Any]:
    content = args["content"].rstrip() + "\n"
    if len(content) > USER_LIMIT:
        return _text_result(f"refused: {len(content)} > {USER_LIMIT} cap. Trim and retry.")
    USER_PATH.write_text(content)
    return _text_result(f"ok: wrote {len(content)} chars to user.md")


@tool(
    "update_facts",
    f"Rewrite the entire facts.md file (cap {FACTS_LIMIT} chars). Pass the FULL new content. "
    "Use for durable atomic facts about the user's environment, data layout, recurring "
    "conventions. facts.md is always loaded into the system prompt — keep it dense.",
    {"content": str},
)
async def update_facts(args: dict[str, Any]) -> dict[str, Any]:
    content = args["content"].rstrip() + "\n"
    if len(content) > FACTS_LIMIT:
        return _text_result(f"refused: {len(content)} > {FACTS_LIMIT} cap. Trim and retry.")
    FACTS_PATH.write_text(content)
    return _text_result(f"ok: wrote {len(content)} chars to facts.md")


@tool(
    "pin_fact",
    "Append a single one-line bullet to facts.md. Cheaper than update_facts when you "
    "want to add one fact without rewriting the whole file. Refuses if facts.md would "
    f"exceed {FACTS_LIMIT} chars after append.",
    {"text": str},
)
async def pin_fact(args: dict[str, Any]) -> dict[str, Any]:
    line = args["text"].strip()
    if not line:
        return _text_result("refused: empty fact")
    bullet = f"- {line}\n"
    current = FACTS_PATH.read_text() if FACTS_PATH.exists() else ""
    if len(current) + len(bullet) > FACTS_LIMIT:
        return _text_result(f"refused: would exceed {FACTS_LIMIT} char cap. Use update_facts to consolidate.")
    with open(FACTS_PATH, "a") as f:
        f.write(bullet)
    return _text_result(f"ok: pinned ({len(current) + len(bullet)} chars total)")


@tool(
    "recall",
    "Search past episodic session summaries and facts.md for relevant context. "
    "Use when the user references something from a prior session ('remember when…', "
    "'what did we decide about X'). Returns matching snippets with source file.",
    {"query": str, "limit": int},
)
async def recall(args: dict[str, Any]) -> dict[str, Any]:
    query = (args.get("query") or "").strip().lower()
    limit = int(args.get("limit") or 5)
    if not query:
        return _text_result("empty query")
    terms = [t for t in re.split(r"\W+", query) if len(t) >= 3]
    if not terms:
        return _text_result("query too short (need a 3+ char term)")

    hits: list[tuple[str, str, int]] = []  # (source, snippet, score)
    sources = list(EPISODIC_DIR.glob("*.md"))
    if FACTS_PATH.exists():
        sources.append(FACTS_PATH)
    for path in sources:
        try:
            text = path.read_text()
        except OSError:
            continue
        # paragraph-level scoring: count term occurrences
        for chunk in re.split(r"\n\s*\n", text):
            chunk_l = chunk.lower()
            score = sum(chunk_l.count(t) for t in terms)
            if score == 0:
                continue
            label = path.relative_to(AGENT_DIR).as_posix()
            snippet = chunk if len(chunk) <= 600 else chunk[:600] + "…"
            hits.append((label, snippet, score))
    hits.sort(key=lambda x: -x[2])
    if not hits:
        return _text_result(f"no matches for {query!r}")
    out = [f"=== {label} (score {score}) ===\n{snippet}" for label, snippet, score in hits[:limit]]
    return _text_result("\n\n".join(out))


# ── Server bundle ────────────────────────────────────────────────────────────

BOOKMARK_TOOLS = [
    search_bookmarks, get_stats, list_categories, list_bookmarks, show_bookmark, read_article,
    update_user, update_facts, pin_fact, recall,
]
SERVER = create_sdk_mcp_server(name="bookmarks", version="0.2.0", tools=BOOKMARK_TOOLS)
TOOL_NAMES = [
    "mcp__bookmarks__search_bookmarks",
    "mcp__bookmarks__get_stats",
    "mcp__bookmarks__list_categories",
    "mcp__bookmarks__list_bookmarks",
    "mcp__bookmarks__show_bookmark",
    "mcp__bookmarks__read_article",
    "mcp__bookmarks__update_user",
    "mcp__bookmarks__update_facts",
    "mcp__bookmarks__pin_fact",
    "mcp__bookmarks__recall",
]


# ── System prompt ────────────────────────────────────────────────────────────

INSTRUCTIONS = """You are a personal knowledge assistant over a curated collection of X (Twitter) bookmarks.

The user has bookmarked tweets they found valuable. Many bookmarks point at long-form articles
(X-native or external blogs); their full bodies are pre-fetched to disk and accessible via the
read_article tool.

You have a tiered persistent memory:
- user.md (below) — WHO the user is. Always loaded.
- facts.md (below) — durable atomic facts. Always loaded.
- episodic memory — per-session summaries, lazy-loaded via the recall() tool.
- conversation history — managed by the harness (within and across sessions).

When you learn something durable about the user, decide which file:
- Identity / communication style / working agreements → update_user (full rewrite)
- Environment / data layout / one-off facts → pin_fact (append) or update_facts (rewrite to consolidate)
NEVER write task progress, partial todos, or session outcomes to memory — those go to history automatically.

When the user references something from a prior session ("remember when…", "what did we say about X"),
call recall() before answering.

Bookmark tasks:
- "find me a tweet about X" → search_bookmarks, return the most relevant + URL.
- "summarize / explain / what does X say" → read_article on the relevant tweet_id.
- "do we have anything on Y" → search and give a direct yes/no with examples.
- Always include the tweet URL.
- Be concise. One good result beats ten mediocre ones.
"""


def build_system_prompt() -> str:
    user_md = USER_PATH.read_text() if USER_PATH.exists() else "(empty)"
    facts_md = FACTS_PATH.read_text() if FACTS_PATH.exists() else "(empty)"
    return (
        f"{INSTRUCTIONS}\n\n"
        f"=== user.md ===\n{user_md}\n\n"
        f"=== facts.md ===\n{facts_md}"
    )


def build_options(continue_conv: bool = True) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=build_system_prompt(),
        model=MODEL,
        mcp_servers={"bookmarks": SERVER},
        tools=TOOL_NAMES,
        allowed_tools=TOOL_NAMES,
        setting_sources=[],
        cwd=str(AGENT_DIR),
        continue_conversation=continue_conv,
        permission_mode="default",
    )


# ── Render + log assistant messages ──────────────────────────────────────────

def render_message(msg) -> None:
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text)
                log_event("assistant", block.text)
            elif isinstance(block, ToolUseBlock):
                short = block.name.replace("mcp__bookmarks__", "")
                print(f"\033[2m[→ {short}({_short_args(block.input)})]\033[0m")
                log_event("tool_use", {"name": short, "input": dict(block.input)})
    elif isinstance(msg, UserMessage):
        # tool_result blocks come back wrapped in a UserMessage
        for block in msg.content:
            if hasattr(block, "type") and block.type == "tool_result":
                content = getattr(block, "content", "")
                # content may be a list of blocks or a string
                if isinstance(content, list):
                    text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
                else:
                    text = str(content)
                log_event("tool_result", {"id": getattr(block, "tool_use_id", ""), "content": text[:2000]})


def _short_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:40] + "…"
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


# ── Session-end summariser ──────────────────────────────────────────────────

SUMMARY_PROMPT = """You are summarising a finished agent session for long-term memory.

Read the transcript below (raw turn log: user prompts, tool calls + results, assistant text) and produce a tight episodic summary that a future session of the same agent could use to recall what happened.

Rules:
- 8–15 bullets max. Dense, no fluff.
- Cover: what the user wanted, what was found/done, key facts uncovered, decisions made, open questions.
- Quote tweet URLs / article slugs / file paths verbatim when they're load-bearing.
- Skip mechanical detail (which tool fired when) unless it's a finding.
- Speak in past tense, third person ("user asked X, agent answered Y").

Transcript:
{transcript}

Output the summary as plain markdown. No preamble.
"""


def _read_history(session_id: str, max_chars: int = 60000) -> tuple[str, int]:
    """Returns (transcript_text, turn_count). Truncates if too large."""
    path = HISTORY_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return "", 0
    rows = []
    turns = 0
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for r in rows:
        if r.get("role") == "user":
            turns += 1
    pieces = []
    for r in rows:
        role = r.get("role", "?")
        content = r.get("content", "")
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        pieces.append(f"[{r.get('ts','')}] {role}: {content}")
    transcript = "\n".join(pieces)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
        transcript = "[... earlier turns truncated ...]\n" + transcript
    return transcript, turns


def maybe_summarise_session() -> None:
    transcript, turns = _read_history(SESSION_ID)
    if turns < COMPRESSION_TURN_THRESHOLD:
        return
    if not transcript:
        return
    print(f"\033[2m[compressing {turns} turns into episodic memory…]\033[0m")
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model=SUMMARISER_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
        )
        body = "".join(b.text for b in resp.content if hasattr(b, "text"))
    except Exception as e:
        print(f"\033[2m[summariser failed: {e}]\033[0m")
        return
    out = EPISODIC_DIR / f"{SESSION_ID}.md"
    fm = (
        f"---\nsession_id: {SESSION_ID}\nturn_count: {turns}\n"
        f"summarised_at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n---\n\n"
    )
    out.write_text(fm + body.strip() + "\n")
    print(f"\033[2m[wrote {out.relative_to(AGENT_DIR)}]\033[0m")


# ── Modes ────────────────────────────────────────────────────────────────────

async def one_shot(question: str) -> None:
    log_event("user", question)
    options = build_options(continue_conv=True)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(question)
        async for msg in client.receive_response():
            render_message(msg)
    maybe_summarise_session()


async def repl() -> None:
    print(f"Bookmark agent ready. Session {SESSION_ID}. Ctrl+C to quit.\n")
    options = build_options(continue_conv=True)
    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                question = await asyncio.to_thread(input, "> ")
            except (KeyboardInterrupt, EOFError):
                print()
                break
            question = question.strip()
            if not question:
                continue
            log_event("user", question)
            await client.query(question)
            async for msg in client.receive_response():
                render_message(msg)
            print()
    maybe_summarise_session()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        asyncio.run(one_shot(question))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
