#!/usr/bin/env python3
"""
Bookmark agent — ask natural language questions about your X bookmarks.

Usage:
  python agent.py "get me the tweet about KV cache optimization"
  python agent.py "do we have anything on YC advice?"
  python agent.py          # interactive mode

Memory layer (Hermes-pattern, layers 1 + 3):
  ~/.ft-bookmarks/agent/MEMORY.md   — durable facts (always in system prompt)
  ~/.ft-bookmarks/agent/USER.md     — user profile  (always in system prompt)
  ~/.ft-bookmarks/agent/sessions.db — SQLite FTS5 of every Q→A turn
The model can rewrite memory via update_memory / update_user, and search
past conversations via session_search.
"""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(override=True)
import anthropic

BOOKMARKS_PATH = os.path.expanduser("~/.ft-bookmarks/bookmarks.jsonl")
ARTICLES_DIR = os.path.expanduser("~/.ft-bookmarks/articles")
AGENT_DIR = os.path.expanduser("~/.ft-bookmarks/agent")
MEMORY_PATH = os.path.join(AGENT_DIR, "MEMORY.md")
USER_PATH = os.path.join(AGENT_DIR, "USER.md")
SESSIONS_DB = os.path.join(AGENT_DIR, "sessions.db")

MEMORY_LIMIT = 2200   # chars — Hermes convention
USER_LIMIT = 1375     # chars

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

os.makedirs(AGENT_DIR, exist_ok=True)


# ── Memory files ─────────────────────────────────────────────────────────────

def _read(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return f.read()


def _write_capped(path: str, content: str, limit: int) -> str:
    content = content.rstrip() + "\n"
    if len(content) > limit:
        return f"refused: {len(content)} chars exceeds limit of {limit}. Trim and retry."
    with open(path, "w") as f:
        f.write(content)
    return f"ok: wrote {len(content)} chars to {os.path.basename(path)}"


def update_memory(content: str) -> str:
    return _write_capped(MEMORY_PATH, content, MEMORY_LIMIT)


def update_user(content: str) -> str:
    return _write_capped(USER_PATH, content, USER_LIMIT)


# ── Session log (SQLite FTS5) ────────────────────────────────────────────────

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(SESSIONS_DB)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS turns USING fts5(
            role UNINDEXED,
            content,
            session_id UNINDEXED,
            ts UNINDEXED,
            tokenize='porter unicode61'
        )
    """)
    return conn


def log_turn(session_id: str, role: str, content: str) -> None:
    if not content.strip():
        return
    with _db() as conn:
        conn.execute(
            "INSERT INTO turns(role, content, session_id, ts) VALUES (?, ?, ?, ?)",
            (role, content, session_id, datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )


def session_search(query: str, limit: int = 5) -> str:
    if not query.strip():
        return "empty query"
    try:
        with _db() as conn:
            rows = conn.execute(
                """SELECT role, content, session_id, ts
                   FROM turns WHERE turns MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
    except sqlite3.OperationalError as e:
        return f"search error: {e}"
    if not rows:
        return f"no past turns matched {query!r}"
    out = []
    for role, content, sid, ts in rows:
        snippet = content if len(content) <= 400 else content[:400] + "…"
        out.append(f"[{ts}] {role}:\n{snippet}")
    return "\n\n".join(out)


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_bookmarks",
        "description": (
            "Full-text BM25 search across all bookmarked tweets. "
            "Use this for keyword or topic queries. Returns ranked results with tweet text, author, URL. "
            "Try multiple queries if the first doesn't find what you need."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_stats",
        "description": "Get overall stats: total bookmarks, top authors, date range, language breakdown.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_categories",
        "description": "Show how bookmarks are distributed across categories (if classified).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_bookmarks",
        "description": "List bookmarks with optional filters. Good for browsing by author or date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "Filter by @handle (without @)"},
                "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
            },
        },
    },
    {
        "name": "show_bookmark",
        "description": "Get full details for a specific bookmark by its tweet ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "tweet_id": {"type": "string", "description": "Tweet ID from a search result"},
            },
            "required": ["tweet_id"],
        },
    },
    {
        "name": "read_article",
        "description": (
            "Read the full body of the article linked from a bookmark, when one was enriched. "
            "Many bookmarks just point at long-form articles (X-native or external blogs); the body "
            "is on disk as markdown with images. Use this whenever the user asks to summarize, "
            "explain, or quote from an article — search returns the tweet snippet, this returns the article."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tweet_id": {"type": "string", "description": "Tweet ID of the bookmark"},
            },
            "required": ["tweet_id"],
        },
    },
    {
        "name": "session_search",
        "description": (
            "Search past conversation turns (your prior questions + my prior answers) using full-text "
            "search. Use this when the user references something from before — 'what did we conclude "
            "about X', 'remind me of that article you found', 'what was the harness thing'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "FTS query, e.g. 'agent harness memory'"},
                "limit": {"type": "integer", "description": "Max matches (default 5)", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "update_memory",
        "description": (
            f"Rewrite the entire MEMORY.md file (cap {MEMORY_LIMIT} chars). Use for durable facts "
            "about the user's environment, conventions, recurring decisions, lessons learned. "
            "NOT for task progress, session outcomes, or temporary state. "
            "You must pass the FULL new content — this replaces the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Full new contents of MEMORY.md"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "update_user",
        "description": (
            f"Rewrite the entire USER.md file (cap {USER_LIMIT} chars). Use for the user's identity, "
            "interests, communication style, decisions about how to work together. "
            "You must pass the FULL new content — this replaces the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Full new contents of USER.md"},
            },
            "required": ["content"],
        },
    },
]


# ── Tool dispatch ────────────────────────────────────────────────────────────

def run_ft(*args) -> str:
    result = subprocess.run(["ft", *args], capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def search_bookmarks(query: str, limit: int = 10) -> str:
    return run_ft("search", query, "--limit", str(limit))


def get_stats() -> str:
    return run_ft("stats")


def list_categories() -> str:
    return run_ft("categories")


def list_bookmarks(author: str | None = None, limit: int = 20) -> str:
    args = ["list", "--limit", str(limit)]
    if author:
        args += ["--author", author]
    return run_ft(*args)


def show_bookmark(tweet_id: str) -> str:
    return run_ft("show", tweet_id)


def read_article(tweet_id: str) -> str:
    folder = os.path.join(ARTICLES_DIR, tweet_id)
    if not os.path.isdir(folder):
        return f"No enriched article on disk for tweet {tweet_id}."
    candidates = sorted(p for p in os.listdir(folder) if p.endswith(".md"))
    if not candidates:
        return f"No markdown body found in {folder}"
    with open(os.path.join(folder, candidates[0])) as f:
        return f.read()


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "search_bookmarks":
        return search_bookmarks(inputs["query"], inputs.get("limit", 10))
    if name == "get_stats":
        return get_stats()
    if name == "list_categories":
        return list_categories()
    if name == "list_bookmarks":
        return list_bookmarks(inputs.get("author"), inputs.get("limit", 20))
    if name == "show_bookmark":
        return show_bookmark(inputs["tweet_id"])
    if name == "read_article":
        return read_article(inputs["tweet_id"])
    if name == "session_search":
        return session_search(inputs["query"], inputs.get("limit", 5))
    if name == "update_memory":
        return update_memory(inputs["content"])
    if name == "update_user":
        return update_user(inputs["content"])
    return f"Unknown tool: {name}"


# ── System prompt + memory ──────────────────────────────────────────────────

INSTRUCTIONS = """You are a personal knowledge assistant for a curated collection of X (Twitter) bookmarks.

The user has saved tweets they found valuable. Many bookmarks point at long-form articles
(X-native or external blogs); their full bodies are pre-fetched to disk and accessible via
the read_article tool.

You have persistent memory:
- MEMORY.md and USER.md (below) are always loaded. They hold durable facts and the user model.
- session_search tool: full-text search over every past Q→A turn in this database.
- When you learn a durable fact about the user's environment, conventions, or preferences,
  call update_memory or update_user. Pass the FULL new file content (small, replace-style).
- Never write task progress, transient state, or session outcomes into MEMORY/USER. Those
  belong in the session log, which is kept automatically.

Guidelines:
- For "find me a tweet about X" — search and return the most relevant result with the URL.
- For "summarize / explain / what does X say" — use read_article to pull the full body.
- For "what did we say about Y" / "remember when..." — use session_search.
- Always include the tweet URL so the user can open it.
- Be concise. One good result beats ten mediocre ones.
"""


def build_system() -> list:
    memory = _read(MEMORY_PATH, "(empty)").rstrip()
    user = _read(USER_PATH, "(empty)").rstrip()
    full = (
        INSTRUCTIONS
        + "\n\n=== MEMORY.md ===\n"
        + memory
        + "\n\n=== USER.md ===\n"
        + user
    )
    # One cacheable block. When MEMORY/USER change, the cache invalidates;
    # the next call re-caches. Net cost ~free for repeated queries.
    return [{"type": "text", "text": full, "cache_control": {"type": "ephemeral"}}]


# ── Agent loop ───────────────────────────────────────────────────────────────

def run_agent(question: str) -> str:
    session_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_turn(session_id, "user", question)

    messages = [{"role": "user", "content": question}]
    final_text_parts: list[str] = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=build_system(),
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_text_parts = [b.text for b in response.content if b.type == "text"]
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            final_text_parts = [b.text for b in response.content if b.type == "text"]
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    answer = "".join(final_text_parts)
    log_turn(session_id, "assistant", answer)
    return answer


def interactive():
    print("Bookmark agent ready. Memory + session search enabled. (Ctrl+C to quit)\n")
    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not question:
            continue
        answer = run_agent(question)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(run_agent(question))
    else:
        interactive()
