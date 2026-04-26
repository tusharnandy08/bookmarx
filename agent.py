#!/usr/bin/env python3
"""
Bookmark agent — chat interface over your X bookmarks, built on the
Claude Agent SDK (Anthropic's official agent harness — same engine as
Claude Code, with built-in session resumption and tool dispatch).

Usage:
  python agent.py                    # interactive chat REPL
  python agent.py "summarize the Ray Dalio piece"   # one-shot question

Memory:
  Within a session: full conversation context retained automatically.
  Across sessions: continue_conversation=True resumes the previous
    session, so the agent remembers prior turns even after you exit.
  Long-term durable facts: ~/.ft-bookmarks/agent/memory.md, injected
    into the system prompt every session. The agent can rewrite this
    file via the update_memory tool when it learns durable facts about
    you (preferences, environment, recurring decisions).

Requirements: `claude` CLI installed and authenticated, ANTHROPIC_API_KEY
in .env (or `claude login`).
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

# ── Paths ────────────────────────────────────────────────────────────────────

BOOKMARKS_PATH = Path.home() / ".ft-bookmarks" / "bookmarks.jsonl"
ARTICLES_DIR = Path.home() / ".ft-bookmarks" / "articles"
AGENT_DIR = Path.home() / ".ft-bookmarks" / "agent"
MEMORY_PATH = AGENT_DIR / "memory.md"

AGENT_DIR.mkdir(parents=True, exist_ok=True)
if not MEMORY_PATH.exists():
    MEMORY_PATH.write_text(
        "# Memory\n\n"
        "Durable facts about the user, their environment, and recurring conventions.\n"
        "Rewrite this file via update_memory when you learn something durable.\n"
        "Skip task progress / session outcomes — those live in session history.\n\n"
        "## Environment\n"
        "- Bookmarks at ~/.ft-bookmarks/bookmarks.jsonl, synced via `ft sync` from Comet.\n"
        "- Enriched articles at ~/.ft-bookmarks/articles/<tweetId>/<author-slug>.md, opened as an Obsidian vault.\n"
        "- 207 articles enriched out of 217 link-bearing bookmarks.\n\n"
        "## User\n"
        "- (unknown — fill in as you learn)\n"
    )

MODEL = "claude-sonnet-4-5"


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


# ── Memory tool ──────────────────────────────────────────────────────────────

MEMORY_LIMIT = 4000  # chars


@tool(
    "update_memory",
    f"Rewrite the entire memory file (cap {MEMORY_LIMIT} chars). Pass the FULL new content — "
    "this is a replace, not append. Use for durable facts about the user, environment, "
    "preferences, recurring conventions. Do NOT use for task progress or session outcomes.",
    {"content": str},
)
async def update_memory(args: dict[str, Any]) -> dict[str, Any]:
    content = args["content"].rstrip() + "\n"
    if len(content) > MEMORY_LIMIT:
        return _text_result(
            f"refused: {len(content)} chars exceeds limit of {MEMORY_LIMIT}. Trim and retry."
        )
    MEMORY_PATH.write_text(content)
    return _text_result(f"ok: wrote {len(content)} chars to memory.md")


# ── Server bundle ────────────────────────────────────────────────────────────

BOOKMARK_TOOLS = [
    search_bookmarks, get_stats, list_categories,
    list_bookmarks, show_bookmark, read_article, update_memory,
]
SERVER = create_sdk_mcp_server(name="bookmarks", version="0.1.0", tools=BOOKMARK_TOOLS)
TOOL_NAMES = [
    "mcp__bookmarks__search_bookmarks",
    "mcp__bookmarks__get_stats",
    "mcp__bookmarks__list_categories",
    "mcp__bookmarks__list_bookmarks",
    "mcp__bookmarks__show_bookmark",
    "mcp__bookmarks__read_article",
    "mcp__bookmarks__update_memory",
]


# ── System prompt ────────────────────────────────────────────────────────────

INSTRUCTIONS = """You are a personal knowledge assistant over a curated collection of X (Twitter) bookmarks.

The user has bookmarked tweets they found valuable. Many bookmarks point at long-form articles
(X-native or external blogs); their full bodies are pre-fetched to disk and accessible via
the read_article tool.

You have memory:
- Persistent durable facts live in memory.md (shown below). Always loaded into your prompt.
- When you learn a durable fact about the user (preferences, working style, environment),
  call update_memory with the FULL new file content. Replace, don't append in your head —
  re-write the whole file thoughtfully.
- Within-session and cross-session conversation history is retained automatically by the
  harness; you don't need a search tool for that.

Guidelines:
- For "find me a tweet about X" — search_bookmarks, return the most relevant with the URL.
- For "summarize / explain / what does X say" — read_article on the relevant tweet_id.
- For "do we have anything on Y" — search and give a direct yes/no with examples.
- Always include the tweet URL so the user can open it.
- Be concise. One good result beats ten mediocre ones.
"""


def build_system_prompt() -> str:
    memory = MEMORY_PATH.read_text() if MEMORY_PATH.exists() else "(empty)"
    return f"{INSTRUCTIONS}\n\n=== memory.md ===\n{memory}"


def build_options(continue_conv: bool = True) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=build_system_prompt(),
        model=MODEL,
        mcp_servers={"bookmarks": SERVER},
        # `tools` restricts what's advertised to the model; `allowed_tools` is the
        # permission whitelist. Set both to our MCP tools so no built-in Claude
        # Code tools (Bash, Edit, Write, Read) are ever in scope.
        tools=TOOL_NAMES,
        allowed_tools=TOOL_NAMES,
        # Don't load user/project/local settings — keep the agent self-contained.
        setting_sources=[],
        # Scope sessions to AGENT_DIR so continue_conversation doesn't collide
        # with whatever Claude Code session might exist in the project dir.
        cwd=str(AGENT_DIR),
        continue_conversation=continue_conv,
        permission_mode="default",
    )


# ── Render assistant messages ────────────────────────────────────────────────

def render_message(msg) -> None:
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text)
            elif isinstance(block, ToolUseBlock):
                # Show a compact trace so the user sees what's being called
                short = block.name.replace("mcp__bookmarks__", "")
                print(f"\033[2m[→ {short}({_short_args(block.input)})]\033[0m")


def _short_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:40] + "…"
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


# ── Modes ────────────────────────────────────────────────────────────────────

async def one_shot(question: str) -> None:
    options = build_options(continue_conv=True)
    async with ClaudeSDKClient(options=options) as client:
        await client.query(question)
        async for msg in client.receive_response():
            render_message(msg)


async def repl() -> None:
    print("Bookmark agent ready (Claude Agent SDK). Memory + cross-session continuation on.")
    print("Ctrl+C to quit.\n")
    # First turn resumes prior session; subsequent turns share this client's context.
    options = build_options(continue_conv=True)
    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                question = await asyncio.to_thread(input, "> ")
            except (KeyboardInterrupt, EOFError):
                print()
                return
            question = question.strip()
            if not question:
                continue
            await client.query(question)
            async for msg in client.receive_response():
                render_message(msg)
            print()


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        asyncio.run(one_shot(question))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
