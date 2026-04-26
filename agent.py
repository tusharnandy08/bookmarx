#!/usr/bin/env python3
"""
Bookmark agent — ask natural language questions about your X bookmarks.

Usage:
  python agent.py "get me the tweet about KV cache optimization"
  python agent.py "do we have anything on YC advice?"
  python agent.py          # interactive mode
"""

import subprocess
import sys
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)
import anthropic

BOOKMARKS_PATH = os.path.expanduser("~/.ft-bookmarks/bookmarks.jsonl")
ARTICLES_DIR = os.path.expanduser("~/.ft-bookmarks/articles")

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

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
]


def run_ft(*args) -> str:
    result = subprocess.run(
        ["ft", *args],
        capture_output=True,
        text=True,
    )
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
    path = os.path.join(ARTICLES_DIR, tweet_id, "body.md")
    if not os.path.exists(path):
        return f"No enriched article on disk for tweet {tweet_id}. (Either it has no link, the link was skipped, or enrichment failed.)"
    with open(path) as f:
        return f.read()


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "search_bookmarks":
        return search_bookmarks(inputs["query"], inputs.get("limit", 10))
    elif name == "get_stats":
        return get_stats()
    elif name == "list_categories":
        return list_categories()
    elif name == "list_bookmarks":
        return list_bookmarks(inputs.get("author"), inputs.get("limit", 20))
    elif name == "show_bookmark":
        return show_bookmark(inputs["tweet_id"])
    elif name == "read_article":
        return read_article(inputs["tweet_id"])
    return f"Unknown tool: {name}"


SYSTEM = """You are a personal knowledge assistant for a curated collection of X (Twitter) bookmarks.

The user has saved tweets they found valuable. Many bookmarks point at long-form articles
(X-native articles or external blogs); their full bodies have been pre-fetched to disk
and are available via the read_article tool.

Your job is to help them retrieve and reason over this knowledge base.

Guidelines:
- For "find me a tweet about X" — search and return the most relevant result with the URL.
- For "do we have anything on Y" — search and give a direct yes/no with examples if found.
- For broad topic questions — search multiple related terms, synthesize what's in the collection.
- For "summarize / explain / what does X say about ..." — use read_article on the relevant
  tweet_id to pull the full article body, not just the tweet snippet.
- Always include the tweet URL so the user can open it.
- Be concise. One good result is better than ten mediocre ones.
- If nothing is found after a couple searches, say so plainly.
"""


def run_agent(question: str) -> str:
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        # Process tool calls
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
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


def interactive():
    print("Bookmark agent ready. Ask anything about your bookmarks. (Ctrl+C to quit)\n")
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
