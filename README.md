# twitter_bookmarks

Enrich and explore your X (Twitter) bookmarks. The base sync is handled by [field-theory](https://fieldtheory.dev) (`ft`). On top of it, this repo adds:

- **Article enrichment** — fetches the body (with images) of every linked article (external blogs + X-native long-form) and writes it as markdown.
- **Video tracing** — for every bookmarked video clip of a lecture, finds the original full-length lecture on YouTube.
- **Semantic graph** — embeds every article body, links related articles with `[[wiki-links]]` annotated by time offset, so the `~/.ft-bookmarks/articles/` directory becomes an Obsidian-compatible vault.
- **Chat (optional)** — a small Claude Agent SDK REPL with persistent memory that can answer questions over the enriched corpus.

## Layout

```
twitter_bookmarks/
├── sync_all.py             # one command to refresh everything
├── enrichment/             # data pipelines (the project's core)
│   ├── enrich_articles.py     # external article links → markdown bodies
│   ├── enrich_x_articles.py   # x.com/i/article/... → markdown via GraphQL
│   ├── enrich_videos.py       # video bookmarks → YouTube originals
│   └── build_graph.py         # embed articles, link related, write Obsidian vault
├── chat/                   # downstream consumer, dissociated from the pipelines
│   └── agent.py               # Claude Agent SDK REPL with memory
├── tools/                  # one-shot helpers (auth bridge, probes, fixers)
│   ├── x_cookies.mjs          # extract X auth cookies from Comet via ft's chrome-cookies
│   ├── probe_x_article.py     # capture X GraphQL hashes via Playwright
│   └── fix_frontmatter.py     # one-shot YAML repair
└── requirements.txt
```

Data lives outside the repo, in ft's standard data dir:

```
~/.ft-bookmarks/
├── bookmarks.jsonl                     # tweets (synced by ft)
├── bookmarks.db                        # ft's SQLite + FTS5 index
├── articles/
│   ├── <tweetId>/<author-slug>.md      # article bodies (frontmatter + body + Related)
│   ├── <tweetId>/images/...
│   ├── INDEX.md                        # chronological list, [[wiki-links]]
│   └── .embeddings.npz                 # build_graph cache
├── videos/
│   ├── traces.jsonl                    # one row per attempted bookmark, every status
│   └── <tweetId>/source.md             # sidecar for traced lectures
└── agent/
    ├── history/<sessionId>.jsonl       # raw turn log per agent invocation
    ├── memory/
    │   ├── user.md                     # WHO (always in system prompt)
    │   ├── facts.md                    # durable facts (always in system prompt)
    │   └── episodic/<sessionId>.md     # session summaries (lazy via recall tool)
    └── scratchpad/                     # transient task notes
```

## Install

Prereqs:

- Python 3.11+
- Node 20+ with `npm` (for `ft` and the cookie bridge)
- macOS keychain access (X-native articles need cookies from your logged-in browser; this repo assumes [Comet](https://comet.perplexity.ai/), which `ft` already supports)

```bash
# Field-theory CLI
npm install -g fieldtheory
ft sync                                  # initial bookmark pull (one-time)

# Claude CLI (used by claude-agent-sdk for the chat REPL)
npm install -g @anthropic-ai/claude-code

# Python deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Anthropic API key in .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

First-time X-native article fetch will trigger a macOS keychain prompt asking for permission to read Comet's Safe Storage. Accept it once.

## Usage

### Refresh everything

```bash
.venv/bin/python sync_all.py
```

Runs, in order: `ft sync` → external articles → X-native articles → video traces → graph build.

```bash
python sync_all.py --skip sync           # skip ft sync if you just synced
python sync_all.py --skip videos         # opt-out flags (repeatable)
python sync_all.py --only graph          # rebuild a single stage
```

Each stage is idempotent — re-runs skip already-enriched bookmarks.

### Run pipelines individually

```bash
python enrichment/enrich_articles.py          # external article bodies
python enrichment/enrich_x_articles.py        # X-native articles (cookies needed)
python enrichment/enrich_videos.py            # YouTube traces
python enrichment/build_graph.py              # embeddings + Obsidian links
```

Common flags across enrichers: `--tweet-id <id>` (one bookmark), `--limit N`, `--force` (re-fetch even if cached).

### Browse in Obsidian

`File → Open folder as vault → ~/.ft-bookmarks/articles`. The folder is hidden (starts with `.`); in the macOS file picker, press **Cmd+Shift+.** to show hidden folders.

### Chat with the corpus

```bash
.venv/bin/python chat/agent.py                    # interactive REPL
.venv/bin/python chat/agent.py "your question"    # one-shot
```

Tools: `search_bookmarks`, `read_article`, `show_bookmark`, `list_bookmarks`, `get_stats`, `list_categories`, plus four memory tools — `update_user`, `update_facts`, `pin_fact`, `recall`.

Memory is tiered (inspired by Xu et al., [arXiv:2512.05470](https://arxiv.org/abs/2512.05470)):

- **Always in the system prompt:** `memory/user.md` (who you are) and `memory/facts.md` (durable atomic facts).
- **Lazy via `recall(query)`:** `memory/episodic/<sessionId>.md` — per-session summaries written by a Haiku call when a session ends with more than 30 user turns.
- **Raw log:** every prompt, tool call, tool result, and assistant text block lands in `history/<sessionId>.jsonl`.

`continue_conversation=True` keeps the SDK's within-session message history across invocations; the tiered memory is what survives compactions and resets.

## Known gotchas

- **`ft` skips X URLs** in its own article enricher (`bookmark-enrich.js`). That's why this repo's `enrich_x_articles.py` exists — it routes around that limit using authenticated GraphQL.
- **X GraphQL hash rotation.** The `TWEET_QUERY_HASH` constant in `enrichment/enrich_x_articles.py` is a captured hash. X rotates these every few weeks. When article fetches start failing, re-run `tools/probe_x_article.py` to capture a fresh hash and update the constant.
- **`tools/probe_x_article.py` requires Playwright + Chromium** (`.venv/bin/playwright install chromium`) — only needed when refreshing the GraphQL hash.
- **Cookie bridge requires Comet to be installed** with an active X login. To switch browsers, edit the `browserId` arg in `tools/x_cookies.mjs`.
- **Memory files at `~/.ft-bookmarks/agent/memory/{user,facts}.md`** are always injected into the agent's system prompt. Edit them by hand to seed; the agent rewrites them via `update_user` / `update_facts` / `pin_fact` when it learns something durable. Episodic session summaries are written automatically when a session crosses 30 turns; tune `COMPRESSION_TURN_THRESHOLD` in `chat/agent.py` if needed.
- **Legacy `memory.md` migration** runs once on first startup of the new agent — sections under `## User` move to `user.md`, `## Environment` / `## Conventions` / `## Facts` move to `facts.md`, and the old file is renamed to `memory.md.bak`.

## What's where

| You want to... | Look at |
| --- | --- |
| Fetch new bookmarks | `ft sync` (or `sync_all.py`) |
| Add a new enrichment kind | new script in `enrichment/`, add to `STAGES` in `sync_all.py` |
| Change the agent's tools | `chat/agent.py` — `@tool` decorators on the bookmark functions |
| Edit the agent's persistent memory | `~/.ft-bookmarks/agent/memory/user.md` and `facts.md` |
| See what was said in a past session | `~/.ft-bookmarks/agent/history/<sessionId>.jsonl` (raw) or `memory/episodic/<sessionId>.md` (summary) |
| Re-render the Obsidian graph | `python sync_all.py --only graph` |
| Refresh the X auth cookies | edit `tools/x_cookies.mjs` (browserId), or run `node tools/x_cookies.mjs` |
| Update the X GraphQL hash | `python tools/probe_x_article.py`, then edit `enrichment/enrich_x_articles.py:TWEET_QUERY_HASH` |
