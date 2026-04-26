"""
One-shot fixer: re-quote multi-line YAML frontmatter values that were emitted
as raw multiline (broken) by an earlier version of the pipeline.

Re-emits frontmatter with any value containing a newline or special chars
JSON-quoted on a single line. Idempotent.
"""

import json
import re
from pathlib import Path

ARTICLES = Path.home() / ".ft-bookmarks" / "articles"
KEY_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$")
FM_RE = re.compile(r"\A(---\n)(.*?)(\n---\n)(.*)", re.DOTALL)


def needs_quoting(v: str) -> bool:
    if not v:
        return False
    if "\n" in v:
        return True
    # YAML doesn't tolerate leading/trailing whitespace, : in odd places, etc.
    if v.lstrip() != v or v.rstrip() != v:
        return True
    return False


def fix_one(path: Path) -> bool:
    text = path.read_text()
    m = FM_RE.match(text)
    if not m:
        return False
    head, fm_body, tail, body = m.groups()

    pairs: list[tuple[str, str]] = []
    cur_key: str | None = None
    cur_val_lines: list[str] = []

    def flush():
        if cur_key is not None:
            val = "\n".join(cur_val_lines).rstrip()
            pairs.append((cur_key, val))

    for line in fm_body.split("\n"):
        km = KEY_RE.match(line)
        if km:
            flush()
            cur_key = km.group(1)
            cur_val_lines = [km.group(2)]
        else:
            cur_val_lines.append(line)
    flush()

    out_lines = []
    for k, v in pairs:
        if needs_quoting(v):
            v = json.dumps(v, ensure_ascii=False)
        out_lines.append(f"{k}: {v}")

    new_text = head + "\n".join(out_lines) + tail + body
    if new_text != text:
        path.write_text(new_text)
        return True
    return False


def main():
    fixed = 0
    total = 0
    for md in ARTICLES.glob("*/*.md"):
        total += 1
        if fix_one(md):
            fixed += 1
    print(f"scanned {total} files, fixed {fixed}")


if __name__ == "__main__":
    main()
