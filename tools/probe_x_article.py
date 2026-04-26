"""
Probe one X-native article: load with cookies, capture the GraphQL response.

Goal is to learn (1) the actual operation name X uses to fetch a single article,
(2) the response schema, so we can build a real fetcher.
"""

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ARTICLE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://x.com/i/article/2046382777705390080"
OUT_DIR = Path("/tmp/xart_probe")
OUT_DIR.mkdir(exist_ok=True)


def get_cookies():
    raw = subprocess.check_output(["node", "tools/x_cookies.mjs"])
    data = json.loads(raw)
    parts = {}
    for kv in data["cookieHeader"].split("; "):
        k, _, v = kv.partition("=")
        parts[k] = v
    return parts


async def main():
    cookies_dict = get_cookies()
    cookies = [
        {"name": k, "value": v, "domain": ".x.com", "path": "/", "secure": True}
        for k, v in cookies_dict.items()
    ]

    captured = []  # list of (url, status, body)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context()
        await ctx.add_cookies(cookies)

        async def on_response(resp):
            url = resp.url
            if "/i/api/graphql/" not in url:
                return
            # Match operation name from URL path
            m = re.search(r"/graphql/[^/]+/([A-Za-z0-9_]+)", url)
            op = m.group(1) if m else "?"
            try:
                body = await resp.text()
            except Exception:
                body = ""
            captured.append((op, resp.status, url, body))

        page = await ctx.new_page()
        page.on("response", on_response)
        try:
            await page.goto(ARTICLE_URL, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"goto warning: {e}")
        # Wait for graphql traffic to settle
        await page.wait_for_timeout(8000)
        await browser.close()

    print(f"captured {len(captured)} graphql responses")
    for op, status, url, body in captured:
        print(f"  [{status}] {op}  ({len(body)} bytes)")
    print()
    print("URL DETAILS:")
    for op, status, url, body in captured:
        if op in ("TweetResultByRestId", "ArticleRedirectScreenQuery"):
            print(f"  {op}:")
            print(f"    {url}")

    # Save anything that mentions article-y content
    for op, status, url, body in captured:
        if "article" in op.lower() or "article" in body[:5000].lower():
            fname = OUT_DIR / f"{op}.json"
            try:
                parsed = json.loads(body)
                fname.write_text(json.dumps(parsed, indent=2))
                print(f"  -> saved {fname}")
            except json.JSONDecodeError:
                fname.write_text(body)
                print(f"  -> saved {fname} (not JSON)")


if __name__ == "__main__":
    asyncio.run(main())
