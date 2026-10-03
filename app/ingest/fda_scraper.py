"""Polite FDA warning-letter fetcher.

Identifies itself honestly, honors robots.txt, sleeps between requests, and
aborts on the first sign of bot-protection (never tries to evade it).
"""
import os
import re
import sys
import time
import urllib.robotparser
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.fda.gov"
LIST_URL = (
    BASE + "/inspections-compliance-enforcement-and-criminal-investigations/"
    "compliance-actions-and-activities/warning-letters"
)
# Identify honestly to FDA: set SCRAPER_CONTACT (e.g. an email) in .env; it isn't committed.
UA = f"regulatory-qa-portfolio/0.1 (personal research project; {os.environ.get('SCRAPER_CONTACT', 'contact unset')})"
DELAY_S = 8
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_LIMIT = 10
MAX_DOCS = 15  # hard cap on documents kept in RAW_DIR


class Blocked(RuntimeError):
    pass


def _get(client: httpx.Client, url: str) -> httpx.Response:
    r = client.get(url, follow_redirects=False)
    body = r.text[:2000].lower()
    if r.status_code in (301, 302, 303, 307, 308) and "apology" in r.headers.get("location", "").lower():
        raise Blocked(f"redirected to bot-protection page for {url}")
    if "abuse-detection" in body or "excessive-requests" in body or r.status_code in (403, 429):
        raise Blocked(f"blocked ({r.status_code}) for {url}")
    r.raise_for_status()
    return r


def allowed(url: str) -> bool:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(BASE + "/robots.txt")
    try:
        rp.read()
    except Exception:
        return False  # fail closed
    return rp.can_fetch(UA, url)


def letter_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.select("a[href*='/warning-letters/']"):
        href = a["href"]
        if re.search(r"/warning-letters/[a-z0-9-]+-\d{5,}-\d{8}$", href):
            out.append(href if href.startswith("http") else BASE + href)
    return list(dict.fromkeys(out))


def main(limit: int = DEFAULT_LIMIT) -> None:
    limit = min(limit, MAX_DOCS)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in RAW_DIR.glob("*.html")}
    if len(have) >= limit:
        print(f"already have {len(have)} documents (target {limit}) — nothing to fetch")
        return
    if not allowed(LIST_URL):
        sys.exit("robots.txt disallows (or could not be read) — stopping")
    with httpx.Client(headers={"User-Agent": UA}, timeout=30) as client:
        try:
            page = _get(client, LIST_URL)
            links = letter_links(page.text)
            print(f"found {len(links)} letter links on listing page; have {len(have)}, target {limit}")
            for url in links:
                if len(have) >= limit:
                    break
                slug = url.rsplit("/", 1)[-1]
                if slug in have:
                    continue
                time.sleep(DELAY_S)
                if not allowed(url):
                    print("skip (robots):", url)
                    continue
                r = _get(client, url)
                path = RAW_DIR / (slug + ".html")
                path.write_text(r.text)
                have.add(slug)
                print("saved", path.name)
        except Blocked as e:
            sys.exit(f"STOPPED: {e}")
    print(f"done: {len(have)} documents in {RAW_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LIMIT)
