"""
scraper.py
==========
Recursively scrapes ALL content from:
  - https://handbook.gitlab.com/
  - https://about.gitlab.com/direction/

Saves:
  data/scraped_pages.json  — raw pages  (url, title, text)
  data/chunks.json         — chunked docs ready for embedding

Usage:
  python scraper.py
"""

import os
import json
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# ─── Settings ────────────────────────────────────────────────────────────────
SEED_URLS = [
    "https://handbook.gitlab.com/",
    "https://about.gitlab.com/direction/",
]

ALLOWED_DOMAINS = {
    "handbook.gitlab.com",
    "about.gitlab.com",
}

DIRECTION_PREFIX = "/direction/"

MAX_PAGES     = 500
REQUEST_DELAY = 0.6
CHUNK_SIZE    = 400       # words per chunk
CHUNK_OVERLAP = 60        # overlapping words between chunks

OUTPUT_DIR  = "data"
PAGES_FILE  = "data/scraped_pages.json"
CHUNKS_FILE = "data/chunks.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GitLabChatbotScraper/1.0; educational-project)"
    )
}

# ─── URL Filtering ────────────────────────────────────────────────────────────

def is_allowed(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc not in ALLOWED_DOMAINS:
        return False
    if p.netloc == "about.gitlab.com":
        return p.path.startswith(DIRECTION_PREFIX)
    return True


def normalise(url: str) -> str:
    return url.split("#")[0].rstrip("/")


# ─── Extraction ───────────────────────────────────────────────────────────────

NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "noscript", "svg", "img", "button",
    "form", "iframe", "figure", "meta", "link",
]

TEXT_TAGS = [
    "h1", "h2", "h3", "h4", "h5", "h6",
    "p", "li", "td", "th", "blockquote", "pre", "code",
]


def extract_text(soup: BeautifulSoup) -> str:
    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="content")
        or soup.find(class_="main-content")
        or soup.body
    )
    if main is None:
        return ""

    lines = []
    for elem in main.find_all(TEXT_TAGS):
        text = elem.get_text(separator=" ", strip=True)
        if text and len(text) > 10:
            lines.append(text)

    return "\n".join(lines)


def extract_links(base_url: str, soup: BeautifulSoup) -> list:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href).split("#")[0]
        if abs_url and is_allowed(abs_url):
            links.append(abs_url)
    return list(set(links))


def get_title(soup: BeautifulSoup) -> str:
    tag = soup.find("title")
    if tag:
        return tag.get_text(strip=True).split("|")[0].strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else "Untitled"


# ─── Crawler ──────────────────────────────────────────────────────────────────

def crawl() -> list:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visited = set()
    queue   = list(SEED_URLS)
    pages   = []

    print(f"\n Spider starting | max={MAX_PAGES} pages | delay={REQUEST_DELAY}s")
    print(f"   Seeds: {SEED_URLS}\n")

    while queue and len(visited) < MAX_PAGES:
        url  = queue.pop(0)
        norm = normalise(url)

        if norm in visited:
            continue
        visited.add(norm)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
        except requests.exceptions.RequestException as e:
            print(f"  ERROR {url}: {e}")
            continue

        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}  {url}")
            continue
        if "text/html" not in resp.headers.get("Content-Type", ""):
            continue

        soup  = BeautifulSoup(resp.text, "html.parser")
        title = get_title(soup)
        text  = extract_text(soup)

        if len(text.strip()) < 150:
            continue

        page_id = hashlib.md5(url.encode()).hexdigest()[:8]
        pages.append({"id": page_id, "url": url, "title": title, "text": text})

        new_links = extract_links(url, soup)
        queue.extend([l for l in new_links if normalise(l) not in visited])

        print(f"  OK [{len(pages):>3}]  {title[:70]}")

        time.sleep(REQUEST_DELAY)

    print(f"\nDone — {len(pages)} pages scraped.")
    return pages


# ─── Chunker ──────────────────────────────────────────────────────────────────

def chunk_page(page: dict, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list:
    words  = page["text"].split()
    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end        = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "id":    f"{page['id']}_c{idx:03d}",
            "url":   page["url"],
            "title": page["title"],
            "text":  chunk_text,
        })
        idx   += 1
        start += chunk_size - overlap
        if start >= len(words):
            break

    return chunks


def build_all_chunks(pages: list) -> list:
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    return all_chunks


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pages = crawl()

    with open(PAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved pages  -> {PAGES_FILE}  ({len(pages)} docs)")

    chunks = build_all_chunks(pages)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks -> {CHUNKS_FILE}  ({len(chunks)} chunks)")
    print("\nNext step: python embed.py")
