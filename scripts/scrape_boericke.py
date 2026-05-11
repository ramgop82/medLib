"""
Scrape Boericke's Materia Medica from homeoint.org (public domain).
Saves each remedy as a text file in data/boericke/
"""

import os
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://www.homeoint.org/books/boericmm"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "boericke")
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def get_remedy_links():
    """Get all remedy page links from alphabetical index pages."""
    all_links = []

    for letter in ALPHABET:
        url = f"{BASE_URL}/{letter}.htm"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                href = a["href"]
                name = a.get_text(strip=True)
                if "/" in href and href.endswith(".htm") and "index" not in href:
                    full_url = f"{BASE_URL}/{href}"
                    all_links.append({"name": name, "url": full_url})

            time.sleep(0.5)
        except Exception as e:
            print(f"  Error on {letter}: {e}")

    return all_links


def scrape_remedy(url: str) -> str:
    """Scrape a single remedy page and return clean text."""
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching remedy links from index pages...")
    links = get_remedy_links()
    print(f"Found {len(links)} remedies.")

    print("\nScraping remedy pages...")
    for i, remedy in enumerate(links, 1):
        name = remedy["name"]
        url = remedy["url"]

        # Create safe filename
        safe_name = name.replace("/", "-").replace(" ", "_").lower()
        filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

        if os.path.exists(filepath):
            continue

        text = scrape_remedy(url)
        if text:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

        if i % 50 == 0:
            print(f"  Progress: {i}/{len(links)}")

        time.sleep(0.3)

    print(f"\nDone! Scraped remedies saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
