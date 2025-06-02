import os
import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description="Website Crawler for Text Extraction")
parser.add_argument("url", help="Base URL of the website to crawl (e.g., https://example.com)")
args = parser.parse_args()

base_url = args.url.rstrip("/")
output_dir = "docs"
max_depth = 2

visited = set()
os.makedirs(output_dir, exist_ok=True)

# ----------------------
# Utility Functions
# ----------------------

def is_internal(url):
    return url.startswith("/") or url.startswith(base_url)

def normalize_url(url):
    full_url = urljoin(base_url, url.split("#")[0])
    return full_url.rstrip("/")

def filename_from_url(url):
    parsed = urlparse(url)
    if parsed.path in ["", "/"]:
        return "home.txt"
    clean_path = parsed.path.strip("/").replace("/", "_").replace("?", "_")
    return clean_path + ".txt"

def clean_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return "\n".join(
        line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()
    )

# ----------------------
# Crawler Function
# ----------------------

def crawl(url, depth=0):
    norm_url = normalize_url(url)
    if norm_url in visited or depth > max_depth:
        return
    visited.add(norm_url)

    try:
        response = requests.get(norm_url, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Skipped (non-200): {norm_url}")
            return

        soup = BeautifulSoup(response.text, "html.parser")
        text = clean_text(response.text)
        filename = filename_from_url(norm_url)
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved: {filename}")
        else:
            print(f"🔁 Skipped (already saved): {filename}")

        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]
            if is_internal(link):
                crawl(link, depth + 1)

    except Exception as e:
        print(f"❌ Error at {norm_url}: {e}")

# ----------------------
# Start Crawl
# ----------------------

if __name__ == "__main__":
    crawl(base_url)
