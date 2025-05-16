import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import random

BASE_URL = "https://clickhole.com/category/news/page/{}/"
TOTAL_PAGES = 375
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
MAX_WORKERS = 10
OUTPUT_FILE = "clickhole_news_full_articles.json"

# ---------- Retry-enabled article fetch ----------

def fetch_with_retries(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            return res.text
        except Exception as e:
            print(f"[Retry {attempt+1}] {url}: {e}")
            time.sleep(delay + random.random())
    return None

# ---------- Scrape full article ----------

def scrape_article_content(article_url):
    html = fetch_with_retries(article_url)
    if not html:
        print(f"[Article Error] {article_url}: failed after retries.")
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")

        # Title
        title_tag = soup.find("h1", class_="entry-title") or soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else "No title"

        # Try multiple common content containers
        content_container = soup.find("div", class_="entry-content") or soup.find("div", class_="post-content")
        if not content_container:
            print(f"[No Content Container] {article_url}")
            return {"url": article_url, "title": title, "content": ""}

        paragraphs = content_container.find_all("p")
        content = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        return {
            "url": article_url,
            "title": title,
            "content": content
        }

    except Exception as e:
        print(f"[Article Error] {article_url}: {e}")
        return None

# ---------- Get article links ----------

def get_article_links_from_page(page_num):
    url = BASE_URL.format(page_num)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        articles = soup.find_all("article")
        links = [a["href"] for article in articles if (a := article.find("a", href=True))]
        print(f"[Page {page_num}] Found {len(links)} links")
        return links

    except Exception as e:
        print(f"[Page {page_num}] Error: {e}")
        return []

# ---------- Main ----------

def main():
    start = time.time()

    print("Collecting article links...")
    all_links = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(get_article_links_from_page, i) for i in range(1, TOTAL_PAGES + 1)]
        for future in as_completed(futures):
            links = future.result()
            all_links.extend(links)

    print(f"Total article links found: {len(all_links)}")

    print("Scraping article content...")
    articles_data = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(scrape_article_content, url) for url in all_links]
        for future in as_completed(futures):
            data = future.result()
            if data:
                articles_data.append(data)

    print(f"Scraped {len(articles_data)} articles in {time.time() - start:.2f} seconds.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
