import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import random

BASE_URL = "https://www.theonion.com/latest"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
MAX_WORKERS = 5
OUTPUT_FILE = "theonion_articles.json"
DELAY = 1  
MAX_PAGES = 50  

def fetch_with_retries(url, retries=3):
    for attempt in range(retries):
        try:
            time.sleep(DELAY * (attempt + 1) * random.uniform(0.5, 1.5))
            res = requests.get(url, headers=HEADERS, timeout=10)
            res.raise_for_status()
            return res.text
        except Exception as e:
            print(f"[Retry {attempt+1}] {url}: {e}")
    return None

def scrape_article_content(article_url):
    html = fetch_with_retries(article_url)
    if not html:
        return None

    try:
        soup = BeautifulSoup(html, 'html.parser')
        title = None
        for selector in ['h1.sc-1efpnfq-0', 'h1.headline', 'h1']:
            title_tag = soup.select_one(selector)
            if title_tag:
                title = title_tag.get_text(strip=True)
                break
        if not title:
            print(f"[No Title] {article_url}")
            return None

        content = ""
        for selector in ['div.js_post-content', 'div.post-content', 'article']:
            content_div = soup.select_one(selector)
            if content_div:
                paragraphs = content_div.find_all('p')
                content = '\n'.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                break
        if not content:
            print(f"[No Content] {article_url}")
            return None

        return {
            'url': article_url,
            'title': title,
            'content': content
        }
    except Exception as e:
        print(f"[Error] {article_url}: {str(e)}")
        return None

def get_article_links(page_url):
    html = fetch_with_retries(page_url)
    if not html:
        return []

    soup = BeautifulSoup(html, 'html.parser')
    links = set()

    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/') and 'theonion.com' not in href:
            full_url = 'https://theonion.com' + href
            links.add(full_url)

    return list(links)

def main():
    print("Starting The Onion scraper with pagination...")

    all_article_links = set()
    for page_num in range(1, MAX_PAGES + 1):
        page_url = f"{BASE_URL}?page={page_num}"
        print(f"Fetching page {page_num} - {page_url}")
        links = get_article_links(page_url)
        if not links:
            print(f"[No Links Found] on page {page_num}, stopping early.")
            break
        print(f"Found {len(links)} links on page {page_num}")
        all_article_links.update(links)
        time.sleep(DELAY * random.uniform(1, 2))  # Respectful delay

    print(f"Total unique articles to scrape: {len(all_article_links)}")

    articles = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scrape_article_content, url): url for url in all_article_links}
        for future in as_completed(futures):
            article = future.result()
            if article:
                articles.append(article)
                print(f"Scraped: {article['title']}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(articles)} articles to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
