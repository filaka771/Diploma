import json
import csv

INPUT_JSON = "theonion_news_articles.json"
OUTPUT_CSV = "theonion_news_articles.csv"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT_CSV, "w", encoding="utf-8", newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    writer.writerow(["url", "title", "content"])
    
    for article in data:
        writer.writerow([
            article.get("url", ""),
            article.get("title", "").replace("\n", " ").strip(),
            article.get("content", "").replace("\n", " ").strip()
        ])

print(f"Converted to CSV: {OUTPUT_CSV}")
