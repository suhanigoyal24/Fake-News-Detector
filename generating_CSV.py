# dynamic_news_scraper.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- SOURCES ----------------
real_sources = [
    "https://www.hindustantimes.com/",
    "https://www.indiatoday.in/",
    "https://www.ndtv.com/latest",
    "https://www.dnaindia.com/",
    "https://www.bhaskar.com/",  # Dainik Bhaskar
    "https://timesofindia.indiatimes.com/",
    "https://www.reuters.com/world/",
    "https://www.bbc.com/news"
]

fake_sources = [
    "https://www.altnews.in/fake-news/",
    "https://www.boomlive.in/fact-check/",
    "https://www.factchecker.in/",
    "https://www.thequint.com/webqoof",
    "https://www.indiatoday.in/fact-check",
    "https://factcheck.pib.gov.in/",
    "https://www.snopes.com/fact-check/",
    "https://factcheck.afp.com/",
    "https://www.reuters.com/fact-check/"
]

# ---------------- SCRAPER ----------------
def scrape_url(url, label):
    """
    Scrapes a single URL for article titles and URLs.
    """
    articles = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        for link in soup.find_all("a", href=True):
            title = link.get_text(strip=True)
            if len(title) < 8:
                continue

            href = link["href"]
            full_url = href if href.startswith("http") else urljoin(url, href)

            articles.append({
                "text": title,
                "label": label,
                "article_url": full_url
            })

    except Exception as e:
        print(f"[ERROR] Failed to scrape {url} -> {e}")

    return articles

# ---------------- DYNAMIC SCRAPING ----------------
def scrape_all(sources, label, max_workers=8):
    """
    Scrape all sources in parallel using ThreadPoolExecutor.
    """
    all_articles = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_url, url, label): url for url in sources}
        for future in as_completed(future_to_url):
            articles = future.result()
            all_articles.extend(articles)

    return all_articles

# ---------------- LOAD EXISTING CSV ----------------
def load_existing_csv(path="news_dataset.csv"):
    try:
        df_existing = pd.read_csv(path, encoding="utf-8")
        df_existing = df_existing.loc[:, ~df_existing.columns.duplicated()]
        df_existing.reset_index(drop=True, inplace=True)
        return df_existing
    except FileNotFoundError:
        return pd.DataFrame(columns=["text", "label", "article_url"])

# ---------------- SAVE CSV ----------------
def save_csv(df, path="news_dataset.csv"):
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"✔ CSV updated successfully with {len(df)} articles!")

# ---------------- MAIN ----------------
def main():
    print("📰 Scraping REAL news sources...")
    real_articles = scrape_all(real_sources, "real")
    print("📰 Scraping FAKE news sources...")
    fake_articles = scrape_all(fake_sources, "fake")

    # Combine and remove duplicates
    new_df = pd.DataFrame(real_articles + fake_articles)
    new_df.drop_duplicates(subset=["text", "article_url"], inplace=True)
    new_df.reset_index(drop=True, inplace=True)

    # Merge with existing CSV dynamically
    df_existing = load_existing_csv()
    df_combined = pd.concat([df_existing, new_df], ignore_index=True)
    df_combined.drop_duplicates(subset=["text", "article_url"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    save_csv(df_combined)

if __name__ == "__main__":
    main()
