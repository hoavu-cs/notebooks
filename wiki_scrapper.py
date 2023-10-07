import wikipedia
import requests
import json
import csv
import time
import pandas as pd

# Constants
wikipedia.set_lang("vi")
SIZE = 500000
BATCH_SIZE = 100  # number of articles to fetch in one request
cache = set()

def fetch_random_articles(batch_size):
    random_article_url = f"https://vi.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit={batch_size}&format=json"
    response = requests.get(random_article_url)
    data = json.loads(response.text)
    return [item['title'] for item in data['query']['random']]

def fetch_article_content(title):
    try:
        content = wikipedia.page(title).summary
        return (title, content)
    except wikipedia.exceptions.DisambiguationError:
        print(f"Disambiguation page found for {title}. Skipping.")
    except wikipedia.exceptions.PageError:
        print(f"Article {title} does not exist. Skipping.")
    return None

def main():
    with open('article_summaries.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["Title", "Summary"])

        total_batches = SIZE // BATCH_SIZE

        for batch_number in range(total_batches):
            titles = fetch_random_articles(BATCH_SIZE)
            articles = []
            for title in titles:
                if title not in cache:
                    cache.add(title)
                    articles.append(fetch_article_content(title))
            for article in articles:
                if article:
                    writer.writerow(article)
            print(f"Fetched {(batch_number + 1) * BATCH_SIZE} articles")

if __name__ == "__main__":
    main()

df = pd.read_csv('article_summaries.csv')
print(df.head())
