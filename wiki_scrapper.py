import wikipediaapi
import requests
import json
import csv
import time

# Initialize Wikipedia API wrapper
wiki_wiki = wikipediaapi.Wikipedia("vi")

# Initialize counter
counter = 0

# Open a CSV file to write summaries
with open('article_summaries.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f, quotechar='"', quoting=csv.QUOTE_ALL)
    
    # Write the header
    writer.writerow(["Title", "Summary"])
    
    while counter < 5:
        try:
            # Fetch a random Vietnamese Wikipedia article
            random_article_url = "https://vi.wikipedia.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=1&format=json"
            response = requests.get(random_article_url)
            data = json.loads(response.text)

            # Extract article title
            title = data['query']['random'][0]['title']

            # Fetch article
            page = wiki_wiki.page(title)

            # Check if page exists
            if not page.exists():
                print(f"Article {title} does not exist. Skipping.")
                continue

            # Write plain text summary to CSV file
            writer.writerow([page.title, page.summary])

            # Increment counter
            counter += 1

            # Print status
            print(f"Fetched {counter} articles")

            # Add delay to avoid rate-limiting
            time.sleep(1)

        except Exception as e:
            print(f"An error occurred: {e}")
