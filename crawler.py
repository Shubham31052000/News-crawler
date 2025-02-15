import os
import requests
from bs4 import BeautifulSoup
import schedule
import time
import subprocess

# Crawl website and generate HTML
def crawl_and_generate_html():
    base_url = "https://www.visive.ai/news"
    visited_urls = set()
    data = []

    def crawl(url):
        if url in visited_urls:
            return
        visited_urls.add(url)

        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title, URL, and published date
            title = soup.title.string if soup.title else "No Title"
            published_date = soup.find('meta', {'itemprop': 'datePublished'})
            date = published_date['content'] if published_date else "Unknown Date"

            data.append({'title': title, 'url': url, 'date': date})

            # Find and crawl links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    href = base_url + href
                if base_url in href and href not in visited_urls:
                    crawl(href)
        except Exception as e:
            print(f"Failed to crawl {url}: {e}")

    crawl(base_url)

    # Generate HTML
    with open("index.html", "w") as file:
        file.write("<html><head><title>News</title></head><body>")
        file.write("<h1>Latest News</h1>")
        file.write("<ul>")
        for item in data:
            file.write(f"<li><a href='{item['url']}'>{item['title']}</a> - {item['date']}</li>")
        file.write("</ul>")
        file.write("</body></html>")

    print("HTML file updated.")

# Push changes to GitHub
def push_to_github():
    try:
        repo_path = os.getcwd()
        os.chdir(repo_path)

        subprocess.run(["git", "add", "index.html"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update index.html"], check=True)
        subprocess.run(["git", "push"], check=True)

        print("Changes pushed to GitHub successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to push changes: {e}")

# Full automation workflow
def run_crawler():
    crawl_and_generate_html()
    push_to_github()

# Schedule the script to run every 6 hours
schedule.every(6).hours.do(run_crawler)

print("Scheduler is running. Press Ctrl+C to stop.")
while True:
    schedule.run_pending()
    time.sleep(1)
