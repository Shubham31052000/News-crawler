# import os
# import requests
# from bs4 import BeautifulSoup
# import re
# from datetime import datetime
# import subprocess

# # Crawl website and generate HTML
# def crawl_and_generate_html():
#     base_url = "https://www.visive.ai"
#     news_url = f"{base_url}/news"
#     visited_urls = set()
#     data = []
#     failed_urls = []
#     # Regex for valid URLs
#     valid_url_pattern = re.compile(r"^https://www\.visive\.ai/news/[a-zA-Z0-9\-]+$")

#     def extract_published_date(html):
#         # Check for schema.org date published
#         date_match = re.search(r'<meta\s+itemprop="datePublished"\s+content="([^"]*)"', html)
#         if date_match:
#             return date_match.group(1)
        
#         # Check for <time> tag with pubdate attribute
#         time_match = re.search(r'<time[^>]*pubdate[^>]*datetime="([^"]*)"[^>]*>', html)
#         if time_match:
#             return time_match.group(1)
        
#         # Look for common date patterns
#         date_patterns = [
#             r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",  # YYYY-MM-DD or YYYY/MM/DD
#             r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})",  # DD/MM/YYYY
#             r"(\w{3,9})\s+(\d{1,2}),\s+(\d{4})",  # Month DD, YYYY
#         ]
#         for pattern in date_patterns:
#             match = re.search(pattern, html)
#             if match:
#                 if pattern == date_patterns[0]:
#                     return f"{match.group(1)}-{int(match.group(2)):02}-{int(match.group(3)):02}"
#                 elif pattern == date_patterns[1]:
#                     return f"{int(match.group(3)):04}-{int(match.group(2)):02}-{int(match.group(1)):02}"
#                 elif pattern == date_patterns[2]:
#                     month = datetime.strptime(match.group(1), "%B").month
#                     return f"{match.group(3)}-{month:02}-{int(match.group(2)):02}"
#         return ""

#     def crawl(url):
#         if url in visited_urls:
#             return
#         visited_urls.add(url)

#         try:
#             response = requests.get(url)
#             if response.status_code != 200:
#                 print(f"Skipping {url}: {response.status_code} - {response.reason}")
#                 failed_urls.append(url)
#                 return
            
#             html = response.text
#             soup = BeautifulSoup(html, 'html.parser')

#             # Skip homepage and main news page
#             if url != base_url and url != news_url:
#                 title = soup.title.string if soup.title else "No Title"
#                 published_date = extract_published_date(html)

#                 if valid_url_pattern.match(url) and published_date:
#                     try:
#                         parsed_date = datetime.strptime(published_date, "%Y-%m-%d")
#                         data.append({'title': title, 'url': url, 'date': parsed_date})
#                     except ValueError:
#                         print(f"Invalid date format for {url}: {published_date}")

#             # Extract and crawl links
#             for link in soup.find_all('a', href=True):
#                 href = link['href']
#                 if href.startswith('/'):
#                     href = f"{base_url}{href}"
#                 if valid_url_pattern.match(href) and href not in visited_urls:
#                     crawl(href)
#         except requests.exceptions.RequestException as e:
#             print(f"Failed to crawl {url}: {e}")
#             failed_urls.append(url)

#     crawl(news_url)

#     # Sort data by date (latest first) and take the top 10
#     sorted_data = sorted(data, key=lambda x: x['date'], reverse=True)[:10]
#     print(sorted_data)

#     # Generate HTML
#     with open("index.html", "w") as file:
#         file.write("""
#                 <html>
#                 <head>
#                     <title>News</title>
#                     <meta name="viewport" content="width=device-width, initial-scale=1">
#                     <style>
#                         body {
#                             font-family: Arial, sans-serif;
#                             padding: 20px;
#                             box-sizing: border-box;
#                         }
#                         a {
#                             font-size: 24px;
#                             color: black;
#                             font-weight: normal;
#                             text-decoration: underline;
#                             display: block;
#                             margin-bottom: 8px;
#                         }
#                         .date {
#                             font-size: 16px;
#                             color: rgb(59, 59, 59);
#                             display: block;
#                             margin-bottom: 20px;
#                         }
#                         @media (max-width: 600px) {
#                             a {
#                                 font-size: 18px;
#                             }
#                             .date {
#                                 font-size: 14px;
#                             }
#                         }
#                     </style>
#                 </head>
#                 <body>
#                 """)
#         # file.write("<h1>Latest News</h1>")
        
#         for item in sorted_data:
#             formatted_date = item['date'].strftime("%Y-%m-%d")
#             file.write(f"<a target='_blank' href='{item['url']}'>{item['title']}</a>")
#             file.write(f"<span class='date'>Published date: {formatted_date}</span>")
    
#         file.write("</body></html>")

#     # Save failed URLs to a log file
#     if failed_urls:
#         with open("failed_urls.log", "w") as log_file:
#             log_file.write("\n".join(failed_urls))
#         print(f"Failed URLs logged in 'failed_urls.log'.")

#     print("HTML file updated.")

# # Push changes to GitHub
# def push_to_github():
#     try:
#         repo_path = os.getcwd()
#         os.chdir(repo_path)

#         subprocess.run(["git", "add", "index.html"], check=True)
#         subprocess.run(["git", "commit", "-m", "Auto-update index.html"], check=True)
#         subprocess.run(["git", "push"], check=True)

#         print("Changes pushed to GitHub successfully!")
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to push changes: {e}")

# # Run the script
# if __name__ == "__main__":
#     crawl_and_generate_html()
#     push_to_github()
import os
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import subprocess

# Crawl website and generate HTML
def crawl_and_generate_html():
    base_url = "https://www.visive.ai"
    news_url = f"{base_url}/news"
    visited_urls = set()
    data = []
    failed_urls = []

    # Regex for valid URLs
    valid_url_pattern = re.compile(r"^https://www\.visive\.ai/news/[a-zA-Z0-9\-]+$")

    def extract_published_date(html):
        # Check for schema.org date published
        date_match = re.search(r'<meta\s+itemprop="datePublished"\s+content="([^"]*)"', html)
        if date_match:
            return date_match.group(1)

        # Check for <time> tag with pubdate attribute
        time_match = re.search(r'<time[^>]*pubdate[^>]*datetime="([^"]*)"[^>]*>', html)
        if time_match:
            return time_match.group(1)

        # Look for common date patterns
        date_patterns = [
            r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})",  # YYYY-MM-DD or YYYY/MM/DD
            r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})",  # DD/MM/YYYY
            r"(\w{3,9})\s+(\d{1,2}),\s+(\d{4})",  # Month DD, YYYY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, html)
            if match:
                if pattern == date_patterns[0]:
                    return f"{match.group(1)}-{int(match.group(2)):02}-{int(match.group(3)):02}"
                elif pattern == date_patterns[1]:
                    return f"{int(match.group(3)):04}-{int(match.group(2)):02}-{int(match.group(1)):02}"
                elif pattern == date_patterns[2]:
                    month = datetime.strptime(match.group(1), "%B").month
                    return f"{match.group(3)}-{month:02}-{int(match.group(2)):02}"
        return ""

    def crawl(url):
        if url in visited_urls:
            return
        visited_urls.add(url)

        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Skipping {url}: {response.status_code} - {response.reason}")
                failed_urls.append(url)
                return

            html = response.text
            soup = BeautifulSoup(html, 'html.parser')

            # Skip homepage and main news page
            if url != base_url and url != news_url:
                title = soup.title.string if soup.title else "No Title"
                published_date = extract_published_date(html)

                if valid_url_pattern.match(url) and published_date:
                    try:
                        parsed_date = datetime.strptime(published_date, "%Y-%m-%d")
                        data.append({'title': title, 'url': url, 'date': parsed_date})
                    except ValueError:
                        print(f"Invalid date format for {url}: {published_date}")

            # Extract and crawl links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    href = f"{base_url}{href}"
                if valid_url_pattern.match(href) and href not in visited_urls:
                    crawl(href)
        except requests.exceptions.RequestException as e:
            print(f"Failed to crawl {url}: {e}")
            failed_urls.append(url)

    crawl(news_url)

    # Sort data by date (latest first) and take the top 10
    sorted_data = sorted(data, key=lambda x: x['date'], reverse=True)[:8]

    # Generate HTML
    with open("index.html", "w", encoding="utf-8") as file:
        file.write("""
                <html>
                <head>
                    <title>News</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            padding: 20px;
                            box-sizing: border-box;
                        }
                        a {
                            font-size: 24px;
                            color: black;
                            font-weight: normal;
                            text-decoration: underline;
                            display: block;
                            margin-bottom: 8px;
                        }
                        .date {
                            font-size: 16px;
                            color: rgb(59, 59, 59);
                            display: block;
                            margin-bottom: 20px;
                        }
                        @media (max-width: 600px) {
                            a {
                                font-size: 18px;
                            }
                            .date {
                                font-size: 14px;
                            }
                        }
                    </style>
                </head>
                <body>
                """)

        for item in sorted_data:
            formatted_date = item['date'].strftime("%Y-%m-%d")
            file.write(f"<a target='_blank' href='{item['url']}'>{item['title']}</a>")
            file.write(f"<span class='date'>Published date: {formatted_date}</span>")

        file.write("</body></html>")

    # Save failed URLs to a log file
    if failed_urls:
        with open("failed_urls.log", "w") as log_file:
            log_file.write("\n".join(failed_urls))
        print(f"Failed URLs logged in 'failed_urls.log'.")

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

# Run the script
if __name__ == "__main__":
    crawl_and_generate_html()
    push_to_github()
