# local_ai_news_pipeline.py

import os
import base64
import feedparser
import requests
import re
import time
import json
from datetime import datetime, timezone, date
import uuid
import io
from dotenv import load_dotenv
from github import Github, InputGitTreeElement
import random
from xml.sax.saxutils import escape as xml_escape

# --- Google GenAI & Drive Imports ---
# pip install google-genai google-api-python-client google-auth-httplib2 google-auth-oauthlib
from google import genai
from google.genai import types as genai_types
from google.api_core import exceptions as api_core_exceptions
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseUpload


# --------------------------------------------------
# 0. CONFIGURATION & INITIALIZATION
# --------------------------------------------------
load_dotenv()

# --- OTHER SERVICES & PATHS ---
READABILITY_SERVICE_URL = os.getenv("READABILITY_SERVICE_URL", "https://readability-199784609767.us-central1.run.app/")

# --- Path to the script's own directory, for reading data/state files ---
SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_BASE_DIR, "data")
ARTICLES_DB_FILE = os.path.join(DATA_DIR, "ai_articles_database.json")
PIPELINE_STATE_FILE = os.path.join(DATA_DIR, "pipeline_state.json")

# --- Path for WRITING output, configurable via an environment variable ---
# Defaults to a local './output_html' folder for easy local testing.
# The GitHub Action will override this to point to the Repo B checkout.
OUTPUT_HTML_DIR = os.getenv("OUTPUT_HTML_DIR", os.path.join(SCRIPT_BASE_DIR, "output_html"))

# This is an output file, so it correctly uses the OUTPUT_HTML_DIR
NEWS_SITEMAP_FILENAME = "news_sitemap.xml"

# --- CORE API KEYS ---
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
GITHUB_TOKEN = os.getenv("GITS_TOKEN")

# --- IMAGE GENERATION CONFIG (Gemini + Google Drive) ---
GEMINI_API_KEYS_STR = os.getenv("GEMINI_API_KEYS", "")
GEMINI_API_KEYS = [key.strip() for key in GEMINI_API_KEYS_STR.split(',') if key.strip()]
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
SERVICE_ACCOUNT_FILE_PATH = os.getenv("SERVICE_ACCOUNT_FILE_PATH")





# --- DATA SOURCES & CONSTANTS ---
AI_RSS_FEEDS = [
    {"name": "Artificial Intelligence Google Alert", "url": "https://www.google.co.in/alerts/feeds/08615545546582653290/14485627415139611533"},
    {"name": "AI robotics Google Alert", "url": "https://www.google.co.in/alerts/feeds/17781850005949356879/4482254516846037427"},
    # {"name": "Computer vision Google Alert", "url": "https://www.google.co.in/alerts/feeds/17781850005949356879/11410094533971671766"},
    # {"name": "AI VISION Google Alert", "url": "https://www.google.co.in/alerts/feeds/17781850005949356879/16069242934067765173"}
]
DEFAULT_ARTICLE_IMAGE_URL = "https://lh3.googleusercontent.com/d/1_qDsCmt7qY75xWabnoPD-I5K_pNXsm3m=w1920?authuser=0"
HYPERLINK_BASE_URL_FOR_TAGS = 'https://www.google.com/search?q='

# --- MODEL CONFIGURATION ---
DEEPINFRA_REWRITE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
GEMINI_IMAGE_GEN_MODEL = "gemini-2.0-flash-preview-image-generation" # Changed from preview model

# --- PROMPT TEMPLATES ---
LLM_REWRITE_AND_PROMPT_GENERATION_TEMPLATE = """You are a veteran News Editor, SEO Strategist, Content Humanizer, and Visual Prompt Engineer with 30+ years of experience. Your role is to process the provided news content and:

1. **Rewrite** it into a 1000-word human-sounding article.
2. Apply **advanced SEO optimization techniques**.
3. Generate a highly relevant **AI image prompt**.
4. Return everything in a strictly structured **JSON object** for publishing systems.

---

### STEP 1: ARTICLE REWRITING (1000 words)

Rewrite the article in a professional, engaging, human-like tone using the following **SEO-enhanced content strategy**:

####  SEO TECHNIQUES TO APPLY:
- Ensure **primary keyword density**: Use the main topic keyword naturally in title, excerpt, H2s, intro, conclusion, and ~3–5 times in the body.
- Use **LSI keywords and synonyms** throughout the text for semantic relevance.
- Add **internal linking placeholders** in brackets like `[Related: Future of AI in Healthcare]`.
- Insert **structured subheadings** (`##`, `###`) that include relevant keywords.
- Include **featured snippet-style short paragraphs** answering key questions.
- Begin with a **strong hook** and end with a **clear takeaway**.
- Use **short sentences**, **active voice**, and **natural transitions**.
- Avoid keyword stuffing, unnatural repetition, or AI-typical filler phrases.

---

### STEP 2: METADATA GENERATION

#### Generate the following:
- **title**: Engaging, SEO-friendly headline (under 60 characters preferred)
- **excerpts**: 25–35 word meta description with primary keywords
- **tags**: 5 comma-separated SEO-friendly tags
- **faqs**: 5 unique FAQs based on common user questions

---

### STEP 3: IMAGE PROMPT GENERATION

You are an expert visual prompt engineer with over 30 years of experience in creating highly relevant and vivid image prompts based on written content.

Generate an image prompt that:
- Is under 60 words
- Visually represents the main topic and mood of the **rewritten** article
- Includes scene, objects, people, mood, lighting, and camera angle
- Avoids brand names, logos, or real people
-landscape image 16:9 ratio
-if there is any  context of big orgaiztion,or famous person then add any symbol of those . andimages should be ralistic. 
-don use flashy words in the prompt like futuristic,ai robot,glowing  which resulting in the same kind of images only . i wanted realsistic images only 
---

### INPUT FIELDS:
Original Title: `{title}`
Original Excerpt/RSS Content: `{excerpts}`
Full Article Text (e.g., from Readability): `{main_content}`

---

### STRICT OUTPUT FORMAT EXAMPLE (JSON):

{{
  "title": "SEO-optimized rewritten title",
  "excerpts": "Engaging meta description with keyword focus.",
  "content": "Fully rewritten, SEO-optimized article using markdown headings (##, ###) and \\n\\n paragraph breaks. Includes internal linking suggestions like [Related: Future of AI in Healthcare].",
  "tags": "ai, artificial intelligence, machine learning, future technology, automation",
  "faqs": [
    {{
      "question": "How is AI transforming business operations?",
      "answer": "AI enables smarter automation, better forecasting, and real-time decision-making across industries like logistics, finance, and customer service."
    }},
    {{
      "question": "Is AI a threat to job security?",
      "answer": "AI may replace some repetitive jobs, but it also creates new roles in AI oversight, data management, and ethical governance."
    }},
    {{
      "question": "Can AI be trusted for decision-making?",
      "answer": "With proper transparency, training data controls, and ethics frameworks, AI can be used responsibly in decision-making processes."
    }},
    {{
      "question": "What industries benefit most from AI?",
      "answer": "Healthcare, transportation, finance, marketing, and manufacturing are major sectors rapidly adopting AI for efficiency and innovation."
    }},
    {{
      "question": "How can small businesses use AI?",
      "answer": "Small businesses can use AI tools for marketing automation, customer service chatbots, analytics, and process optimization."
    }}
  ],
  "image_prompt": "A glowing humanoid robot standing in front of a digital world map with binary code flowing around, high-tech background, cinematic lighting, viewed from a low angle in a futuristic setting."
}}
RULES:
Output must be a valid JSON object with double quotes and \\n\\n for newlines.

Do not include source URLs, author names, or citations.

Do not add AI disclaimers, code blocks, or model metadata.

Everything must reflect the rewritten version, not raw input.

DO NOT USE HEDINGS LIKE INTRODUCTION ,CONLCUTION ,SUMMARY , [Related: ],expert insights, Background etc.

do not use comman ai phrase Start fron "The" foe example THE rise of ,The evolutoin The revolution, Imagine etc.

dont add any additional ** in the response.
"""

# --- GITHUB CONFIGURATION ---
REPO_OWNER = "bioenable"
REPO_NAME = "visive"
BRANCH_NAME = "main"
COMMIT_MESSAGE_BASE = "Automated AI News Update (Geosquare Style)"
HTML_SUBFOLDER_IN_REPO = ""

# --- RUN LIMITS ---
MAX_READABILITY_CALLS_PER_RUN = 100
MAX_LLM_REWRITE_CALLS_PER_RUN = 100
MAX_IMAGE_GENERATION_CALLS_PER_RUN = 100

# --- DIRECTORY CREATION ---
for dir_path in [DATA_DIR, OUTPUT_HTML_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

# --- IMAGE GENERATION STATE ---
key_rpm_cooldown_until = {}
key_daily_usage = {}
current_key_selection_index = 0

# --------------------------------------------------
# 1. UTILITY & STATE MANAGEMENT FUNCTIONS
# --------------------------------------------------
# (This section remains unchanged from your provided code, as it's already robust)
def load_pipeline_state():
    if os.path.exists(PIPELINE_STATE_FILE):
        try:
            with open(PIPELINE_STATE_FILE, 'r') as f: return json.load(f)
        except json.JSONDecodeError: pass
    return {"current_step": "FETCH_RSS"}

def save_pipeline_state(state):
    with open(PIPELINE_STATE_FILE, 'w') as f: json.dump(state, f, indent=2)
    print(f"  Pipeline state saved. Next step: {state.get('current_step')}")

def load_articles_db():
    if os.path.exists(ARTICLES_DB_FILE):
        try:
            with open(ARTICLES_DB_FILE, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e:
            print(f"Warning: Error loading {ARTICLES_DB_FILE}: {e}. Starting empty."); return {}
    return {}

def save_articles_db(data):
    try:
        with open(ARTICLES_DB_FILE, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e: print(f"Error saving {ARTICLES_DB_FILE}: {e}")

def clean_llm_json_string(json_string):
    if not json_string or not isinstance(json_string, str): return None
    cleaned_string = json_string.strip()
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", cleaned_string, re.DOTALL)
    if match: cleaned_string = match.group(1).strip()
    else:
        json_start_brace = cleaned_string.find('{'); json_start_bracket = cleaned_string.find('[')
        start_index = -1
        if json_start_brace != -1 and json_start_bracket != -1: start_index = min(json_start_brace, json_start_bracket)
        elif json_start_brace != -1: start_index = json_start_brace
        elif json_start_bracket != -1: start_index = json_start_bracket
        if start_index != -1:
            temp_string = cleaned_string[start_index:]
            open_braces, open_brackets, last_char_index = 0, 0, -1
            for i, char in enumerate(temp_string):
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                elif char == '[': open_brackets += 1
                elif char == ']': open_brackets -= 1
                if open_braces == 0 and open_brackets == 0 and i > 0:
                    if (not temp_string[i + 1:].strip() or temp_string[i + 1:].strip().startswith("```")) and (start_index == 0 or (start_index > 0 and cleaned_string[start_index - 1] not in ['"', "'"])):
                        last_char_index = i; break
            if last_char_index != -1: cleaned_string = temp_string[:last_char_index + 1]
            else: cleaned_string = temp_string
    try: return json.loads(cleaned_string)
    except json.JSONDecodeError as e_direct:
        print(f"  LLM Clean: Initial parse failed: {e_direct}. Attempting repair...")
        def replace_newlines_in_strings(match_obj): return match_obj.group(0).replace('\n', '\\n')
        try:
            repaired_string = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', replace_newlines_in_strings, cleaned_string)
            repaired_string = re.sub(r',\s*([\}\]])', r'\1', repaired_string)
            return json.loads(repaired_string)
        except json.JSONDecodeError as e_rep:
            print(f"  LLM Clean: Failed to parse LLM JSON after repair: {e_rep}."); return None

def extract_actual_url_from_google_alert(redirect_url):
    if not redirect_url or not isinstance(redirect_url, str): return ""
    try:
        match = re.search(r"[?&]url=([^&]+)", redirect_url)
        return requests.utils.unquote(match.group(1)) if match else redirect_url
    except Exception: return redirect_url

def sanitize_filename(title_str):
    if not title_str: return f"article-{int(time.time())}.html"
    fn = title_str.lower()
    fn = re.sub(r'[^\w\s-]', '', fn)
    fn = re.sub(r'[-\s]+', '-', fn).strip('-_')
    fn = fn[:100]
    return f"{fn}.html" if fn else f"article-{int(time.time())}.html"

# --------------------------------------------------
# 2. RSS FETCHER
# --------------------------------------------------
# (This function remains unchanged)
def fetch_new_rss_articles(articles_db):
    print("\n>>> Starting RSS Fetching (Deduplication by URL) <<<")
    existing_actual_urls = {data.get('link_actual') for data in articles_db.values() if data.get('link_actual')}
    print(f"  Loaded {len(existing_actual_urls)} existing URLs from DB for deduplication.")
    newly_added_count = 0
    for feed_config in AI_RSS_FEEDS:
        print(f"  Fetching: {feed_config['name']}...")
        parsed_feed = feedparser.parse(feed_config['url'])
        if parsed_feed.bozo: print(f"    Warning: feedparser issues with {feed_config['name']}. Exception: {parsed_feed.bozo_exception}")
        for entry in parsed_feed.entries:
            actual_url = extract_actual_url_from_google_alert(getattr(entry, 'link', ""))
            if not actual_url or actual_url in existing_actual_urls: continue
            existing_actual_urls.add(actual_url)
            article_id = getattr(entry, 'id', str(uuid.uuid4()))
            title = getattr(entry, 'title', "No Title")
            published_struct = getattr(entry, 'published_parsed', getattr(entry, 'updated_parsed', None))
            published_iso = datetime.fromtimestamp(time.mktime(published_struct), tz=timezone.utc).isoformat() if published_struct else datetime.now(timezone.utc).isoformat()
            summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
            content_detail = ""
            if hasattr(entry, 'content') and entry.content:
                if isinstance(entry.content, list) and entry.content:
                    html_obj = next((c for c in entry.content if hasattr(c, 'type') and c.type == 'text/html'), None)
                    content_detail = html_obj.value if html_obj else entry.content[0].value
                elif isinstance(entry.content, str): content_detail = entry.content
            articles_db[article_id] = {'id': article_id, 'title_original': title, 'link_actual': actual_url,'published_date_iso': published_iso, 'content_rss': summary or content_detail, 'feed_name': feed_config['name'], 'source_url_feed': feed_config['url'], 'status': 'new', 'last_processed_timestamp': datetime.now(timezone.utc).isoformat(),'readability_raw_json': None, 'llm_raw_json': None, 'processed_llm_data': None, 'final_image_url': None, 'readability_try_count': 0, 'html_filename': None}
            newly_added_count += 1
            print(f"    + Added (New URL): {title[:60]}...")
        time.sleep(1)
    print(f"RSS Fetching: Added {newly_added_count} new, unique articles.")
    save_articles_db(articles_db)
    return articles_db

# --------------------------------------------------
# 3. READABILITY PROCESSOR
# --------------------------------------------------
# (This section remains unchanged)
def process_readability_for_articles(articles_db):
    print("\n>>> Starting Readability Processing <<<")
    processed_count = 0
    for article_id, article_data in articles_db.items():
        if article_data.get('status') == 'new' and article_data.get('link_actual'):
            if processed_count >= MAX_READABILITY_CALLS_PER_RUN: print(f"  Readability: Limit ({MAX_READABILITY_CALLS_PER_RUN}) reached."); break
            print(f"  Readability: Processing '{article_data['title_original'][:50]}...'")
            readability_json_str = get_content_with_readability(article_data['link_actual'])
            article_data['readability_raw_json'] = readability_json_str
            article_data['readability_try_count'] = article_data.get('readability_try_count', 0) + 1
            if readability_json_str and '"error":' not in readability_json_str.lower(): article_data['status'] = 'readability_fetched'
            else: article_data['status'] = 'readability_failed'; print(f"    Readability failed for {article_data['title_original'][:50]}")
            article_data['last_processed_timestamp'] = datetime.now(timezone.utc).isoformat()
            processed_count += 1; time.sleep(1)
    save_articles_db(articles_db)
    print(f"Readability Processing: Attempted on {processed_count} articles.")
    return articles_db

def get_content_with_readability(url_to_fetch):
    if not url_to_fetch or not READABILITY_SERVICE_URL: return None
    try:
        api_url = f"{READABILITY_SERVICE_URL}?url={requests.utils.quote(url_to_fetch)}"; response = requests.get(api_url, timeout=60)
        response.raise_for_status(); return response.text
    except Exception as e:
        print(f"    Readability error for {url_to_fetch[:70]}: {e}"); return json.dumps({"error": str(e)})

# --------------------------------------------------
# 4. LLM CONTENT PROCESSOR (MODIFIED)
# --------------------------------------------------
def _call_deepinfra_api(prompt_text, max_tokens=8000, temperature=0.7):
    """Generic function to call the DeepInfra Chat Completions API."""
    if not DEEPINFRA_API_KEY: print("ERROR: DEEPINFRA_API_KEY not set."); return None
    api_url = "https://api.deepinfra.com/v1/openai/chat/completions"
    payload = {"model": DEEPINFRA_REWRITE_MODEL, "messages": [{"role": "user", "content": prompt_text}], "max_tokens": max_tokens, "temperature": temperature}
    headers = {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"}
    response = None
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        if result.get("choices") and result["choices"][0].get("message"): return result["choices"][0]["message"]["content"]
        else: print(f"    LLM: DeepInfra response missing structure: {str(result)[:200]}..."); return None
    except requests.exceptions.RequestException as e:
        err_msg = f"    LLM: Error calling DeepInfra: {e}"
        print(err_msg + (f" | Response: {response.text[:200]}" if response else "")); return None
    except Exception as e: print(f"    LLM: Unexpected error in _call_deepinfra_api: {e}"); return None

def rewrite_articles_with_llm(articles_db):
    """Rewrites article content and generates an image prompt using an LLM."""
    print("\n>>> Starting LLM Article Rewriting & Image Prompt Generation <<<")
    processed_count = 0
    # Create a list of item IDs to iterate over to prevent issues with changing dict size during loop
    articles_to_process = list(articles_db.items())

    for article_id, article_data in articles_to_process:
        # THE FIX IS HERE: We ONLY look for articles in the 'readability_fetched' state.
        # If the status is 'new', 'llm_rewritten', 'llm_failed_xyz', or anything else, this code block is skipped.
        if article_data.get('status') == 'readability_fetched':
            # This is the only part that needs to be inside the status check.
            if processed_count >= MAX_LLM_REWRITE_CALLS_PER_RUN:
                print(f"  LLM Rewrite: Limit ({MAX_LLM_REWRITE_CALLS_PER_RUN}) reached.")
                break # Stop processing for this run

            try:
                # Defensive Check: If for some reason it has this status but already has LLM data, skip it.
                if article_data.get('llm_raw_json'):
                    print(f"  LLM Rewrite: Skipping '{article_data['title_original'][:50]}...' as it already has LLM data despite status.")
                    # Optionally fix the status
                    article_data['status'] = 'llm_rewritten'
                    continue

                readability_data = json.loads(article_data.get('readability_raw_json', '{}'))
                main_content = readability_data.get('textContent', readability_data.get('content', ''))
                if not main_content:
                    article_data['status'] = 'llm_failed_no_readability_content'
                    continue

                prompt = LLM_REWRITE_AND_PROMPT_GENERATION_TEMPLATE.format(
                    title=article_data.get('title_original', ''),
                    excerpts=article_data.get('content_rss', '')[:1000],
                    main_content=main_content[:12000]
                )
                print(f"  LLM Rewrite: Processing '{article_data['title_original'][:50]}...'")
                llm_json_str = _call_deepinfra_api(prompt)

                if llm_json_str:
                    article_data['llm_raw_json'] = llm_json_str
                    article_data['status'] = 'llm_rewritten' # Status is changed, so it won't be picked up again
                else:
                    article_data['status'] = 'llm_failed_api_error' # Status is changed

                article_data['last_processed_timestamp'] = datetime.now(timezone.utc).isoformat()
                processed_count += 1
                time.sleep(2)

            except json.JSONDecodeError:
                article_data['status'] = 'llm_failed_readability_parse_error'
            except Exception as e:
                article_data['status'] = 'llm_failed_unknown_error'
                print(f"  LLM Rewrite: Unknown error: {e}")

    save_articles_db(articles_db)
    print(f"LLM Rewriting: Attempted on {processed_count} articles.")
    return articles_db
# --------------------------------------------------
# 5. IMAGE GENERATOR (Gemini + Google Drive) - NEW SECTION
# --------------------------------------------------

def _is_rpd_limit_error(exception):
    return "daily quota" in str(exception).lower() or "per day" in str(exception).lower()

def _get_available_gemini_client():
    """Gets an available Gemini API key and returns the key index and a configured client object."""
    global current_key_selection_index
    if not GEMINI_API_KEYS:
        print("  ERROR: No Gemini API keys found in .env.")
        return None, None

    today_str = date.today().isoformat()
    num_keys = len(GEMINI_API_KEYS)
    for i in range(num_keys):
        key_idx = (current_key_selection_index + i) % num_keys
        if key_idx in key_rpm_cooldown_until and time.time() < key_rpm_cooldown_until[key_idx]:
            continue
        if key_idx not in key_daily_usage or key_daily_usage[key_idx]['date_str'] != today_str:
            key_daily_usage[key_idx] = {'date_str': today_str, 'count': 0}
        if key_daily_usage[key_idx]['count'] >= 100: # Hardcoded RPD limit
            continue

        current_key_selection_index = (key_idx + 1) % num_keys
        selected_api_key = GEMINI_API_KEYS[key_idx]
        try:
            # CORRECTED: Instantiate a Client object, do not use genai.configure
            client = genai.Client(api_key=selected_api_key)
            print(f"    ImageGen: Selected Gemini Key index {key_idx} (ends ...{selected_api_key[-4:]}).")
            return key_idx, client
        except Exception as e:
            print(f"    ImageGen: Failed to create client with key index {key_idx}: {e}")
            key_rpm_cooldown_until[key_idx] = time.time() + 65
            continue

    print("    ImageGen: No API key is currently available.")
    return None, None

def _authenticate_drive():
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE_PATH, scopes=['https://www.googleapis.com/auth/drive'])
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"    ImageGen: Error authenticating with Google Drive: {e}")
        return None

def _upload_to_drive_and_get_url(drive_service, image_data, image_filename):
    if not image_data:
        return None
    try:
        file_metadata = {"name": image_filename, "parents": [DRIVE_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(image_data), mimetype='image/png', resumable=True)
        print(f"    ImageGen: Uploading '{image_filename}' to Google Drive...")
        file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        file_id = file.get("id")
        drive_service.permissions().create(fileId=file_id, body={'type': 'anyone', 'role': 'reader'}).execute()
        # Using the w1200 parameter for a reasonably sized image
        return f"https://lh3.googleusercontent.com/d/{file_id}=w1200"
    except Exception as e:
        print(f"    ImageGen: Error during file upload: {e}")
        return None

def _generate_and_upload_image(prompt: str, base_filename: str, genai_client, key_idx):
    """Generates an image using a provided Gemini client, uploads it, and returns the URL."""
    try:
        print(f"    ImageGen: Sending request to Gemini for '{base_filename[:40]}...'")

        # --- THIS IS THE CORRECTED PART ---
        # It now perfectly matches the logic from your original working script.
        generate_content_config = genai_types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
        contents = [genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=prompt)])]

        response_stream = genai_client.models.generate_content_stream(
            model=GEMINI_IMAGE_GEN_MODEL,
            contents=contents,
            config=generate_content_config  # CORRECTED: Use 'config' instead of 'generation_config'
        )
        # --- END OF CORRECTION ---

        image_data = None
        for chunk in response_stream:
             if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    # Check for the correct attribute for image data in the response
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        break
                if image_data:
                    break

        if not image_data:
            print("    ImageGen: API call succeeded but no image data was returned.")
            return None

        key_daily_usage[key_idx]['count'] += 1
        drive_service = _authenticate_drive()
        if not drive_service:
            return None

        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', base_filename) + ".png"
        return _upload_to_drive_and_get_url(drive_service, image_data, safe_filename)

    except api_core_exceptions.ResourceExhausted as e:
        print(f"    ImageGen: RESOURCE EXHAUSTED for key index {key_idx}. Error: {e}")
        if _is_rpd_limit_error(e):
            key_daily_usage[key_idx]['count'] = 100
        else:
            key_rpm_cooldown_until[key_idx] = time.time() + 65
        return None
    except Exception as e:
        print(f"    ImageGen: Unexpected error during generation: {e}")
        return None
    
def generate_images_for_articles(articles_db):
    """Generates images for articles that have a prompt but no final image URL."""
    print("\n>>> Starting On-Demand Image Generation <<<")
    processed_count = 0

    # Get a client once for this run, if available.
    key_idx, genai_client = _get_available_gemini_client()
    if not genai_client:
        print("  Image Generation: No available Gemini client. Skipping this step for now.")
        return articles_db

    # Create a list of item IDs to iterate over to prevent issues with changing dict size during loop
    articles_to_process = list(articles_db.items())

    for article_id, article_data in articles_to_process:
        # THE FIX IS HERE: We ONLY look for articles in the 'llm_processed_data_extracted' state
        # AND that don't already have an image URL. This is the specific "work queue" for this function.
        if article_data.get('status') == 'llm_processed_data_extracted' and not article_data.get('final_image_url'):
            # This is the only part that needs to be inside the status check.
            if processed_count >= MAX_IMAGE_GENERATION_CALLS_PER_RUN:
                print(f"  Image Generation: Limit ({MAX_IMAGE_GENERATION_CALLS_PER_RUN}) reached.")
                break # Stop processing for this run

            # Defensive Check: If status is correct but no LLM data exists, skip and fix status.
            pd = article_data.get('processed_llm_data')
            if not pd:
                article_data['status'] = 'image_gen_failed_no_llm_data'
                continue

            image_prompt = pd.get('image_prompt')
            base_filename = sanitize_filename(pd.get('final_title', 'untitled-article')).replace('.htm', '') # Corrected .html to .htm

            if not image_prompt or not base_filename:
                article_data['final_image_url'] = DEFAULT_ARTICLE_IMAGE_URL
                article_data['status'] = 'image_generation_skipped' # Status changed
                continue

            print(f"  Image Generation: Processing '{base_filename[:50]}...'")
            generated_url = _generate_and_upload_image(image_prompt, base_filename, genai_client, key_idx)

            if generated_url:
                article_data['final_image_url'] = generated_url
                article_data['status'] = 'image_generated' # Status changed
                print(f"    Success! Image URL: {generated_url}")
            else:
                article_data['final_image_url'] = DEFAULT_ARTICLE_IMAGE_URL
                article_data['status'] = 'image_generation_failed' # Status changed
                print("    Failed. Using default placeholder image.")
                
                # If a key fails (e.g., rate limit), try to get a new one for the next article
                key_idx, genai_client = _get_available_gemini_client()
                if not genai_client:
                    print("  No more clients available, ending image generation for this run.")
                    break # Exit the loop completely if all keys are exhausted

            article_data['last_processed_timestamp'] = datetime.now(timezone.utc).isoformat()
            processed_count += 1
            time.sleep(13) # Rate limit delay

    save_articles_db(articles_db)
    print(f"Image Generation: Attempted on {processed_count} articles.")
    return articles_db

# --------------------------------------------------
# 6. PROCESS LLM OUTPUT & PREPARE PUBLISHABLE DATA (MODIFIED)
# --------------------------------------------------

def _get_common_navbar_html():
    """Generates the common HTML for the site's responsive navbar."""
    return f"""
    <header class="fixed w-full top-0 z-50 bg-background/80 backdrop-filter backdrop-blur-lg border-b border-border">
      <div class="flex h-16 items-center px-4 md:px-8 lg:px-12 max-w-screen-xl mx-auto">
         <a href="https://news.visive.ai/" class="text-2xl font-bold text-primary mr-auto">VISIVE.AI</a>
        <nav class="hidden md:flex items-center space-x-6 text-base">
          <a href="https://news.visive.ai/" class="text-foreground font-medium transition-colors hover:text-primary">Home</a>
          <a href="https://news.visive.ai/news.html" class="text-foreground font-medium transition-colors hover:text-primary">News</a>
          <a href="https://www.visive.ai/solutions" class="text-foreground font-medium transition-colors hover:text-primary">Solutions</a>
          <a href="{'https://www.visive.ai/how-we-help/introductions'}" class="block py-2 text-base font-medium transition-colors hover:text-primary">How we Help</a>
          <a href="{'https://www.visive.ai/demos'}" class="block py-2 text-base font-medium transition-colors hover:text-primary">Demos</a>
          <a href="https://www.visive.ai/pricing" class="text-foreground font-medium transition-colors hover:text-primary">Pricing</a>
          <a href="https://www.visive.ai/contact-us" class="text-foreground font-medium transition-colors hover:text-primary">Contact Us</a>
        </nav>
        <button class="md:hidden ml-auto" id="mobile-menu-toggle" aria-label="Open menu">
          <svg class="h-8 w-8 text-foreground" stroke="currentColor" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg>
        </button>
      </div>
      <div id="mobile-menu" class="hidden md:hidden px-4 pb-4 bg-background/80 backdrop-filter backdrop-blur-lg">
        <a href="https://news.visive.ai/" class="block py-2 text-base font-medium transition-colors hover:text-primary">Home</a>.
        <a href="https://news.visive.ai/news.html" class="text-foreground font-medium transition-colors hover:text-primary">News</a>
        <a href="https://www.visive.ai/solutions" class="block py-2 text-base font-medium transition-colors hover:text-primary">Solutions</a>
        <a href="{'https://www.visive.ai/how-we-help/introductions'}" class="block py-2 text-base font-medium transition-colors hover:text-primary">How we Help</a>
        <a href="{'https://www.visive.ai/demos'}" class="block py-2 text-base font-medium transition-colors hover:text-primary">Demos</a>
        <a href="https://www.visive.ai/pricing" class="block py-2 text-base font-medium transition-colors hover:text-primary">Pricing</a>
        <a href="https://www.visive.ai/contact-us" class="block py-2 text-base font-medium transition-colors hover:text-primary">Contact Us</a>
      </div>
      <script>
        const btn = document.getElementById('mobile-menu-toggle');
        const menu = document.getElementById('mobile-menu');
        if (btn && menu) {{
          btn.addEventListener('click', () => {{ menu.classList.toggle('hidden'); }});
        }}
      </script>
    </header>
    """




def process_llm_outputs_and_prepare_publish_data(articles_db):
    """Parses raw LLM JSON output, cleans it, and structures it for the next steps."""
    print("\n>>> Starting LLM Output Processing & Publish Prep <<<")
    updated_count = 0
    for article_id, article_data in articles_db.items():
        if article_data.get('status') == 'llm_rewritten' and article_data.get('llm_raw_json'):
            print(f"  Processing LLM output for: {article_data['title_original'][:50]}...")
            llm_parsed = clean_llm_json_string(article_data['llm_raw_json'])
            if not llm_parsed:
                article_data['status'] = 'llm_parse_failed'; continue
            
            tags_llm = llm_parsed.get("tags", "")
            tags_list = [t.strip() for t in tags_llm.split(',') if t.strip()] if isinstance(tags_llm, str) else (tags_llm if isinstance(tags_llm, list) else [])
            hyperlinked_tags_md = ", ".join([f"[{t}]({HYPERLINK_BASE_URL_FOR_TAGS}{requests.utils.quote(t)})" for t in tags_list])
            faqs_llm = llm_parsed.get("faqs", [])
            faqs_plain = ""
            if isinstance(faqs_llm, list):
                faq_pairs = [f"Q: {f.get('question','').strip()}\nA: {f.get('answer','').strip()}" for f in faqs_llm if f.get('question') and f.get('answer')]
                faqs_plain = "\n\n".join(faq_pairs)
            elif isinstance(faqs_llm, str): faqs_plain = faqs_llm.strip()
                
            article_data['processed_llm_data'] = {
                "final_title": llm_parsed.get("title", article_data['title_original']).strip(),
                "final_excerpts": llm_parsed.get("excerpts", "").strip(),
                "raw_content_for_html": llm_parsed.get("Content", llm_parsed.get("content", "")).strip(),
                "raw_faqs_for_html": faqs_plain,
                "markdown_tags_for_html": hyperlinked_tags_md,
                "image_prompt": llm_parsed.get("image_prompt", "").strip(), # Extract the image prompt
                "category_display": "Artificial Intelligence",
                "source_feed_name": article_data.get('feed_name'),
                "published_date_iso": article_data.get('published_date_iso'),
            }
            article_data['status'] = 'llm_processed_data_extracted'
            article_data['last_processed_timestamp'] = datetime.now(timezone.utc).isoformat()
            updated_count += 1
            
    save_articles_db(articles_db)
    print(f"LLM Output Processing: Updated {updated_count} articles.")
    return articles_db

# --------------------------------------------------
# 7. HTML & SITEMAP GENERATOR
# --------------------------------------------------
# (The HTML function itself remains unchanged, as it's perfectly fine)
# This is the full, corrected function as requested.
def create_single_news_blog_page_html_local(data_for_template, related_articles_list):
    """
    Generates the HTML for a single blog page with corrections for related articles,
    image size, and added support for lists in the content.
    """
    title = data_for_template.get('final_title', "AI News Article")
    publish_date_iso_or_str = data_for_template.get('published_date_iso', "")
    excerpts = data_for_template.get('final_excerpts', "")
    main_content_raw = data_for_template.get('raw_content_for_html', "")
    faqs_raw = data_for_template.get('raw_faqs_for_html', "")
    tags_markdown_str = data_for_template.get('markdown_tags_for_html', "")
    article_category_display = data_for_template.get('category_display', "Artificial Intelligence")
    article_image_url_to_use = data_for_template.get('article_image_url_to_use', DEFAULT_ARTICLE_IMAGE_URL)

    source_site = "Visive.ai"

    display_date = "Date N/A"
    if publish_date_iso_or_str:
        try:
            dt_obj = datetime.fromisoformat(publish_date_iso_or_str.replace('Z', '+00:00'))
            display_date = dt_obj.strftime("%b %d, %Y")
        except ValueError:
            try:
                dt_obj = datetime.strptime(publish_date_iso_or_str.split(" ")[0], "%m/%d/%Y")
                display_date = dt_obj.strftime("%b %d, %Y")
            except ValueError:
                display_date = publish_date_iso_or_str

    # --- Content HTML parsing with list support --- (Unchanged)
    content_html = ""
    if main_content_raw:
        in_ul = False
        in_ol = False
        for line in main_content_raw.strip().split('\n'):
            line = line.strip()
            if not (line.startswith('* ') or line.startswith('- ') or re.match(r'\d+\.\s', line)):
                if in_ul:
                    content_html += "</ul>\n"
                    in_ul = False
                if in_ol:
                    content_html += "</ol>\n"
                    in_ol = False
            if not line:
                continue
            if line.startswith("### "):
                content_html += f"<h3>{line.lstrip('### ').strip()}</h3>\n"
            elif line.startswith("## "):
                content_html += f"<h2>{line.lstrip('## ').strip()}</h2>\n"
            elif line.startswith("# "):
                content_html += f"<h2>{line.lstrip('# ').strip()}</h2>\n"
            elif line.startswith('* ') or line.startswith('- '):
                if not in_ul:
                    content_html += "<ul>\n"
                    in_ul = True
                content_html += f"  <li>{line.lstrip('* - ').strip()}</li>\n"
            elif re.match(r'\d+\.\s', line):
                if not in_ol:
                    content_html += "<ol>\n"
                    in_ol = True
                content_html += f"  <li>{re.sub(r'^\d+\.\s*', '', line).strip()}</li>\n"
            else:
                content_html += f"<p>{line}</p>\n"
        if in_ul: content_html += "</ul>\n"
        if in_ol: content_html += "</ol>\n"

    # --- FAQ Section --- (Unchanged)
    faqs_html_list = ""
    if faqs_raw:
        faqs_html_list = """
        <section class="w-full py-8">
          <div class="mx-auto w-full">
            <h2 class="text-3xl font-bold mb-6 text-foreground">Frequently Asked Questions</h2>
            <div class="grid gap-4">
        """
        faq_items = faqs_raw.split('\n\n')
        for item_str in faq_items:
            if item_str.strip():
                question_part, answer_part_raw = "", ""
                q_match = re.match(r"Q:\s*(.*)", item_str, re.IGNORECASE)
                a_match = re.search(r"\nA:\s*(.*)", item_str, re.DOTALL | re.IGNORECASE)
                if q_match:
                    question_part = q_match.group(1).strip()
                    if a_match:
                        answer_part_raw = a_match.group(1).strip()
                    else:
                        answer_part_raw = item_str[q_match.end():].strip() if item_str[q_match.end():].strip() else "Details in article."
                else:
                    if a_match:
                        answer_part_raw = a_match.group(1).strip()
                        question_part = item_str[:a_match.start()].strip() if item_str[:a_match.start()].strip() else "Question"
                    else:
                        question_part = item_str.strip()
                        answer_part_raw = "Details provided within the article context."
                if question_part:
                    answer_part_html = answer_part_raw.replace('\\n', '<br/>').replace('\n', '<br/>')
                    faqs_html_list += f"<div class='rounded-lg border p-4 bg-muted'><h3 class='font-semibold text-lg'>{question_part}</h3><p class='text-muted-foreground text-base leading-relaxed'>{answer_part_html}</p></div>"
        faqs_html_list += "</div></div></section>"

    # --- Tags Section --- (Unchanged)
    article_specific_tags_html_section = ""
    if tags_markdown_str:
        article_specific_tags_html_section = """
        <section class="w-full py-8">
          <div class="mx-auto w-full">
            <h2 class="text-3xl font-bold mb-6 text-foreground">Article Tags</h2>
            <div class='flex flex-wrap gap-2'>
        """
        for md_tag in tags_markdown_str.split(','):
            md_tag_s = md_tag.strip()
            match = re.match(r'\[(.*?)\]\((.*?)\)', md_tag_s)
            if match:
                tag_text_display = match.group(1)
                tag_url_display = match.group(2)
                article_specific_tags_html_section += f"<a href='{tag_url_display}' target='_blank' rel='noopener noreferrer' class='inline-block rounded px-4 py-2 text-sm font-medium bg-muted text-muted-foreground hover:bg-primary hover:text-primary-foreground transition-colors'>#{tag_text_display}</a>\n"
            elif md_tag_s:
                tag_text_display = md_tag_s
                tag_search_url = f"{HYPERLINK_BASE_URL_FOR_TAGS}{requests.utils.quote(tag_text_display)}"
                article_specific_tags_html_section += f"<a href='{tag_search_url}' target='_blank' rel='noopener noreferrer' class='inline-block rounded px-4 py-2 text-sm font-medium bg-muted text-muted-foreground hover:bg-primary hover:text-primary-foreground transition-colors'>#{tag_text_display}</a>\n"
        article_specific_tags_html_section += "</div></div></section>\n"

    # --- SEO Meta Tags --- (Unchanged)
    canonical_url = f"https://news.visive.ai/{sanitize_filename(title)}"
    meta_description = excerpts if excerpts else f"Latest news and insights on Artificial Intelligence: {title}"
    meta_keywords = ', '.join([t.strip() for t in tags_markdown_str.split(',') if t.strip()])
    og_image = article_image_url_to_use if article_image_url_to_use else DEFAULT_ARTICLE_IMAGE_URL
    og_url = canonical_url
    publisher = "Visive.ai"
    author = "Visive AI News Team"
    json_ld = f'''<script type="application/ld+json">{{
      "@context": "https://schema.org",
      "@type": "NewsArticle",
      "headline": "{xml_escape(title)}",
      "image": ["{og_image}"],
      "datePublished": "{publish_date_iso_or_str}",
      "dateModified": "{publish_date_iso_or_str}",
      "author": {{"@type": "Person", "name": "{author}"}},
      "publisher": {{"@type": "Organization", "name": "{publisher}", "logo": {{"@type": "ImageObject", "url": "https://www.visive.ai/favicon.ico"}}}},
      "description": "{xml_escape(meta_description)}",
      "mainEntityOfPage": "{og_url}"
    }}</script>'''
    breadcrumbs_html = f"""
    <nav class="w-full py-4 text-base" aria-label="Breadcrumb" itemscope itemtype="https://schema.org/BreadcrumbList">
      <ol class="flex flex-wrap items-center gap-2">
        <li itemprop="itemListElement" itemscope itemtype="https://schema.org/ListItem">
          <a href="{'https://news.visive.ai/'}" class="text-muted-foreground hover:underline" itemprop="item"><span itemprop="name">Home</span></a>
          <meta itemprop="position" content="1" />
        </li>
        <li>/</li>
        <li itemprop="itemListElement" itemscope itemtype="https://schema.org/ListItem">
          <a href="{'https://news.visive.ai/news.html'}" class="text-muted-foreground hover:underline" itemprop="item"><span itemprop="name">News</span></a>
          <meta itemprop="position" content="2" />
        </li>
        <li>/</li>
        <li itemprop="itemListElement" itemscope itemtype="https://schema.org/ListItem">
          <span class="text-foreground" itemprop="name">{xml_escape(title)}</span>
          <meta itemprop="position" content="3" />
        </li>
      </ol>
    </nav>
    """

    # --- Responsive Navbar --- (Unchanged)
        # --- Responsive Navbar --- (CHANGED)
    navbar_html = _get_common_navbar_html()

    # --- Main Article Section --- (Unchanged)
    article_image_html = ''
    if article_image_url_to_use and article_image_url_to_use != DEFAULT_ARTICLE_IMAGE_URL:
        alt_text = xml_escape(title)
        article_image_html = (
            '<figure class="my-8 rounded-lg overflow-hidden shadow-md">'
            f'<img src="{article_image_url_to_use}" alt="{alt_text}" class="w-full h-[450px] object-cover">'
            '</figure>'
        )
    else:
        article_image_html = '<div class="my-8 w-full h-72 flex items-center justify-center rounded-lg bg-muted border border-dashed border-muted-foreground"><span class="text-muted-foreground text-xl">Featured Image Placeholder</span></div>'

    source_display = f'<span class="mx-1"></span><span>Source: {source_site}</span>' if source_site else ''

    # --- ### START OF CORRECTED SECTION ### ---
    # This section now generates cards with images for related articles.
    related_articles_cards = ""
    if related_articles_list:
        for article in related_articles_list:
            # Safely get the image URL, title, and filename for the related article
            related_image_url = article.get('image_url', DEFAULT_ARTICLE_IMAGE_URL)
            related_title = xml_escape(article.get('title', 'Related Article'))
            related_filename = article.get('html_filename', '#')

            related_articles_cards += f"""
              <div class="rounded-lg border bg-muted overflow-hidden flex flex-col shadow-md hover:shadow-xl transition-shadow duration-300">
                <a href="https://news.visive.ai/{related_filename}" class="block">
                  <img src="{related_image_url}" alt="Image for {related_title}" class="w-full h-40 object-cover">
                </a>
                <div class="p-4 flex flex-col flex-grow">
                  <h3 class="font-semibold text-lg mb-2 text-foreground flex-grow">{related_title}</h3>
                  <a href="https://news.visive.ai/{related_filename}" class="text-primary hover:underline font-medium text-base mt-auto self-start">Read Article →</a>
                </div>
              </div>
            """
    else:
        # This fallback is kept just in case, but should be hit less often now
        for i in range(6):
             related_articles_cards += f"""
              <div class="rounded-lg border p-4 bg-muted">
                <h3 class="font-semibold text-lg mb-2 text-foreground">Discover More News</h3>
                <p class="text-muted-foreground text-base">Explore other latest articles and insights on Artificial Intelligence.</p>
                <a href="https://news.visive.ai" class="text-primary hover:underline font-medium text-base">View All News →</a>
              </div>
            """

    related_articles_html = f"""
    <section class="w-full py-8">
      <div class="mx-auto w-full">
        <h2 class="text-3xl font-bold mb-6 text-foreground">Related News Articles</h2>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {related_articles_cards}
        </div>
      </div>
    </section>
    """
    # --- ### END OF CORRECTED SECTION ### ---

    # --- Final HTML --- (Unchanged)
    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{xml_escape(title)} | AI News | Visive</title>
    <meta name="description" content="{xml_escape(meta_description)}">
    <meta name="keywords" content="{xml_escape(meta_keywords)}">
    <meta name="robots" content="index, follow">
    <link rel="canonical" href="{canonical_url}">
    <link rel="icon" href="https://www.visive.ai/favicon.ico" type="image/x-icon">
    <!-- Open Graph -->
    <meta property="og:title" content="{xml_escape(title)}">
    <meta property="og:description" content="{xml_escape(meta_description)}">
    <meta property="og:image" content="{og_image}">
    <meta property="og:type" content="article">
    <meta property="og:url" content="{og_url}">
    <!-- Twitter Card -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{xml_escape(title)}">
    <meta name="twitter:description" content="{xml_escape(meta_description)}">
    <meta name="twitter:image" content="{og_image}">
    {json_ld}
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Merriweather:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
    <style>
      :root {{
        --primary: #1A73E8; --primary-foreground: #FFFFFF;
        --muted: #EEF4FF; --muted-foreground: #4A5568;
        --background: #FFFFFF; --foreground: #212529;
        --border: #D2E3FC;
      }}
      body {{ font-family: 'Inter', sans-serif; background: var(--background); color: var(--foreground); font-size: 1.1rem; line-height: 1.6; }}
      .bg-muted {{ background-color: var(--muted); }} .text-muted-foreground {{ color: var(--muted-foreground); }}
      .text-primary {{ color: var(--primary); }} .hover\\:text-primary-foreground:hover {{ color: var(--primary-foreground); }}
      .hover\\:bg-primary:hover {{ background-color: var(--primary); }} .rounded-lg {{ border-radius: 0.75rem; }}
      .border {{ border: 1px solid var(--border); }} .bg-background {{ background-color: var(--background); }}
      .text-foreground {{ color: var(--foreground); }} .font-bold {{ font-weight: 700; }} .font-semibold {{ font-weight: 600; }}
      .transition-colors {{ transition-property: color, background-color, border-color, text-decoration-color, fill, stroke; transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1); transition-duration: 150ms; }}
      .hover\\:underline:hover {{ text-decoration: underline; }} .backdrop-filter {{ backdrop-filter: blur(10px); }}
      .backdrop-blur-lg {{ backdrop-filter: blur(10px); }}

      .prose {{ font-size: 1.15rem; line-height: 1.8; }}
      .prose h1 {{ font-size: 2.1rem; line-height: 1.2; margin-top: 1.3em; margin-bottom: 0.6em; }}
      .prose h2 {{ font-size: 1.8rem; line-height: 1.3; margin-top: 1em; margin-bottom: 0.5em; }}
      .prose h3 {{ font-size: 1.4rem; line-height: 1.4; margin-top: 1em; margin-bottom: 0.5em; }}
      .prose p {{ margin-top: 1em; margin-bottom: 1em; }}
      .prose img {{ margin-top: 1.5em; margin-bottom: 1.5em; }}
      /* NEW: Styles for lists */
      .prose ul, .prose ol {{ margin-top: 1em; margin-bottom: 1em; padding-left: 2em; }}
      .prose ul li, .prose ol li {{ margin-top: 0.5em; }}
      .prose ul {{ list-style-type: disc; }}
      .prose ol {{ list-style-type: decimal; }}
    </style></head>
<body class="antialiased w-full">{navbar_html}<main class="w-full pt-16">
    <div class="px-4 md:px-8 lg:px-12 w-full mx-auto max-w-screen-xl">{breadcrumbs_html}
        <article class="mt-2">
            <header class="mb-6">
                <h1 class="text-4xl lg:text-5xl font-bold mt-4 mb-4 text-foreground">{xml_escape(title)}</h1>
                {f'<p class="text-muted-foreground text-2xl mb-4">{xml_escape(excerpts)}</p>' if excerpts else ''}
                <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-base text-muted-foreground mb-4">
                    <span>{display_date}</span>{source_display}
                </div>
            </header>

            {article_image_html}
            <div class="prose prose-xl max-w-none mt-8">
                {content_html}
            </div>
            {faqs_html_list}
            {article_specific_tags_html_section}
            {related_articles_html}
        </article>
    </div>
</main>
<footer class="w-full py-10 mt-16 bg-background border-t text-center text-muted-foreground"><p class="text-base">© {datetime.now().year} Visive.ai. All rights reserved.</p></footer></body></html>"""



def create_index_page_html(articles, all_articles, active_categories, pagination_info):
    """
    Creates the special, non-paginated homepage with Top Story, Featured, and Latest News sections.
    Includes hero animation, dates, proper headings, AND A PAGINATION CONTROL.
    """
    # --- Head, Animation Styles, and Hero Section (Unchanged) ---
    page_title = "The Forefront of AI"
    html = f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visive AI News - {page_title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
      :root{{--primary:#3B82F6;--muted:#F9FAFB;--muted-foreground:#6B7280;--background:#FFFFFF;--foreground:#111827;--border:#E5E7EB;}}
      body{{font-family: 'Inter', sans-serif;}}
      @keyframes slide-up-fade {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}
      .animate-hero {{ animation: slide-up-fade 0.7s ease-out forwards; }}
    </style>
    </head><body>{_get_common_navbar_html()}
    <main class="w-full pt-16">
        <div class="px-4 md:px-8 lg:px-12 w-full mx-auto max-w-screen-xl py-8 md:py-12">
        <section class="text-center mb-12 md:mb-16">
            <h1 class="text-4xl md:text-5xl lg:text-6xl font-extrabold tracking-tighter text-foreground animate-hero" style="animation-delay: 0.1s;">{page_title}</h1>
            <p class="mt-4 max-w-3xl mx-auto text-lg md:text-xl text-muted-foreground animate-hero" style="animation-delay: 0.2s;">Your daily source for curated news and breakthroughs from the world of AI.</p>
        </section>"""
    
    top_story = articles[0] if articles else None
    featured_articles = articles[1:5] if len(articles) > 1 else []
    latest_articles = articles[5:] if len(articles) > 5 else []

    if top_story:
        article_date = datetime.fromisoformat(top_story['published_date_iso'].replace('Z', '+00:00')).strftime("%B %d, %Y")
        html += f"""<section class="mb-12"><h2 class="text-3xl font-bold tracking-tight text-foreground mb-6">Top Story</h2>
        <a href="{top_story['html_filename']}" class="block group"><div class="grid lg:grid-cols-2 gap-8 items-center">
        <div class="overflow-hidden rounded-xl"><img src="{top_story['image_url']}" alt="{xml_escape(top_story['title'])}" class="w-full h-auto aspect-video object-cover transition-transform duration-300 group-hover:scale-105"></div>
        <div><h3 class="text-3xl lg:text-4xl font-bold text-foreground mb-4 group-hover:text-primary transition-colors">{xml_escape(top_story['title'])}</h3>
        <p class="text-lg text-muted-foreground line-clamp-3 mb-4">{xml_escape(top_story['excerpts'])}</p>
        <p class="text-sm font-medium text-muted-foreground">{article_date}</p></div></div></a></section>"""
    
    if featured_articles:
        html += f"""<section class="mb-12"><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">"""
        for article in featured_articles:
            article_date = datetime.fromisoformat(article['published_date_iso'].replace('Z', '+00:00')).strftime("%B %d, %Y")
            html += f"""<a href="{article['html_filename']}" class="block group"><div class="overflow-hidden rounded-lg">
            <img src="{article['image_url']}" alt="{xml_escape(article['title'])}" class="w-full h-40 object-cover transition-transform duration-300 group-hover:scale-105"></div>
            <h3 class="mt-4 text-lg font-bold text-foreground group-hover:text-primary transition-colors leading-tight">{xml_escape(article['title'])}</h3>
            <p class="text-sm text-muted-foreground mt-2">{article_date}</p></a>"""
        html += """</div></section>"""

    if latest_articles:
        html += """<section><div class="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12"><div class="lg:col-span-8"><h2 class="text-3xl font-bold tracking-tight text-foreground mb-6">Latest News</h2><div class="space-y-8">"""
        for article in latest_articles:
            article_date = datetime.fromisoformat(article['published_date_iso'].replace('Z', '+00:00')).strftime("%B %d, %Y")
            html += f"""<a href="{article['html_filename']}" class="block group"><div class="grid grid-cols-1 sm:grid-cols-3 gap-6 items-start">
            <div class="sm:col-span-1"><img src="{article['image_url']}" alt="{xml_escape(article['title'])}" class="w-full h-full aspect-video object-cover rounded-lg"></div>
            <div class="sm:col-span-2"><h3 class="text-xl font-bold text-foreground group-hover:text-primary transition-colors mb-2 leading-tight">{xml_escape(article['title'])}</h3>
            <p class="text-muted-foreground line-clamp-2 mb-3">{xml_escape(article['excerpts'])}</p>
            <p class="text-sm font-medium text-muted-foreground">{article_date}</p></div></div></a>"""
        html += """</div></div><aside class="lg:col-span-4"><div class="sticky top-24 space-y-8">"""
        html += f"""<div class="p-6 bg-muted rounded-lg border"><h3 class="text-xl font-bold text-foreground mb-4">Categories</h3><ul class="space-y-2">"""
        for cat_name, cat_slug in active_categories.items():
            html += f"""<li><a href="category-{cat_slug}.html" class="flex justify-between items-center text-muted-foreground hover:text-primary font-medium"><span>{cat_name}</span><span>→</span></a></li>"""
        html += """</ul></div>"""
        html += f"""<div class="p-6 bg-muted rounded-lg border"><h3 class="text-xl font-bold text-foreground mb-4">Recent Posts</h3><ul class="space-y-4">"""
        for art in all_articles[:5]:
            html += f"""<li><a href="{art['html_filename']}" class="font-semibold text-foreground hover:text-primary leading-tight">{xml_escape(art['title'])}</a></li>"""
        html += """</ul></div></div></aside></div></section>"""

    # --- START OF ADDED PAGINATION BLOCK ---
    if pagination_info and pagination_info['total_pages'] > 1:
        html += f"""
        <nav class="mt-16 border-t border-border pt-8 flex items-center justify-between">
            <span class="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border bg-muted text-muted-foreground cursor-not-allowed">← Previous</span>
            <p class='text-sm text-muted-foreground'>Page 1 of {pagination_info['total_pages']}</p>
            <a href="page-2.html" class="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border bg-white text-foreground hover:bg-muted transition-colors">Next →</a>
        </nav>"""
    # --- END OF ADDED PAGINATION BLOCK ---

    html += """</div></main><footer class="w-full py-10 mt-8 bg-muted border-t text-center text-muted-foreground"><p>© {datetime.now().year} Visive.ai. All rights reserved.</p></footer></body></html>"""
    return html


def create_news_listing_page_html(articles, all_articles, active_categories):
    """
    Creates a dedicated and uniquely designed 'news.html' page.
    Features a modern layout with the heading placed before the breadcrumbs and reduced top space.
    """
    page_title = "News"
    html = f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{page_title} | Visive AI News</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>:root{{--primary:#1A73E8;--muted:#F9FAFB;--muted-foreground:#6B7280;--background:#FFFFFF;--foreground:#111827;--border:#E5E7EB;}}body{{font-family: 'Inter', sans-serif;}}</style>
    </head><body>{_get_common_navbar_html()}
    <main class="w-full pt-16">
        <div class="w-full mx-auto max-w-screen-xl px-4 md:px-8 lg:px-12">
            <!-- Section Header -->
            <div class="pt-8 pb-6">
                <h1 class="text-4xl font-extrabold tracking-tight text-foreground">{page_title}</h1>
                <nav class="mt-2 text-sm" aria-label="Breadcrumb">
                    <ol class="flex items-center gap-2 text-muted-foreground">
                        <li><a href="index.html" class="hover:text-primary transition-colors">Home</a></li>
                        <li><span class="font-bold">/</span></li>
                        <li><span class="font-medium text-foreground">AI News</span></li>
                    </ol>
                </nav>
            </div>"""
    
    top_story = articles[0] if articles else None
    remaining_articles = articles[1:] if len(articles) > 1 else []

    if top_story:
        article_date = datetime.fromisoformat(top_story['published_date_iso'].replace('Z', '+00:00')).strftime("%B %d, %Y")
        html += f"""
        <section class="py-6 border-t border-b border-border">
            <h2 class="text-2xl font-bold tracking-tight text-foreground mb-4">Top Story</h2>
            <a href="{top_story['html_filename']}" class="block group">
                <div class="grid lg:grid-cols-2 gap-8 items-center">
                    <div class="overflow-hidden rounded-xl"><img src="{top_story['image_url']}" alt="{xml_escape(top_story['title'])}" class="w-full h-auto aspect-video object-cover transition-transform duration-300 group-hover:scale-105"></div>
                    <div>
                        <h3 class="text-2xl md:text-3xl font-bold text-foreground mb-3 group-hover:text-primary transition-colors">{xml_escape(top_story['title'])}</h3>
                        <p class="text-base text-muted-foreground line-clamp-3 mb-4">{xml_escape(top_story['excerpts'])}</p>
                        <p class="text-sm font-medium text-muted-foreground mt-auto">{article_date}</p>
                    </div>
                </div>
            </a>
        </section>"""

    if remaining_articles:
        html += """<section class="py-12"><h2 class="text-2xl font-bold tracking-tight text-foreground mb-6">More News</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">"""
        for article in remaining_articles:
            article_date = datetime.fromisoformat(article['published_date_iso'].replace('Z', '+00:00')).strftime("%B %d, %Y")
            html += f"""
                <a href="{article['html_filename']}" class="block group flex flex-col bg-white rounded-xl border shadow-sm hover:shadow-lg transition-all duration-300">
                    <div class="h-48 overflow-hidden rounded-t-xl"><img src="{article['image_url']}" alt="{xml_escape(article['title'])}" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"></div>
                    <div class="p-4 md:p-5 flex flex-col flex-grow"><h3 class="text-lg font-bold text-foreground leading-tight">{xml_escape(article['title'])}</h3>
                    <p class="mt-auto pt-4 text-sm font-medium text-muted-foreground">{article_date}</p></div>
                </a>"""
        html += """</div></section>"""
        
    html += """</div></main><footer class="w-full py-10 mt-8 bg-muted border-t text-center text-muted-foreground">
        <p>©2025 Visive.ai. All rights reserved.</p></footer></body></html>"""
    return html

def create_paginated_page_html(articles, all_articles, active_categories, page_title, pagination_info):
    """
    Creates a clean, paginated archive page with CORRECTED pagination links.
    Used for categories and main news archives (e.g., page 2 and beyond).
    """
    html = f"""<!DOCTYPE html><html lang="en"><head><title>{xml_escape(page_title)} | Visive AI News</title>
    <script src="https://cdn.tailwindcss.com"></script><style>:root{{--primary:#1A73E8;--muted:#F9FAFB;--muted-foreground:#6B7280;--background:#FFFFFF;--foreground:#111827;--border:#E5E7EB;}}body{{font-family: 'Inter', sans-serif;}}</style></head>
    <body>{_get_common_navbar_html()}
    <main class="w-full pt-16"><div class="px-4 md:px-8 lg:px-12 w-full mx-auto max-w-screen-xl py-12">
    <h1 class="text-4xl font-extrabold tracking-tighter text-foreground mb-12">{xml_escape(page_title)}</h1>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">"""
    
    for article in articles:
        article_date = datetime.fromisoformat(article['published_date_iso'].replace('Z', '+00:00')).strftime("%B %d, %Y")
        html += f"""<a href="{article['html_filename']}" class="block group"><div class="overflow-hidden rounded-lg">
        <img src="{article['image_url']}" alt="{xml_escape(article['title'])}" class="w-full h-48 object-cover transition-transform duration-300 group-hover:scale-105"></div>
        <h2 class="mt-4 text-lg font-bold text-foreground group-hover:text-primary transition-colors leading-tight">{xml_escape(article['title'])}</h2>
        <p class="text-sm text-muted-foreground mt-2">{article_date}</p></a>"""
    html += """</div>"""
    
    if pagination_info['total_pages'] > 1:
        html += """<nav class="mt-16 col-span-full flex items-center justify-between border-t border-border pt-8">"""
        # --- PREVIOUS BUTTON (Corrected Logic) ---
        if pagination_info['current_page'] > 1:
            prev_page = pagination_info['current_page'] - 1
            if prev_page == 1:
                href = "index.html" if pagination_info['base_url'] == "" else f"{pagination_info['base_url']}.html"
            else:
                href = f"page-{prev_page}.html" if pagination_info['base_url'] == "" else f"{pagination_info['base_url']}-page-{prev_page}.html"
            html += f'<a href="{href}" class="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border bg-white text-foreground hover:bg-muted transition-colors">← Previous</a>'
        else:
            html += '<span class="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border bg-muted text-muted-foreground cursor-not-allowed">← Previous</span>'
        
        html += f"<p class='hidden md:block text-sm text-muted-foreground'>Page {pagination_info['current_page']} of {pagination_info['total_pages']}</p>"
        
        # --- NEXT BUTTON (Corrected Logic) ---
        if pagination_info['current_page'] < pagination_info['total_pages']:
            next_page = pagination_info['current_page'] + 1
            href = f"page-{next_page}.html" if pagination_info['base_url'] == "" else f"{pagination_info['base_url']}-page-{next_page}.html"
            html += f'<a href="{href}" class="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border bg-white text-foreground hover:bg-muted transition-colors">Next →</a>'
        else:
            html += '<span class="inline-flex items-center px-4 py-2 text-sm font-medium rounded-md border bg-muted text-muted-foreground cursor-not-allowed">Next →</span>'
            
        html += "</nav>"
        
    html += """</div></main></body></html>"""
    return html


def generate_news_sitemap(articles_db):
    print("  Generating news sitemap...")
    sitemap_path = os.path.join(OUTPUT_HTML_DIR, NEWS_SITEMAP_FILENAME)
    # Fix: Add proper null check for processed_llm_data
    published_articles = []
    for item in articles_db.values():
        if (item.get('status') == 'html_generated' and 
            item.get('html_filename') and 
            item.get('processed_llm_data') and 
            isinstance(item['processed_llm_data'], dict) and
            item['processed_llm_data'].get('final_title') and 
            item.get('published_date_iso')):
            published_articles.append(item)
    
    published_articles.sort(key=lambda x: x['published_date_iso'], reverse=True)
    with open(sitemap_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">\n')
        for article in published_articles:
            pd = article['processed_llm_data']; loc = f"https://news.visive.ai/{article['html_filename']}"; title = xml_escape(pd['final_title']); pub_date = article['published_date_iso']
            f.write(f'  <url>\n    <loc>{loc}</loc>\n    <news:news>\n      <news:publication>\n        <news:name>Visive.ai</news:name>\n        <news:language>en</news:language>\n      </news:publication>\n      <news:publication_date>{pub_date}</news:publication_date>\n      <news:title>{title}</news:title>\n    </news:news>\n  </url>\n')
        f.write("</urlset>\n")
    print(f"  {NEWS_SITEMAP_FILENAME} generated with {len(published_articles)} entries.")



import math # Make sure this import is at the top of your Python file

def generate_and_save_html_pages(articles_db):
    """
    Orchestrates the generation of the site. It regenerates main/category/news pages every run,
    but ONLY generates HTML for NEW individual articles to improve efficiency.
    """
    print("\n>>> Starting Professional Site Generation (Efficient Mode) <<<")
    if not os.path.exists(OUTPUT_HTML_DIR):
        os.makedirs(OUTPUT_HTML_DIR)

    # 1. GATHER ALL PUBLISHABLE DATA (This part is unchanged and correct)
    all_publishable_articles_info = []
    for aid, d in articles_db.items():
        processed_llm_data = d.get('processed_llm_data')
        if processed_llm_data and isinstance(processed_llm_data, dict) and processed_llm_data.get('final_title'):
            all_publishable_articles_info.append({
                'id': aid, 'title': processed_llm_data['final_title'],
                'excerpts': processed_llm_data.get('final_excerpts', ''),
                'html_filename': sanitize_filename(processed_llm_data['final_title']),
                'image_url': d.get('final_image_url', DEFAULT_ARTICLE_IMAGE_URL),
                'published_date_iso': d.get('published_date_iso', datetime.now(timezone.utc).isoformat())
            })
    
    sorted_articles = sorted(all_publishable_articles_info, key=lambda x: x['published_date_iso'], reverse=True)
    if not sorted_articles:
        print("  No publishable articles found. Skipping HTML generation.")
        return articles_db

    # 2. DEFINE SITE STRUCTURE & ACTIVE CATEGORIES (Unchanged and correct)
    ARTICLES_PER_PAGE = 12
    PREDEFINED_CATEGORIES = {
        "Machine Learning": "machine-learning", "Hardware": "hardware", "Ethics": "ethics",
        "NLP": "nlp", "Robotics": "robotics", "Business": "business", "Innovation": "innovation"
    }
    
    active_categories = {}
    for cat_name, cat_slug in PREDEFINED_CATEGORIES.items():
        if any(cat_name.lower() in a['title'].lower() or cat_name.lower() in a['excerpts'].lower() for a in sorted_articles):
            active_categories[cat_name] = cat_slug

    print(f"  > Found {len(active_categories)} active categories: {list(active_categories.keys())}")  

    # 3. REGENERATE MAIN, NEWS, AND CATEGORY PAGES (Unchanged and correct)
    # These pages MUST be regenerated every run to reflect the latest content.
    print("  Regenerating main, news, and category pages...")
    
    # Homepage (index.html)
    homepage_articles = sorted_articles[:13]
    total_pages = math.ceil(len(sorted_articles) / ARTICLES_PER_PAGE)
    pagination_info = {"current_page": 1, "total_pages": total_pages, "base_url": ""}
    homepage_html = create_index_page_html(homepage_articles, sorted_articles, active_categories, pagination_info)
    with open(os.path.join(OUTPUT_HTML_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(homepage_html)

    # News Listing Page (news.html)
    news_page_articles = sorted_articles[:16]
    news_page_html = create_news_listing_page_html(news_page_articles, sorted_articles, active_categories)
    with open(os.path.join(OUTPUT_HTML_DIR, "news.html"), "w", encoding="utf-8") as f:
        f.write(news_page_html)   

    # Main Paginated Archive (page-2.html, etc.)
    for page_num in range(2, total_pages + 1):
        start_index = (page_num - 1) * ARTICLES_PER_PAGE
        articles_on_page = sorted_articles[start_index : start_index + ARTICLES_PER_PAGE]
        filename = f"page-{page_num}.html"
        pagination_info = {"current_page": page_num, "total_pages": total_pages, "base_url": ""}
        page_html = create_paginated_page_html(articles_on_page, sorted_articles, active_categories, f"Latest News - Page {page_num}", pagination_info)
        with open(os.path.join(OUTPUT_HTML_DIR, filename), "w", encoding="utf-8") as f:
            f.write(page_html)

    # Paginated Category Archives
    for cat_name, cat_slug in active_categories.items():
        cat_articles = [a for a in sorted_articles if cat_name.lower() in a['title'].lower() or cat_name.lower() in a['excerpts'].lower()]
        total_cat_pages = math.ceil(len(cat_articles) / ARTICLES_PER_PAGE)
        
        for page_num in range(1, total_cat_pages + 1):
            start_index = (page_num - 1) * ARTICLES_PER_PAGE
            articles_on_page = cat_articles[start_index : start_index + ARTICLES_PER_PAGE]
            filename = f"category-{cat_slug}.html" if page_num == 1 else f"category-{cat_slug}-page-{page_num}.html"
            pagination_info = {"current_page": page_num, "total_pages": total_cat_pages, "base_url": f"category-{cat_slug}"}
            page_title = f"Category: {cat_name}" + (f" - Page {page_num}" if total_cat_pages > 1 else "")
            page_html = create_paginated_page_html(articles_on_page, sorted_articles, active_categories, page_title, pagination_info)
            with open(os.path.join(OUTPUT_HTML_DIR, filename), "w", encoding="utf-8") as f:
                f.write(page_html)

    # 4. EFFICIENTLY GENERATE ONLY NEW INDIVIDUAL ARTICLE PAGES
    print("  Generating HTML for new articles only...")
    newly_generated_count = 0
    # This is the "work queue": articles that have an image but do not yet have their HTML generated.
    articles_to_publish_ids = [aid for aid, data in articles_db.items() if data.get('status') in ['image_generated', 'image_generation_failed']]
    
    if not articles_to_publish_ids:
        print("  No new articles to generate HTML for.")
    else:
        for article_id in articles_to_publish_ids:
            article_data = articles_db[article_id]
            
            article_info = next((info for info in all_publishable_articles_info if info['id'] == article_id), None)
            if not article_info or not article_data.get('processed_llm_data'):
                continue

            template_data = article_data['processed_llm_data'].copy()
            template_data['article_image_url_to_use'] = article_info['image_url']
            
            potential_related = [a for a in sorted_articles if a['id'] != article_id]
            related_for_template = random.sample(potential_related, min(3, len(potential_related)))

            html_content = create_single_news_blog_page_html_local(template_data, related_for_template)
            filename = article_info['html_filename']
            with open(os.path.join(OUTPUT_HTML_DIR, filename), "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # CRITICAL: Update the status to 'html_generated' to prevent regeneration on the next run.
            article_data['html_filename'] = filename
            article_data['status'] = 'html_generated'
            newly_generated_count += 1
            print(f"    + Generated new article page: {filename}")

    print(f"  Generated {newly_generated_count} new individual article pages.")

    # 5. FINALIZE (Unchanged)
    generate_news_sitemap(articles_db)
    save_articles_db(articles_db)
    print(f"Professional Site Generation Complete: All pages are now up-to-date.")
    return articles_db
# --------------------------------------------------
# 8. GITHUB UPLOADER
# --------------------------------------------------
# (This section remains unchanged)
def upload_html_pages_to_github(gh_token, repo_owner_str, repo_name_str, branch_str, commit_msg_base, local_html_folder):
    if not all([gh_token, repo_owner_str, repo_name_str, branch_str]): print("ERROR: GitHub configuration is incomplete."); return
    g = Github(gh_token)
    try: repo = g.get_repo(f"{repo_owner_str}/{repo_name_str}"); print(f"GitHub: Accessed {repo.full_name}")
    except Exception as e: print(f"GitHub: Error accessing repo: {e}"); return
    if not os.path.exists(local_html_folder) or not os.listdir(local_html_folder): print(f"GitHub: No files in '{local_html_folder}' to upload."); return
    tree_elements, committed_files = [], []
    for filename in os.listdir(local_html_folder):
        if filename.endswith((".htm", ".html", ".xml")):
            local_path = os.path.join(local_html_folder, filename)
            with open(local_path, "r", encoding="utf-8") as f: content = f.read()
            repo_path = os.path.join(HTML_SUBFOLDER_IN_REPO, filename).replace("\\", "/")
            try:
                existing = repo.get_contents(repo_path, ref=branch_str)
                if base64.b64decode(existing.content).decode('utf-8') == content: print(f"  GitHub: Skipping '{repo_path}', unchanged."); continue
            except Exception: pass # File is new
            print(f"  GitHub: Preparing '{repo_path}'...")
            blob = repo.create_git_blob(content, "utf-8")
            tree_elements.append(InputGitTreeElement(path=repo_path, mode='100644', type='blob', sha=blob.sha))
            committed_files.append(filename)
    if not tree_elements: print("GitHub: No new/modified files to commit."); return
    try:
        branch_obj = repo.get_branch(branch_str)
        base_tree = repo.get_git_tree(sha=branch_obj.commit.commit.tree.sha)
        new_tree = repo.create_git_tree(tree_elements, base_tree)
        msg = f"{commit_msg_base}: {', '.join(committed_files[:3])}{'...' if len(committed_files)>3 else ''}"
        new_commit = repo.create_git_commit(msg, new_tree, [repo.get_git_commit(branch_obj.commit.sha)])
        ref = repo.get_git_ref(f"heads/{branch_str}")
        ref.edit(new_commit.sha)
        print(f"  GitHub: Branch '{branch_str}' updated. {len(committed_files)} file(s) committed.")
    except Exception as e: print(f"  GitHub: Commit/push error: {e}")

# --------------------------------------------------
# 9. MAIN PIPELINE ORCHESTRATION (MODIFIED)
# --------------------------------------------------
def run_complete_pipeline():
    """Runs the entire news processing pipeline, managed by a state machine."""
    print("🚀 Starting AI News Full Processing Pipeline (Stateful with Image Generation)...")
    pipeline_state = load_pipeline_state()
    articles_database = load_articles_db()
    
    # UPDATED PIPELINE FLOW
    pipeline_steps = [
        "FETCH_RSS", "PROCESS_READABILITY", "REWRITE_LLM",
        "PROCESS_LLM_OUTPUT", "GENERATE_IMAGES", "GENERATE_HTML", "UPLOAD_GITHUB"
    ]
    
    try:
        current_step_name = pipeline_state.get("current_step", "FETCH_RSS")
        while current_step_name in pipeline_steps:
            print(f"\n===== EXECUTING PIPELINE STEP: {current_step_name} =====")
            
            if current_step_name == "FETCH_RSS":
                articles_database = fetch_new_rss_articles(articles_database)
                current_step_name = "PROCESS_READABILITY"
            
            elif current_step_name == "PROCESS_READABILITY":
                articles_database = process_readability_for_articles(articles_database)
                current_step_name = "REWRITE_LLM"
            
            elif current_step_name == "REWRITE_LLM":
                articles_database = rewrite_articles_with_llm(articles_database)
                current_step_name = "PROCESS_LLM_OUTPUT"
            
            elif current_step_name == "PROCESS_LLM_OUTPUT":
                articles_database = process_llm_outputs_and_prepare_publish_data(articles_database)
                current_step_name = "GENERATE_IMAGES" # New Transition
            
            elif current_step_name == "GENERATE_IMAGES": # New Step
                articles_database = generate_images_for_articles(articles_database)
                current_step_name = "GENERATE_HTML"
            
            elif current_step_name == "GENERATE_HTML":
                articles_database = generate_and_save_html_pages(articles_database)
                current_step_name = "UPLOAD_GITHUB"
            
            elif current_step_name == "UPLOAD_GITHUB":
                upload_html_pages_to_github(
                    gh_token=GITS_TOKEN, repo_owner_str=REPO_OWNER, repo_name_str=REPO_NAME,
                    branch_str=BRANCH_NAME, commit_msg_base=COMMIT_MESSAGE_BASE,
                    local_html_folder=OUTPUT_HTML_DIR
                )
                current_step_name = "FETCH_RSS" # Reset for next run
                pipeline_state["current_step"] = current_step_name
                save_pipeline_state(pipeline_state)
                break # Full cycle complete
            
            pipeline_state["current_step"] = current_step_name
            save_pipeline_state(pipeline_state)

    except Exception as e:
        print(f"\n🔥🔥🔥 PIPELINE HALTED DUE TO AN ERROR AT STEP '{pipeline_state.get('current_step')}': {e} 🔥🔥🔥")
        return
        
    print("\n🏁 AI News Full Processing Pipeline cycle finished.")

if __name__ == "__main__":
    # Updated checks for new .env requirements
    config_ok = True
    if not DEEPINFRA_API_KEY:
        print("🚨 CRITICAL ERROR: 'DEEPINFRA_API_KEY' not found in your .env file.")
        config_ok = False
    if not GEMINI_API_KEYS:
        print("🚨 CRITICAL ERROR: 'GEMINI_API_KEYS' not found or empty in your .env file.")
        config_ok = False
    if not GITS_TOKEN or GITS_TOKEN == "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN" or len(GITS_TOKEN) < 30:
        print("🚨 CRITICAL ERROR: 'GITS_TOKEN' is not set correctly in your .env file.")
        config_ok = False
    if not DRIVE_FOLDER_ID:
        print("🚨 CRITICAL ERROR: 'DRIVE_FOLDER_ID' not found in your .env file.")
        config_ok = False
    if not SERVICE_ACCOUNT_FILE_PATH:
        print("🚨 CRITICAL ERROR: 'SERVICE_ACCOUNT_FILE_PATH' not found in your .env file.")
        config_ok = False
    elif not os.path.exists(SERVICE_ACCOUNT_FILE_PATH):
        print(f"🚨 CRITICAL ERROR: Service account file not found at the path specified in .env: '{SERVICE_ACCOUNT_FILE_PATH}'")
        config_ok = False

    if config_ok:
        print("✅ All required configurations loaded successfully.")
        run_complete_pipeline()
    else:
        print("\nConfiguration errors found. Please fix your .env file and try again.")