import requests, json, pandas as pd
from tqdm import tqdm
import unicodedata


# Normalize input such as "Protégé" in order to search for GitHub Repo correctly

def normalize_name(name):
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')

def search_github_repo(tool_name, token=None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    normalized_name = normalize_name(tool_name).lower()
    query = f"{normalized_name} in:name"
    url = f"https://api.github.com/search/repositories?q={query}&sort=stars"
    r = requests.get(url, headers=headers)
    
    if r.status_code == 200:
        items = r.json().get("items")
        if items:
            return items[0]["full_name"]
            
    return None

def get_readme(repo_name, token=None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{repo_name}/readme"
    r = requests.get(url, headers=headers)
    
    if r.status_code == 200:
        import base64
        content = r.json().get("content")
        return base64.b64decode(content).decode("utf-8")
    return None

def get_repo_metadata(repo_name, token=None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f"https://api.github.com/repos/{repo_name}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()

        license_name = (data.get("license") or {}).get("name")
        topics = data.get("topics") or []
        description = data.get("description") or ""
        language = data.get("language") or ""
        updated_at = data.get("updated_at") or ""
        stars = data.get("stargazers_count") or 0
        html_url = data.get("html_url") or ""

        return {
            "license": license_name,
            "description": description,
            "topics": topics,
            "language": language,
            "updated_at": updated_at,
            "stars": stars,
            "url": html_url
        }

        # In case of API errors or rate limits
    return {
        "license": None,
        "description": None,
        "topics": [],
        "language": None,
        "updated_at": None,
        "stars": 0,
        "url": None
    }

import re

def clean_readme(readme_text):
    if not isinstance(readme_text, str):
        return ""

    # Remove markdown headers and formatting
    text = re.sub(r"#+\s*", "", readme_text)  # remove markdown headers
    text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)  # remove inline or block code
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # italics
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # [text](url) → text

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove badges (often image links at top)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Remove remaining URLs
    text = re.sub(r"http\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

    