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

        return {
            "license": data.get("license", {}).get("name"),
            "description": data.get("description"),
            "topics": data.get("topics"),
            "language": data.get("language"),
            "updated_at": data.get("updated_at"),
            "stars": data.get("stargazers_count"),
            "url": data.get("html_url")
        }
    return {}
    