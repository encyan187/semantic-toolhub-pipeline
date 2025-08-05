#!/usr/bin/env python3
import requests
import sys

# ── Configuration ────────────────────────────────────────────────
API_BASE    = "https://llama-max.ai.wu.ac.at"
ENDPOINT    = "/api/models"
                                   # ← your key here

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept":        "application/json",
}

def list_models():
    url = API_BASE.rstrip("/") + ENDPOINT
    try:
        resp = requests.get(url, headers=HEADERS)
    except requests.RequestException as e:
        print("❌ Network error:", e)
        sys.exit(1)

    # 1. Check HTTP status
    if resp.status_code != 200:
        print(f"❌ HTTP {resp.status_code} error")
        print("Response body:\n", resp.text)
        sys.exit(1)

    # 2. Parse JSON
    try:
        payload = resp.json()
    except ValueError as e:
        print("❌ JSON decode error:", e)
        print("Response text (truncated):\n", resp.text[:500])
        sys.exit(1)

    # 3. Extract the "data" list
    models = payload.get("data", [])
    if not models:
        print("→ No chat models found under /api/models.")
        return

    # 4. Print each model’s ID and (optional) name
    print("Available chat models:")
    for m in models:
        mid  = m.get("id", "<no-id>")
        name = m.get("name", "")
        print(f" - {mid}{' ('+name+')' if name else ''}")

if __name__ == "__main__":
    list_models()
