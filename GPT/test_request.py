import re
import requests
import sys

# ── 1. Configuration ────────────────────────────────────────────────
API_BASE    = "https://llama-max.ai.wu.ac.at"               # ← your WebUI URL
MODEL_ID    = "deepseek-r1:latest"                             # ← one of the IDs you saw
API_KEY     = "sk-456ff07dfe994cbc97d9cf0a5d73aeb7"        # ← your API key

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

# ── 2. Build your few‑shot conversation ────────────────────────────
# The "system" message is optional but helps set overall behavior.
messages = [
    {
        "role":    "system",
        "content": "You are a helpful assistant that translates English to German. Answer precisely and correctly."
    },
    # Example 1
    {
        "role":    "user",
        "content": "Translate to German: 'Good morning.'"
    },
    {
        "role":    "assistant",
        "content": "Guten Morgen."
    },
    # Example 2
    {
        "role":    "user",
        "content": "Translate to German: 'How are you today?'"
    },
    {
        "role":    "assistant",
        "content": "Wie geht es dir heute?"
    },
    # Your real question goes last
    {
        "role":    "user",
        "content": "Translate to German: 'I love my mother.'"
    }
]

# ── 3. Send the chat‐completions request ────────────────────────────
url = f"{API_BASE}/api/chat/completions"
payload = {
    "model":    MODEL_ID,
    "messages": messages,
    # "stream": True   # uncomment to stream tokens instead of getting one big reply
}

try:
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
except requests.RequestException as e:
    print("Request failed:", e)
    print("Status code:", getattr(e.response, "status_code", None))
    print("Response body:", getattr(e.response, "text", None))
    sys.exit(1)

# ── 4. Parse and print the reply ────────────────────────────────────
data = resp.json()
reply = data["choices"][0]["message"]["content"]
clean = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

print("Model says:", clean)
