import re

from gpt_config import *
import requests
import sys

def classify_tool(tool_name, tool_desc):

    prompt = PROMPT_TEMPLATE.format(
        taxonomy=", ".join(TAXONOMY),
        tool_name=tool_name,
        tool_desc=tool_desc
    )

    url = f"{API_BASE}/api/chat/completions"
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}]
        # "stream": True   # uncomment to stream tokens instead of getting one big reply
    }

    print(requests.post(url ,headers=HEADERS,json=payload).json())

    try:
        resp = requests.post(url, headers=HEADERS, json=payload)
        resp.raise_for_status()

    except requests.RequestException as e:
        print("Request failed:", e)
        print("Status code:", getattr(e.response, "status_code", None))
        print("Response body:", getattr(e.response, "text", None))
        sys.exit(1)

    print("Status code:", resp.status_code)
    print("Response body:", resp.text)
    # Extract and clean up the model’s answer
    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    clean = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
    # Ensure it matches one of our known categories (fallback to "Unknown")

    return clean
    #return clean if clean in TARGET_LIST else "Unknown"