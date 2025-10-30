import random
import re
import time
from gpt_config import *
import requests
import sys

def classify_tool(tool_name, tool_desc):

    RPM = 5
    SLEEP = 60/RPM

    system = SYSTEM_MSG.format(ALLOWED_JSON=ALLOWED_JSON)


    prompt = PROMPT_TEMPLATE_FEW_SHOT.format(
        ALLOWED_JSON=ALLOWED_JSON,
        tool_name=tool_name,
        tool_desc=tool_desc
    )

    url = f"{API_BASE}/api/chat/completions"

    #url = API_BASE

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role":"system", "content": system},
            {"role": "user", "content": prompt}
        ],

        "temperature": 0}

    #print(requests.post(url ,headers=HEADERS,json=payload).json())

    MAX_ATTEMPTS = 6
    BASE_BACKOFF = 1.0  # seconds
    TIMEOUT = 30  # seconds

    resp = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            break  # success
        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None)
            body = getattr(e.response, "text", None)
            print(f"Request failed (attempt {attempt}/{MAX_ATTEMPTS}): {e}")
            print("Status code:", status)
            print("Response body:", body)

            # non-retryable 4xx (except a few transient ones)
            if status and 400 <= status < 500 and status not in (408, 409, 425, 429):
                sys.exit(1)

            if attempt == MAX_ATTEMPTS:
                sys.exit(1)

            # honor Retry-After if present
            retry_after = None
            try:
                if getattr(e, "response", None):
                    ra = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
                    if ra is not None:
                        retry_after = float(ra)
            except Exception:
                retry_after = None

            # exponential backoff with jitter, capped
            sleep_for = retry_after if retry_after is not None else min(BASE_BACKOFF * (2 ** (attempt - 1)), 60)
            sleep_for += random.uniform(0, 0.25 * sleep_for)
            time.sleep(sleep_for)

    print("Status code:", resp.status_code)
    print("Response body:", resp.text)

    # Extract and clean up the model’s answer
    data = resp.json()
    reply = data["choices"][0]["message"]["content"]
    clean = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()

    #time.sleep(SLEEP)

    return clean
    #return clean if clean in TARGET_LIST else "Unknown"