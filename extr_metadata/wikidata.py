import requests

def search_wikidata_entity(tool_name):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": tool_name,
        "language": "en",
        "format": "json"
    }
    response = requests.get(url, params=params)
    results = response.json().get("search", [])
    return results  # list of dicts


def get_wikidata_info(tool_name):
    # Polite headers (required by WMF UA policy) + ask for JSON
    headers = {
        "Accept": "application/json",
        "User-Agent": "SemanticToolhubClassifier/1.0 (student project; contact: youremail@example.com)"
    }

    with requests.Session() as sess:
        # ---- Step 1: Search for the entity ----
        search_url = "https://www.wikidata.org/w/api.php"
        search_params = {
            "action": "wbsearchentities",
            "search": tool_name,
            "language": "en",
            "format": "json",
            "limit": 1
        }
        r = sess.get(search_url, params=search_params, headers=headers, timeout=20)
        r.raise_for_status()

        # Guard against empty/non-JSON
        text = r.text.strip()
        if not text:
            return {"error": "Empty search response from Wikidata."}
        if "application/json" not in (r.headers.get("Content-Type") or "") and not text.startswith(("{", "[")):
            return {"error": f"Search returned non-JSON (Content-Type={r.headers.get('Content-Type')})."}

        try:
            search_resp = r.json()
        except requests.exceptions.JSONDecodeError:
            return {"error": f"Search JSON decode failed. Head: {text[:200]!r}"}

        search_results = search_resp.get("search", []) or []
        if not search_results:
            return {"error": "No matching Wikidata entity found."}

        # Take the top result
        entity_id = search_results[0]["id"]
        label = search_results[0].get("label", "")
        description = search_results[0].get("description", "")

        # ---- Step 2: Get full entity data ----
        # Try Special:EntityData first (your original approach)
        entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
        r2 = sess.get(entity_url, headers=headers, timeout=20)

        # If HTTP error, surface it now (still capture body for context)
        try:
            r2.raise_for_status()
        except requests.HTTPError:
            head = (r2.text or "")[:200]
            return {"error": f"Entity fetch HTTP error {r2.status_code}. Head: {head!r}"}

        text2 = (r2.text or "").strip()
        ctype2 = r2.headers.get("Content-Type") or ""

        entity_data = None
        # Accept JSON, or fallback if HTML/empty
        if text2 and ("application/json" in ctype2 or text2.startswith(("{", "["))):
            try:
                entity_data = r2.json()
            except requests.exceptions.JSONDecodeError:
                # fall back to wbgetentities below
                entity_data = None

        if entity_data is None:
            # Fallback to the API (more robust, avoids HTML)
            entity_params = {
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "labels|descriptions|aliases|claims",
                "languages": "en",
                "format": "json",
            }
            r3 = sess.get("https://www.wikidata.org/w/api.php",
                          params=entity_params, headers=headers, timeout=20)
            try:
                r3.raise_for_status()
            except requests.HTTPError:
                head = (r3.text or "")[:200]
                return {"error": f"wbgetentities HTTP error {r3.status_code}. Head: {head!r}"}

            t3 = (r3.text or "").strip()
            if not t3:
                return {"error": "Empty wbgetentities response."}
            if "application/json" not in (r3.headers.get("Content-Type") or "") and not t3.startswith(("{", "[")):
                return {"error": f"wbgetentities returned non-JSON (Content-Type={r3.headers.get('Content-Type')})."}

            try:
                entity_data = r3.json()
            except requests.exceptions.JSONDecodeError:
                return {"error": f"wbgetentities JSON decode failed. Head: {t3[:200]!r}"}

        # Extract claims (and keep graceful fallbacks for label/description)
        try:
            ent = entity_data["entities"][entity_id]
        except Exception:
            head = str(entity_data)[:200]
            return {"error": f"Entity {entity_id} missing in response. Head: {head!r}"}

        claims = ent.get("claims", {})

    # Step 3: Extract website and papers
    def get_claim_url(property_id):
        values = claims.get(property_id)
        if values:
            mainsnak = values[0]["mainsnak"]
            if mainsnak.get("datavalue"):
                return mainsnak["datavalue"]["value"]
        return None

    # P856 = official website
    website = get_claim_url("P856")
    
    # P2860 = "cites work", P5326 = "described by source"
    paper_claims = claims.get("P2860", []) + claims.get("P5326", [])
    paper_urls = []

    for paper in paper_claims:
        if "datavalue" in paper["mainsnak"]:
            paper_id = paper["mainsnak"]["datavalue"]["value"]["id"]
            paper_urls.append(f"https://www.wikidata.org/wiki/{paper_id}")

    return {
        "label": label,
        "description": description,
        "wikidata_id": entity_id,
        "wikidata_url": f"https://www.wikidata.org/wiki/{entity_id}",
        "official_website": website,
        "papers": paper_urls
    }
