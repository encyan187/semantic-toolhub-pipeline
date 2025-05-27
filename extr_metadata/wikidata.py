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
    # Step 1: Search for the entity
    search_url = "https://www.wikidata.org/w/api.php"
    search_params = {
        "action": "wbsearchentities",
        "search": tool_name,
        "language": "en",
        "format": "json"
    }
    search_resp = requests.get(search_url, params=search_params).json()
    search_results = search_resp.get("search", [])
    
    if not search_results:
        return {"error": "No matching Wikidata entity found."}
    
    # Take the top result
    entity_id = search_results[0]["id"]
    label = search_results[0]["label"]
    description = search_results[0].get("description", "")
    
    # Step 2: Get full entity data
    entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    entity_data = requests.get(entity_url).json()
    claims = entity_data["entities"][entity_id]["claims"]

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
