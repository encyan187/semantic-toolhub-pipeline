import requests

def search_crossref_papers(tool_name, rows=5):
    url = "https://api.crossref.org/works"
    params = {
        "query": tool_name,
        "rows": rows,
        "sort": "relevance"
    }
    response = requests.get(url, params=params)
    data = response.json()

    items = data.get("message", {}).get("items", [])
    results = []

    for item in items:
        results.append({
            "title": item.get("title", [""])[0],
            "authors": [author.get("family", "") for author in item.get("author", [])],
            "published": item.get("issued", {}).get("date-parts", [[None]])[0][0],
            "DOI": item.get("DOI"),
            "link": f"https://doi.org/{item.get('DOI')}"
        })
    
    return results