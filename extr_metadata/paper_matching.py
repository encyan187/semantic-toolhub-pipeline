import requests
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def find_best_matching_paper(tool_name, tool_description, crossref_results):

    best_match = None

    highest_score = -1

    desc_embedding = model.encode(tool_description, convert_to_tensor=True)

    for paper in crossref_results:

        title = paper.get("title", "")

        title_embedding = model.encode(title, convert_to_tensor=True)

        score = util.pytorch_cos_sim(desc_embedding, title_embedding).item()

        if tool_name.lower() in title.lower():
            score += 0.2

        keywords = ["software", "tool", "framework", "library", "platform", "we present", "we describe" , "semantic" ,
                    "ontology"]

        if any(k in title.lower() for k in keywords):
            score += 0.1

        if score > highest_score:
            highest_score = score
            best_match = {
                "title": title,
                "authors": paper.get("authors", []),
                "year": paper.get("published"),
                "DOI": paper.get("DOI"),
                "link": paper.get("link"),
                "score": round(score, 3)
            }
    return best_match