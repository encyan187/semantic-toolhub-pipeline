import json
from tqdm import tqdm
import pandas as pd
from extr_metadata.github_api import search_github_repo, get_readme, get_repo_metadata
from extr_metadata.wikidata import get_wikidata_info
from extr_metadata.crossref import search_crossref_papers
from extr_metadata.paper_matching import find_best_matching_paper

def run_pipeline(tool_names):

    readme_results = {}

    meta_results = {}

    for tool in tqdm(tool_names[:1]):

        meta = {}

        repo = search_github_repo(tool)

        if repo:

            readme = get_readme(repo)

            metadata = get_repo_metadata(repo)

            if readme:
                readme_results[tool] = {
                    "repo": repo,
                    "readme": readme, }

            if metadata:
                meta.update({
                    "github_url": metadata.get("url"),
                    "language": metadata.get("language"),
                    "last_update": metadata.get("updated_at"),
                    "license": metadata.get("license"),
                    "github_description": metadata.get("description"),
                    "keywords": metadata.get("topics")
                })

        wikidata = get_wikidata_info(tool)

        if wikidata:
            meta.update({
            "wikidata_url": wikidata["wikidata_url"],
            "wiki_description": wikidata["description"],
            "website": wikidata["official_website"],
            "related_papers": wikidata["papers"]
            })


            if not wikidata.get("papers"):

                crossref = search_crossref_papers(tool)

                if crossref and "description" in wikidata:

                    best_paper = find_best_matching_paper(tool , wikidata["description"] , crossref)

                    if best_paper:
                        meta.update( {
                            "related_papers": best_paper.get("title"),
                            "paper_link": best_paper.get("link")
                        })

    meta_results[tool] = meta

    # Save metadata results
    pd.DataFrame.from_dict(meta_results, orient="index") \
        .reset_index(names="tool") \
        .to_csv("outputs/meta.csv", index=False)

    # Save readme results -> input for classification
    pd.DataFrame.from_dict(readme_results, orient="index") \
        .reset_index(names="tool") \
        .to_csv("outputs/readme.csv", index=False)

    return meta_results




