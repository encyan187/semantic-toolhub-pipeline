import time
from tqdm import tqdm
import pandas as pd
from extr_metadata.github_api import search_github_repo, get_readme, get_repo_metadata, clean_readme
from extr_metadata.wikidata import get_wikidata_info
from extr_metadata.crossref import search_crossref_papers
from extr_metadata.paper_matching import find_best_matching_paper

def run_pipeline(tool_names , token = None):

    meta_results = {}

    for tool in tqdm(tool_names):

        meta = {}

        repo = search_github_repo(tool, token=token)

        if repo:

            metadata = get_repo_metadata(repo, token=token)

            readme = get_readme(repo, token=token)

            if readme:
                readme = clean_readme(readme)

            if metadata:
                meta.update({
                    "readme": readme,
                    "repo": repo,
                    "github_url": metadata.get("url" , None),
                    "language": metadata.get("language" , None),
                    "last_update": metadata.get("updated_at" , None),
                    "license": metadata.get("license" , None),
                    "github_description": metadata.get("description" , None),
                    "keywords": metadata.get("topics", None)
                })

        wikidata = get_wikidata_info(tool)
        #wikidata = None

        if wikidata:
            meta.update({
            "wikidata_url": wikidata.get("wikidata_url", None),
            "wiki_description": wikidata.get("description", None),
            "website": wikidata.get("official_website", None),
            "related_papers": wikidata.get("papers", None),
            })


            if not wikidata.get("papers"):

                crossref = search_crossref_papers(tool)

                if crossref and "description" in wikidata:

                    best_paper = find_best_matching_paper(tool , wikidata["description"] , crossref , metadata.get("topics", None))

                    if best_paper:
                        meta.update( {
                            "related_papers": best_paper.get("title"),
                            "paper_link": best_paper.get("link")
                        })

        meta_results[tool] = meta

        time.sleep(5)

    # Save metadata results
    #pd.DataFrame.from_dict(meta_results, orient="index") \
        #.reset_index(names="tool") \
        #.to_csv("outputs/additional_meta_2.csv", index=False)

    return meta_results




