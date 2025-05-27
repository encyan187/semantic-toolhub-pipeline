import json
from tqdm import tqdm
import pandas as pd
from extr_metadata.github_api import search_github_repo, get_readme, get_repo_metadata



def main():

    path_in = "/Users/finnseemann/Documents/Studium/Bachelor_Thesis/Semantic_Toolhub_Pipeline/data/Data_MetaExtr.csv"

    df = pd.read_csv(path_in)
    tool_names = df['Name'].dropna().unique().tolist()

    readme_results = {}
    meta_results = {}
    for tool in tqdm(tool_names[:1]):
        repo = search_github_repo(tool)
        if repo:
            readme = get_readme(repo)

            metadata = get_repo_metadata(repo)

            if readme:
                readme_results[tool] = {
                    "repo": repo,
                    "readme": readme, }

            if metadata:
                meta_results[tool] = {
                    "url": metadata.get("url"),
                    "language": metadata.get("language"),
                    "last_update": metadata.get("updated_at"),
                    "license": metadata.get("license"),
                    "description": metadata.get("description"),
                    "keywords": metadata.get("topics")
                }

    path_readme = "/extr_metadata/outputs/readme.json"
    path_meta = "/extr_metadata/outputs/meta.json"

    with open(path_readme, "w", encoding="utf-8") as f:
        json.dump(readme_results, f, ensure_ascii=False, indent=2)

    with open(path_meta, "w", encoding="utf-8") as f:
        json.dump(meta_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()