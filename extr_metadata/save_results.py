import json
from copy import deepcopy

def save_all_results(results, bert_preds, llm_preds, path="results.json"):

    print("BERT_preds:", bert_preds)
    print("LLM_preds:", llm_preds)


    merged = {}

    if isinstance(bert_preds, list):
        # If each element is itself a list (categories) — keep it as-is
        if all(isinstance(x, list) for x in bert_preds):
            bert_dict = {tool: bert_preds[i] for i, tool in enumerate(results.keys()) if i < len(bert_preds)}
        else:
            bert_dict = {tool: [bert_preds[i]] for i, tool in enumerate(results.keys()) if i < len(bert_preds)}
    else:
        bert_dict = bert_preds or {}

    if isinstance(llm_preds, list):
        if all(isinstance(x, list) for x in llm_preds):
            llm_dict = {tool: llm_preds[i] for i, tool in enumerate(results.keys()) if i < len(llm_preds)}
        else:
            llm_dict = {tool: [llm_preds[i]] for i, tool in enumerate(results.keys()) if i < len(llm_preds)}
    else:
        llm_dict = llm_preds or {}

    # Merge results
    for tool, meta in results.items():
        entry = deepcopy(meta)
        entry["bert_prediction"] = bert_dict.get(tool, [])
        entry["llm_prediction"] = llm_dict.get(tool, [])
        merged[tool] = entry

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to {path}")



