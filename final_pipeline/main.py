import json

from tqdm import tqdm

from load_model import load_model
from create_input import create_input
from pipeline import run_pipeline
from bert_predict import predict
from llm_predict import classify_tool
from save_results import save_all_results
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv("Additional_Tool_Names.csv")

    #tool_name = data["Name"].tolist()

    token = ""

    tool_name = ["Protege", "Apache", "rdf2rml"]

    model, tokenizer = load_model()

    final_results = {}

    for tool in tqdm(tool_name):

        tool_info = {}

        results = run_pipeline([tool], token=token)

        if results:
            meta = results.get(tool) or next(iter(results.values()))
            tool_info.update(meta)
            #tool_info.update(results)

        print(results)

        input = create_input(results)

        print(input)

        bert_prediction = predict(input, model, tokenizer)

        if bert_prediction:
            tool_info["bert prediction"] = bert_prediction

        print(bert_prediction)

        llm_prediction = classify_tool(tool_name,input)

        if llm_prediction:
            tool_info["LLM prediction"] = llm_prediction
        print(llm_prediction)


        final_results[tool] = tool_info

        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

    print("Saved to results.json")



