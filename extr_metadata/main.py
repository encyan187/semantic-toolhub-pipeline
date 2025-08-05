import pandas as pd

from extr_metadata.create_input import create_input
from extr_metadata.pipeline import run_pipeline

if __name__ == "__main__":

    #data = pd.read_csv("Additional_Tool_Names.csv")

    #tool_name = data["Name"].tolist()

    token = ""

    tool_name = ["Protege"]

    results = run_pipeline(tool_name, token=token)

    print(results)

    bert_input = create_input(results)

    print(bert_input)

