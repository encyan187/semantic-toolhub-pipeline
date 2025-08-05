import ast
from classify_tool import *
import pandas as pd

data = pd.read_csv("gpt_final_dataset.csv", converters={"true_labels": ast.literal_eval})

tool_name = data["tool"]
tool_description = data["tool_description"]

test = "knowledge-graph, knowledge-management, obo, ontology, ontology-engineering, owl, protege, protege-desktop, rdf, reasoning, semantics A free, open-source ontology editor and framework for building intelligent systems. protege desktop protege is a free opensource ontology editor that supports the latest owl standard protege has a pluggable architecture and many plugins for different functionalities are available to read more about proteges features please visit the protege home page the latest version of protege can be downloaded from the protege website or from github if you would like to contribute to the protege project please see our contributing guide the developer documentation may be found on the wiki looking for support please ask questions on the protegeuser or protegedev mailing lists if you found a bug or would like to request a feature you may also use this issue tracker protege is released under the bsd clause license instructions for building from source are available on the the wiki Protege Desktop Ontology editor"

category = classify_tool("protege", test)

print(category)