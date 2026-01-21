import json
import torch

#BERT Prediction

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR  = 'BERT_no_readme.pt'
THRESHOLD = 0.6
MAX_LEN = 128


TARGET_LIST = ['Federated Knowledge Graph Querying',
       'Heuristic based Knowledge Graph validation', 'Issue Tracking',
       'Knowledge Graph API Engine with reasoner', 'Knowledge Graph Browser',
       'Knowledge Graph Cleaning & Evaluation & Validation',
       'Knowledge Graph Maintenance', 'Knowledge Graph Materialization',
       'Knowledge Graph Population', 'Knowledge Graph Publication',
       'Knowledge Graph Query Engine', 'Knowledge Graph Querying',
       'Knowledge Graph Reasoning ', 'Knowledge Graph Requirement Elicitation',
       'Knowledge Graph Storage', 'Knowledge Graph version management',
       'Logic Solver', 'Multi-Model Database', 'Numeric Ontology Analysis',
       'Ontology Modelling', 'Ontology Publication', 'Ontology Validation',
       'Ontology Visualization', 'Ontology versioning management',
       'Pure Triplestore', 'RDF API / Library',
       'Schema based Knowledge Graph validation', 'Semantic Artefact Catalog ',
       'Sparql Query Builder', 'Sparql Query Result Visualizer',
       'Standalone Knowledge Graph Reasoner', 'Streaming Data',
       'Triplestore with reasoner', 'Version Management',
       'Virtual Knowledge Graph', 'Visual Ontology Editing']

# LLM Prediction

ALLOWED_JSON = json.dumps(TARGET_LIST, ensure_ascii=False)

SYSTEM_MSG = ("You are a strict multilabel classifier for semantic software tools. "
              f"AllowedCategories = {json.dumps(ALLOWED_JSON)}\n"
              "- Choose zero or more categories ONLY from AllowedCategories.\n"
              "- Use exact spelling (case-sensitive). Do not invent labels.\n"
              "- Prefer specific subclasses; avoid parent+child together.\n"
              "- If none apply, return [].\n"

              "- Category Guidelines (apply only where listed; otherwise use general rules):\n"
              "Knowledge Graph Requirement Elicitation: -Tools generating or collecting competency questions including document templates.\n"
              "Ontology Visualization: -Tools automatically creating ontology graph visualizations.\n"
              "Ontology Evaluation: Analyze ontology schema.\n"
              "Knowledge Graph Population: -Tools concerning R2RML, Interfaces for data ingestion.\n"
              "Knowledge Graph Cleaning, Evaluation & Validation: -Tools that provide tooling for schema or data validation.\n"
              "Schema based Knowledge Graph validation: -Tools using SHACL, SHEX.\n"
              "Heuristic based Knowledge Graph validation: -Tool using subsymbolic approaches to detect incorrect data.\n"
              "Federated Knowledge Graph Querying: -Tools for working explicitly with multiple data sources.\n"
              "Sparql Query Builder: -Tools for low-code query generation.\n"
              "Sparql Query Result Visualizer: -Visualization of query results, e.g., graphs, diagrams. - Interactive query building.\n"
              "Knowledge Graph Reasoning: - Plugins for Editors, Databases.\n"
              "Knowledge Graph Storage: - Tools for storing RDF-based data. - Must have SPARQL Endpoint.\n"
              "Ontology Publication: - Documentation of ontology Schema. - Creation of HTML.\n"
              "Issue Tracking: - Problem / Incidentmanagement for ontologies. - Tracking user request and changes.\n"
              "RDF API / Library: -Programmatically alter / access RDF data.\n"
              "Others: -Highly specific use cases for semantic web tools that do not fall under the previous category.\n"

              "- Identify key signals (e.g., RDF, SPARQL, OWL, triplestore, GUI, visualization, analysis)\n"
              "- Output ONLY a valid JSON array of strings. No explanations."
              )

PROMPT_TEMPLATE_FEW_SHOT = """

Examples:

Tool: Apache Jena
Description: apache, jena, rdf, sparql Apache Jena (or Jena in short) is a free and open source Java framework for building semantic web and Linked Data applications. The framework is composed of different APIs interacting together to process RDF data, including APIs for direct RDF data manipulation, SPARQL querying with support for federated queries and free text search, persisting data, OWL, inference and rules. jena readme welcome to apache jena a java framework for writing semantic web applications see for the project website including documentation the codebase for the active modules is in git Apache Jena, A free and open source Java framework for building Semantic Web and Linked Data applications. open source semantic web framework for Java
Answer: [Ontology Modelling, Schema based Knowledge Graph validation, Knowledge Graph Query Engine, Knowledge Graph API Engine with reasoner, Knowledge Graph Storage, RDF API / Library]

Tool: Stardog
Description: Stardog is a commercial RDF database with support for SPARQL querying and OWL reasoning. It supports multiple reasoning profiles, namely RDFS and OWL2 QL, EL, RL, DL. Besides the core functionality of a triplestore, Stardog offers two graphical user interface solutions with Stardog studio and Stardog explorer. Studio makes it possible to easily manage different repositories in a Stardog database, and it provides basic tools to explore data in those repositories. Stardog explorer is a dedicated search engine on-top of a Stardog database. pellet an open source owl dl reasoner for java gitter pellet is the owl dl reasoner open source agpl or commercial license pure java developed and commercially supported by complexible inc pellet can be used with jena or owlapi libraries pellet provides functionality to check consistency of ontologies compute the classification hierarchy explain inferences and answer sparql queries pellet a closed source nextgen version of pellet is embedded and available in stardog the rdf database feel free to fork this repository and submit pull requests if you want to see changes new features etc in pellet documentation about how to use pellet is in the doc directory and there are some code samples in the examples directory commercial support for pellet is available the pellet faq answers some frequently asked questions there is a pelletusers mailing list for questions and feedback you can search pelletusers archives bug reports and enhancement requests should be sent to the mailing list issues are on github thanks for using pellet Pellet is an OWL 2 reasoner in Java; open source (AGPL) and commercially licensed, commercial support available.
Answer: [Visual Ontology Editing, Virtual Knowledge Graph, Knowledge Graph Materialization, Knowledge Graph Query Engine, Triplestore with reasoner, Multi-Model Database, Knowledge Graph Browser, Version Management]

In the following format: [categorie 1, categorie 2,...]. Keep in mind to only predict categories from the taxonomy.
Make sure you keep this format. Do NOT write any sentences as answers. Only the categories as output.

What categories does the following tool belong to?

Tool: {tool_name}
Description: {tool_desc}
Categories: """.lstrip()

API_BASE    = "https://llama-max.ai.wu.ac.at"

MODEL_ID    = "llama3.1:latest"

API_KEY     = ""

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}