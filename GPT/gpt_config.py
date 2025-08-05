API_BASE    = "https://llama-max.ai.wu.ac.at"

MODEL_ID    = "deepseek-r1:latest"

API_KEY     = "sk-456ff07dfe994cbc97d9cf0a5d73aeb7"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

TAXONOMY = ["Class: Knowledge Graph Requirement Elicitation",
            "Class: Ontology Modelling", "Subclass: Visual Ontology Editing", "Subclass: Ontology Visualization" ,
            "Class: Ontology Evaluation", "Subclass: Numeric Ontology Analysis", "Subclass: Ontology Validation",
            "Class: Knowledge Graph Population", "Subclass: Virtual Knowledge Graph", "Subclass: Knowledge Graph Materialization",
            "Class: Knowledge Graph Cleaning & Evaluation & Validation", "Subclass: Schema based Knowledge Graph validation", "Subclass: Heuristic based Knowledge Graph validation",
            "Class: Knowledge Graph Querying", "Subclass: Knowledge Graph Query Engine", "Subsubclass: Federated Knowledge Graph Querying", "Subclass: Sparql Query Builder", "Subclass: Sparql Query Result Visualizer",
            "Class: Knowledge Graph Reasoning", "Subclass: Standalone Knowledge Graph Reasoner", "Subclass: Knowledge Graph API Engine with reasoner", "Subclass: Triplestore with reasoner",
            "Class: Knowledge Graph Storage" , "Subclass: Pure Triplestore", "Subclass: Multi-Model Database",
            "Class: Knowledge Graph Publication", "Subclass: Ontology Publication", "Subclass: Knowledge Graph Browser",
            "Class: Knowledge Graph Maintenance", "Subclass: Issue Tracking", "Subclass: Version Management", "Subsubclass: Ontology versioning management", "Subsubclass: Knowledge Graph version management" , "Subclass: Semantic Artefact Catalog ",
            "Class: RDF API / Library"
            "Class: Others" , "Subclass: Streaming Data" , "Subclass: Logic Solver" , "Subclass: Knowledge Graph Learning Materials"
            ]


TARGET_LIST = ['Federated Knowledge Graph Querying',
       'Heuristic based Knowledge Graph validation', 'Issue Tracking',
       'Knowledge Graph API Engine with reasoner', 'Knowledge Graph Browser',
       'Knowledge Graph Cleaning & Evaluation & Validation',
       'Knowledge Graph Maintenance', 'Knowledge Graph Materialization',
       'Knowledge Graph Population', 'Knowledge Graph Publication',
       'Knowledge Graph Query Engine', 'Knowledge Graph Querying',
       'Knowledge Graph Reasoning', 'Knowledge Graph Requirement Elicitation',
       'Knowledge Graph Storage', 'Knowledge Graph version management',
       'Logic Solver', 'Multi-Model Database', 'Numeric Ontology Analysis',
       'Ontology Modelling', 'Ontology Publication', 'Ontology Validation',
       'Ontology Visualization', 'Ontology versioning management',
       'Pure Triplestore', 'RDF API / Library',
       'Schema based Knowledge Graph validation', 'Semantic Artefact Catalog',
       'Sparql Query Builder', 'Sparql Query Result Visualizer',
       'Standalone Knowledge Graph Reasoner', 'Streaming Data',
       'Triplestore with reasoner', 'Version Management',
       'Virtual Knowledge Graph', 'Visual Ontology Editing']

PROMPT_TEMPLATE = """
You are a software taxonomy assistant. Classify each software tool into one or more of these taxonomy categories: {taxonomy}. 
For this keep in mind the structure of taxonomies and that each tool can belong to multiple classes and subclasses. 
If a tool belongs to a subclass of a class only categorize the subclasses if it belongs to one, not the parent classes.

Examples:
1. Tool: Apache Jena
   Description: apache, jena, rdf, sparql Apache Jena (or Jena in short) is a free and open source Java framework for building semantic web and Linked Data applications. The framework is composed of different APIs interacting together to process RDF data, including APIs for direct RDF data manipulation, SPARQL querying with support for federated queries and free text search, persisting data, OWL, inference and rules. jena readme welcome to apache jena a java framework for writing semantic web applications see for the project website including documentation the codebase for the active modules is in git Apache Jena, A free and open source Java framework for building Semantic Web and Linked Data applications. open source semantic web framework for Java
   Answer: [Ontology Modelling, Schema based Knowledge Graph validation, Knowledge Graph Query Engine, Knowledge Graph API Engine with reasoner, Knowledge Graph Storage, RDF API / Library]

2. Tool: Stardog
   Description: Stardog is a commercial RDF database with support for SPARQL querying and OWL reasoning. It supports multiple reasoning profiles, namely RDFS and OWL2 QL, EL, RL, DL. Besides the core functionality of a triplestore, Stardog offers two graphical user interface solutions with Stardog studio and Stardog explorer. Studio makes it possible to easily manage different repositories in a Stardog database, and it provides basic tools to explore data in those repositories. Stardog explorer is a dedicated search engine on-top of a Stardog database. pellet an open source owl dl reasoner for java gitter pellet is the owl dl reasoner open source agpl or commercial license pure java developed and commercially supported by complexible inc pellet can be used with jena or owlapi libraries pellet provides functionality to check consistency of ontologies compute the classification hierarchy explain inferences and answer sparql queries pellet a closed source nextgen version of pellet is embedded and available in stardog the rdf database feel free to fork this repository and submit pull requests if you want to see changes new features etc in pellet documentation about how to use pellet is in the doc directory and there are some code samples in the examples directory commercial support for pellet is available the pellet faq answers some frequently asked questions there is a pelletusers mailing list for questions and feedback you can search pelletusers archives bug reports and enhancement requests should be sent to the mailing list issues are on github thanks for using pellet Pellet is an OWL 2 reasoner in Java; open source (AGPL) and commercially licensed, commercial support available.
   Answer: [Visual Ontology Editing, Virtual Knowledge Graph, Knowledge Graph Materialization, Knowledge Graph Query Engine, Triplestore with reasoner, Multi-Model Database, Knowledge Graph Browser, Version Management]

3. Tool: WebVOWL
   Description: javascript, owl, rdf, rdfs, svg, visualization, webvowl  webvowl build status caution the url is not owned bei visualdataweb anymore not related to webvowl anymore this repository was ported from an internal svn repository to github after the release of webvowl due to cleanups with the commit history might show some strange effects run using docker make sure you are inside directory and you have docker installed run the following command to build the docker image run the following command to run webvowl at port visit to use webvowl requirements nodejs for installing the development tools and dependencies development setup simple download and install nodejs from open the terminal in the root directory run to install the dependencies and build the project edit the code run to rebuild all necessary files into the deploy directory run to run the server locally by installing serve by using visit to use webvowl advanced instead of the last step of the simple setup install the npm package globally with now you can execute a few more advanced commands in the terminal or builds the release files into the deploy directory builds the development version starts a local liveupdating webserver with the current development version starts the test runner builds the project and puts it into a zip file additional information to export the vowl visualization to an svg image all css styles have to be included into the svg code this means that if you change the css code in the file you also have to update the code that inlines the styles otherwise the exported svg will not look the same as the displayed graph the tool which creates the code that inlines the styles can be found in the util directory please follow the instructions in its readmeutilvowlcsstodruleconverterreadmemd file Visualizing ontologies on the Web web tool for visualizing ontologies
   Answer: [Visual Ontology Editing, Ontology Visualization, Ontology Publication, Knowledge Graph Browser]

Your response should only be the Categories in the following format: [categorie 1, categorie 2,...], nothing else. 
Do NOT write any sentences as answers. I want only the categories as output. Very important. 

Now classify the following:

Tool: {tool_name}
Description: {tool_desc}
Categories:""".lstrip()