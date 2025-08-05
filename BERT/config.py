import torch
from transformers import BertTokenizer
import numpy as np

DATASET_PATH = 'data/new_final_dataset.csv'
TRAIN_DATASET_PATH = 'data/train.csv'
TEST_DATASET_PATH = 'data/test.csv'
VAL_DATASET_PATH = 'data/val.csv'
CHECKPOINT_PATH = "checkpoints/checkpoint.pt"
BEST_MODEL_PATH = "checkpoints/best_model.pt"


TRAIN_SIZE = 0.8
VAL_SIZE = 0.2
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 4
NUM_EPOCHS = 1
LEARNING_RATE = 0.07988527061600488
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")


TEXT_COLUMN = 'tool_description'
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