from make import *
from new_train import train
import wandb
from new_config import *
from new_split_dataset import split_dataset
from new_utils import *
from sweep_config import *
from test import *

def model_pipeline(hyperparameters):

    with wandb.init(project="test", config=hyperparameters):

        print("Starting model pipeline")

        config = wandb.config

        split_dataset(DATASET_PATH, TRAIN_SIZE)

        pos_weight = get_pos_weights(TRAIN_DATASET_PATH)

        train_loader, val_loader , model, optimizer, scheduler = make(config)

        print(model)

        trained_model = train(train_loader, val_loader , model, optimizer, pos_weight, scheduler,config)

        #print(test(trained_model, test_loader))

    return trained_model

model_pipeline(wandb.config)

