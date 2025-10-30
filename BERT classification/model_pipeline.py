from make import *
from new_train import train
import wandb
from augment_dataset import *
from new_utils import *
from hyperparameter_config import *
from bert_test import final_eval

def model_pipeline(hyperparameters):

    with wandb.init(project="test", config=hyperparameters):

        print("Starting model pipeline")

        config = wandb.config

        augment_dataset(DATASET_PATH)

        pos_weight = get_pos_weights(AUGMENT_DATASET_PATH)

        print(pos_weight)

        data_loader , test_loader,model, optimizer, scheduler = make(config)

        print(model)

        trained_model = train(data_loader , model, optimizer, pos_weight, scheduler,config)

        print(final_eval(trained_model, test_loader))

    return trained_model


model_pipeline(hyperparameter_config)

