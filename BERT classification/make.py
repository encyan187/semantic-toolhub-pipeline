from new_config import DEVICE, TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH, DATASET_PATH, AUGMENT_DATASET_PATH
from new_model import SemanticBERT
from new_data_loader import load_data_loader
import torch
from transformers import get_linear_schedule_with_warmup


def make(config):

    #train_loader = load_data_loader(TRAIN_DATASET_PATH, shuffle=True ,batch_size=config.batch_size)

    #val_loader = load_data_loader(VAL_DATASET_PATH, shuffle=False, batch_size= 4)

    data_loader = load_data_loader(AUGMENT_DATASET_PATH, shuffle=True, batch_size=config.batch_size)

    test_loader = load_data_loader(TEST_DATASET_PATH, shuffle=False, batch_size= 4)

    total_steps = len(data_loader) * config.epochs
    warmup_steps = int(0.1 * total_steps)

    model = SemanticBERT().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return data_loader ,test_loader, model, optimizer, scheduler
