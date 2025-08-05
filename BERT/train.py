from tqdm import tqdm
from BERT.model import SemanticBERT
from config import *
import numpy as np
from utils import *
from evaluate import test
from data_loader import load_data_loader
from transformers import get_linear_schedule_with_warmup



def train_model(model):

    train_loader = load_data_loader(TRAIN_DATASET_PATH, shuffle=True)

    val_loader = load_data_loader(VAL_DATASET_PATH, shuffle=False)

    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)

    model = model().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    valid_loss_min = np.inf
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = 0
        val_loss = 0
        model.train()

        # training loop

        for index, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            input_ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(DEVICE, dtype=torch.long)
            targets = batch["targets"].to(DEVICE, dtype=torch.float)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_func(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss = train_loss + (1/(index+1) * (loss.item() - train_loss))

        # validation loop

        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                input_ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
                attention_mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
                token_type_ids = batch["token_type_ids"].to(DEVICE, dtype=torch.long)

                targets = batch["targets"].to(DEVICE, dtype=torch.float)
                outputs = model(input_ids, attention_mask, token_type_ids)
                outputs = torch.sigmoid(outputs)

                loss = loss_func(outputs, targets)
                val_loss = val_loss + (1/(index+1)) * (loss.item() - val_loss)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_targets).numpy()

            # Print Metrics for each Epoch
            metrics = compute_metrics(y_true, y_pred)
            print(f"Validation Loss: {val_loss:.4f} | {metrics}")

            #Save best Model measured by validation loss
            is_best = val_loss < valid_loss_min
            if is_best:
                valid_loss_min = val_loss

        checkpoint = {
            'epoch': epoch+1,
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_checkpoint(checkpoint, is_best, CHECKPOINT_PATH, BEST_MODEL_PATH)

    model.load_state_dict(torch.load(BEST_MODEL_PATH)["state_dict"])

    return model


if __name__ == '__main__':

    best_model = train_model(SemanticBERT)

    test_loader = load_data_loader(TEST_DATASET_PATH, shuffle=False)

    test(best_model, test_loader)