import os
import numpy as np
from tqdm import tqdm
from new_config import BEST_MODEL_PATH, CHECKPOINT_PATH
from new_utils import *
import wandb
from torch import nn
import torch.nn.functional as F

def train(train_loader, val_loader, model, optimizer, pos_weight, scheduler, config):

    def loss_func(outputs, targets):
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(outputs, targets)

    def focal_loss(logits, labels, gamma=2.0, alpha=0.25):
        # logits, labels: [batch_size, num_labels]
        probas = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        p_t = probas * labels + (1 - probas) * (1 - labels)
        modulator = alpha * (1 - p_t).pow(gamma)
        return (modulator * ce_loss).mean()

    wandb.watch(model, log='all', log_freq=10)

    #valid_loss_min = np.inf
    best_f1_micro = 0.0
    best_model_save_path = None
    checkpoint = None

    for epoch in range(1, config.epochs + 1):

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
            loss = focal_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss = train_loss + (1 / (index + 1) * (loss.item() - train_loss))

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

                loss = focal_loss(outputs, targets)
                outputs = torch.sigmoid(outputs)

                if index == 0:
                    probs = torch.sigmoid(outputs)
                    threshold = 0.6
                    preds = (probs > threshold).int()
                    print("\n=== VALIDATION DEBUG ===")
                    print("Sigmoid probs:", probs[0].detach().cpu().numpy())
                    print("Predicted labels:", preds[0].cpu().numpy())
                    print("True labels:", targets[0].cpu().numpy())

                val_loss = val_loss + (1 / (index + 1)) * (loss.item() - val_loss)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_targets).numpy()

            # Print Metrics for each Epoch
            f1_micro, f1_macro, precision, recall, report = compute_metrics(y_true, y_pred)

            print(f"Validation Loss: {val_loss:.4f} | f1_micro: {f1_micro:.4f} | precision: {precision:.4f} | recall: {recall:.4f}")

            print(report)

            wandb.log({"val_loss": val_loss,
                       "train_loss": train_loss,
                       "f1_micro": f1_micro,
                       "f1_macro": f1_macro,
                       "precision": precision,
                       "recall": recall,
                       "epoch": epoch})

        # Save best Model measured by best f1 micro score


        if f1_micro > best_f1_micro:
            best_f1_micro = f1_micro

            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

    if checkpoint:
        model_filename = f"{wandb.run.name}_f1_{best_f1_micro:.4f}.pt"
        best_model_save_path = os.path.join(BEST_MODEL_PATH, model_filename)
        save_checkpoint(checkpoint, True, CHECKPOINT_PATH, best_model_save_path)
        model.load_state_dict(torch.load(best_model_save_path)["state_dict"])

    return model