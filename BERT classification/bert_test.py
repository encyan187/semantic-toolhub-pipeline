import torch
from new_utils import compute_metrics
from new_config import DEVICE, TEST_DATASET_PATH,TARGET_LIST, VAL_DATASET_PATH
from new_model import SemanticBERT
from new_data_loader import load_data_loader
from new_utils import save_eval


def final_eval(model, dataloader):

    model.eval()
    THRESHOLD = 0.50
    idx = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(DEVICE, dtype=torch.long)
            attention_mask = batch["attention_mask"].to(DEVICE, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(DEVICE, dtype=torch.long)

            targets = batch["targets"].to(DEVICE, dtype=torch.float)
            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.sigmoid(logits)

            pred_bin = (probs >= THRESHOLD)  # bool
            true_bin = (targets >= 0.5)  # bool (robust if float labels)

            for b in range(probs.size(0)):
                true_idx = torch.where(true_bin[b])[0].tolist()
                pred_idx = torch.where(pred_bin[b])[0].tolist()

                if TARGET_LIST and len(TARGET_LIST) == probs.size(1):
                    true_names = [TARGET_LIST[j] for j in true_idx]
                    pred_names = [TARGET_LIST[j] for j in pred_idx]
                else:
                    true_names = true_idx
                    pred_names = pred_idx

                print(f"{idx:04d} true={true_names or '∅'}  pred={pred_names or '∅'}")
                idx += 1

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()

    # Compute metrics
    f1_micro, f1_macro, precision, recall, report = compute_metrics(y_true, y_pred)

    metrics = {"f1_micro" : f1_micro,
               "f1_macro" : f1_macro,
               "precision" : precision,
               "recall" : recall,}


    save_eval(metrics,report, run_name="test")

    return f1_micro, f1_macro, precision, recall, report
