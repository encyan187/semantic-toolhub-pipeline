"""bert_project/
│
├── new_config.py                # All hyperparameters & paths
├── train.py                 # Main training loop
├── evaluate.py              # Evaluation logic (on val/test)
├── model.py                 # BERT model class wrapper
├── dataset.py               # Dataset & DataLoader prep
├── utils.py
----make(config)
---- train(model, train_loader, val_vloader, criterion, optimizer, config)                                ----# Utility functions (metrics, logging)
│
├── run.sh                   # (Optional) Shell script to run training
├── requirements.txt         # Python dependencies
"""
