from config import CHECKPOINT_PATH


class EarlyStopping:
    def __init__(self, patience, min_delta=0.01, path = CHECKPOINT_PATH, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement = 0

    def check_early_stopping(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.no_improvement = 0

        else:
            self.no_improvement += 1