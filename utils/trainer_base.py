import os
import torch
from torch import nn
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from .metrics import calculate_metrics

class BaseTrainer:
    def __init__(self, model, train_loader, valid_loader, test_loader,
                 writer, save_dir, patience=3, min_delta=0.001,
                 validate_every=10):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.writer = writer
        self.save_dir = save_dir
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.validate_every = validate_every
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model = None
        self.best_step = 0
        self.start_time = time.time()

    def save_checkpoint(self, state, filename):
        torch.save(state, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            return checkpoint
        raise FileNotFoundError(f"No checkpoint found at {filename}")

    def early_stopping(self, valid_loss, step):
        if valid_loss < self.best_loss - self.min_delta:
            self.best_loss = valid_loss
            self.counter = 0
            self.best_model = self.model.state_dict()
            self.best_step = step
            self.save_checkpoint({
                'step': step,
                'model_state_dict': self.best_model,
                'loss': valid_loss,
            }, os.path.join(self.save_dir, f'best_model.pth'))
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered. Best model was at step {self.best_step}")
                return True
        return False

    def train_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError