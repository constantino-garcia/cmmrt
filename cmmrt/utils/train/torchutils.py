import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from cmmrt.utils.train.model_selection import stratified_train_test_split

_FLOAT = 'float32'

class EarlyStopping:
    """https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='es.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        if path is None:
            path = str(uuid.uuid4()) + '.pth'
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        if self.verbose:
            self.trace_func('Loading best model...')
        model.load_state_dict(torch.load(self.path))


def torch_dataloaders(X, y, batch_size, test_size=0.0, n_strats=6):
    if test_size > 0:
        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=test_size, n_strats=n_strats
        )
    else:
        X_train, y_train = X, y
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).view(-1, 1)),
        batch_size=batch_size, shuffle=True
    )
    if test_size > 0:
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).view(-1, 1)),
            batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader
    else:
        return train_loader


def to_numpy(x):
    if torch.is_tensor(x):
        return x.cpu().numpy().astype(_FLOAT)
    else:
        return x.astype(_FLOAT)


def to_torch(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    else:
        return torch.tensor(x, requires_grad=False, dtype=torch.float32).to(device)
