"""PyTorch MLP scene classifier with an sklearn interface (see scene_classify.make_model)."""

import copy
import threading

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU, "elu": nn.ELU,
               "leaky_relu": nn.LeakyReLU}
OPTIMIZERS = ("adamw", "adam", "sgd")
_INIT_LOCK = threading.Lock()   # weight init uses torch's global CPU RNG; folds may train in threads


def gpu_devices():
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []


def resolve_device(device):
    """'auto' / 'all' -> first GPU (or cpu); anything else ('cpu', 'cuda:1') as given."""
    if device in ("auto", "all"):
        return (gpu_devices() or ["cpu"])[0]
    return device


class SceneMLP(nn.Module):
    """Linear -> [LayerNorm] -> activation -> Dropout per hidden layer; linear head to class logits."""

    def __init__(self, n_in, n_out=3, hidden=(128, 128, 64), dropout=0.1, layer_norm=False,
                 activation="relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), *([nn.LayerNorm(h)] if layer_norm else []),
                       ACTIVATIONS[activation](), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(BaseEstimator, ClassifierMixin):
    """Softmax MLP on a dense, scaled, NaN-free matrix.

    Early stopping holds out val_frac of the training *groups* (scene seeds) when given,
    so the stopping rule never sees a training scene.
    """

    def __init__(self, hidden=(128, 128, 64), dropout=0.1, layer_norm=False, activation="relu",
                 optimizer="adamw", lr=1e-3, momentum=0.9, weight_decay=1e-4, batch_size=1024,
                 max_epochs=200, patience=15, min_delta=1e-4, val_frac=0.15, class_weight=None,
                 random_state=0, device="auto", verbose=False):
        self.hidden = hidden
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_frac = val_frac
        self.class_weight = class_weight
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    def _t(self, X):
        return torch.as_tensor(np.asarray(X, np.float32), device=self.device_)

    def fit(self, X, y, groups=None):
        dev = self.device_ = resolve_device(self.device)
        if dev.startswith("cuda"):
            with torch.cuda.device(dev):
                torch.cuda.manual_seed(self.random_state)   # this device's dropout RNG only
        X = np.asarray(X, np.float32)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        split = (GroupShuffleSplit(1, test_size=self.val_frac, random_state=self.random_state) if groups is not None
                 else ShuffleSplit(1, test_size=self.val_frac, random_state=self.random_state))
        tr, va = next(split.split(X, yi, groups))
        Xt, yt, Xv, yv = self._t(X[tr]), torch.as_tensor(yi[tr], device=dev), \
            self._t(X[va]), torch.as_tensor(yi[va], device=dev)

        with _INIT_LOCK:   # CPU generator only: torch.manual_seed would reseed other threads' GPUs
            torch.default_generator.manual_seed(self.random_state)
            self.model_ = SceneMLP(X.shape[1], len(self.classes_), tuple(self.hidden), self.dropout,
                                   self.layer_norm, self.activation)
        self.model_.to(dev)
        w = None
        if self.class_weight == "balanced":
            w = torch.as_tensor(len(yi) / (len(self.classes_) * np.bincount(yi)), dtype=torch.float32,
                                device=dev)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        opt = self._optimizer(self.model_.parameters())
        gen = torch.Generator(device="cpu").manual_seed(self.random_state)

        best, best_state, wait, self.history_ = np.inf, None, 0, []
        for ep in range(self.max_epochs):
            self.model_.train()
            for idx in torch.randperm(len(Xt), generator=gen).to(dev).split(self.batch_size):
                opt.zero_grad()
                loss_fn(self.model_(Xt[idx]), yt[idx]).backward()
                opt.step()
            self.model_.eval()
            with torch.no_grad():
                vl = loss_fn(self.model_(Xv), yv).item()
            self.history_.append(vl)
            if vl < best - self.min_delta:
                best, best_state, wait = vl, copy.deepcopy(self.model_.state_dict()), 0
            elif (wait := wait + 1) >= self.patience:
                break
            if self.verbose:
                print(f"[mlp] epoch {ep} val {vl:.4f}")
        self.model_.load_state_dict(best_state)
        self.best_val_loss_, self.n_epochs_ = best, len(self.history_)
        return self

    def _optimizer(self, params):
        kw = dict(lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer == "adamw":
            return torch.optim.AdamW(params, **kw)
        if self.optimizer == "adam":
            return torch.optim.Adam(params, **kw)
        if self.optimizer == "sgd":   # momentum used only here
            return torch.optim.SGD(params, momentum=self.momentum, **kw)
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")

    def predict_proba(self, X):
        self.model_.eval()
        out = []
        with torch.no_grad():
            for xb in self._t(X).split(65536):
                out.append(torch.softmax(self.model_(xb), 1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
