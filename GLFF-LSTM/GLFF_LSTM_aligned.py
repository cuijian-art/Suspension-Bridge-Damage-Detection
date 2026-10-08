# -*- coding: utf-8 -*-
"""
Paper-aligned GLFF-LSTM implementation
======================================
Aligned with the manuscript concept:
  Sparse graph-constrained cross-view contrastive learning for
  physics-aware bridge damage identification

Core components
---------------
1) Train-only z-score normalization (no data leakage)
2) Pearson-correlation Top-k sparse weighted feature graph
3) Union-based symmetrization and symmetric graph normalization
4) Local graph aggregation (Graph MLP)
5) Graph-constrained multi-head attention (Graph Transformer)
6) Dimension-wise local-global adaptive gating
7) Graph-view representation: average pooling + max pooling -> 128-D
8) Sequential view: single-layer unidirectional LSTM -> 128-D
9) Cross-view MSE feature alignment
10) Supervised contrastive learning on L2-normalized graph/sequence views
11) Joint objective:
      L = L_CE + lambda_a * L_align + lambda_c * L_con
12) Independent train/validation/test split; the test set is untouched during
    training/model selection
13) Five independent runs; mean/std and 95% CI are reported

NOTE ABOUT MANUSCRIPT CONSISTENCY
---------------------------------
This code follows the manuscript version in which the graph-level readout is
"average pooling + max pooling" (the version also used in Table 1 and Sec. 2.4.1).
If Sec. 2.3.2 still says "attention pooling + max pooling", revise that wording
(or replace the readout here with an attention-pooling implementation).
"""

import os
import copy
import random
import time
import warnings
from dataclasses import dataclass

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# =============================================================================
# 0. Configuration
# =============================================================================
@dataclass
class Config:
    data_file: str = "wine.xlsx"       # replace with each bridge dataset file
    from typing import Optional

    index_col: Optional[int] = 0

    # split (stratified)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # model
    gnn_hidden_dim: int = 64
    lstm_hidden_dim: int = 128
    num_glff_layers: int = 2
    num_heads: int = 4
    dropout_rate: float = 0.30

    # sparse graph
    k_neighbors: int = 5
    corr_threshold: float = 0.30  # tau

    # joint objective
    align_loss_weight: float = 0.30   # lambda_a
    contrast_loss_weight: float = 0.20  # lambda_c
    contrast_temperature: float = 0.10  # T

    # optimization
    batch_size: int = 32
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    early_stop_patience: int = 50
    early_stop_delta: float = 1e-4

    # repeated runs
    seeds: tuple = (42, 123, 2024, 3407, 666)

    # outputs
    output_dir: str = "paper_aligned_outputs"


CFG = Config()
os.makedirs(CFG.output_dir, exist_ok=True)


# =============================================================================
# 1. Reproducibility
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Data preparation (split BEFORE scaling)
# =============================================================================
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_and_split_data(cfg: Config):
    data = pd.read_excel(cfg.data_file, index_col=cfg.index_col)
    X_raw = data.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_raw = data.iloc[:, -1].to_numpy()
    feature_names = [str(c) for c in data.iloc[:, :-1].columns]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-8

    # First hold out the independent test set.
    X_dev, X_test_raw, y_dev, y_test = train_test_split(
        X_raw,
        y,
        test_size=cfg.test_ratio,
        random_state=cfg.split_seed,
        stratify=y,
    )

    # Split development data into train and validation sets.
    val_fraction_of_dev = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=val_fraction_of_dev,
        random_state=cfg.split_seed,
        stratify=y_dev,
    )

    # IMPORTANT: fit preprocessing statistics ONLY on the training set.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    print("Dataset loaded:")
    print(f"  features      : {X_raw.shape[1]}")
    print(f"  classes       : {len(label_encoder.classes_)}")
    print(f"  train / val / test = {len(y_train)} / {len(y_val)} / {len(y_test)}")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "num_classes": len(label_encoder.classes_),
    }


# =============================================================================
# 3. Correlation-based Top-k sparse WEIGHTED feature graph
# =============================================================================
def build_correlation_graph(
    X_train,
    k_neighbors=5,
    threshold=0.3,
    eps=1e-12,
):
    """
    Build the manuscript-aligned graph using TRAINING DATA ONLY.

    Directed candidate graph:
        A_ij^dir = |rho_ij|, if j is among Top-k neighbors of i and
                              |rho_ij| >= threshold
                   0,        otherwise

    Union-based symmetrization:
        A_ij = max(A_ij^dir, A_ji^dir)

    Self loops and symmetric normalization:
        A_tilde = A + I
        A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}

    The Graph Transformer uses only graph connectivity as a mask; the
    local Graph MLP uses the normalized weighted adjacency A_hat.
    """
    n_features = X_train.shape[1]
    corr = np.corrcoef(X_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    directed = np.zeros((n_features, n_features), dtype=np.float32)
    abs_corr = np.abs(corr)

    k_eff = min(k_neighbors, max(1, n_features - 1))
    for i in range(n_features):
        # Descending absolute-correlation order, excluding self (diag is zero).
        top_idx = np.argsort(abs_corr[i])[::-1][:k_eff]
        for j in top_idx:
            strength = abs_corr[i, j]
            if strength >= threshold:
                directed[i, j] = strength

    # Union-based symmetrization (undirected weighted graph).
    weighted_adj = np.maximum(directed, directed.T)

    # Binary connectivity used by graph-constrained attention.
    connectivity = (weighted_adj > 0).astype(np.float32)

    # Add self-loops for local graph propagation and attention.
    A_tilde = weighted_adj + np.eye(n_features, dtype=np.float32)
    connectivity_with_self = np.maximum(
        connectivity, np.eye(n_features, dtype=np.float32)
    )

    degree = A_tilde.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, eps))
    D_inv_sqrt = np.diag(inv_sqrt_degree)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # 0 for valid graph edges; -1e9 elsewhere.
    attention_mask = np.where(
        connectivity_with_self > 0,
        0.0,
        -1e9,
    ).astype(np.float32)
    attention_mask = attention_mask[None, None, :, :]  # [1,1,N,N]

    num_undirected_edges = int(np.triu(connectivity, k=1).sum())
    mean_degree = float(connectivity.sum(axis=1).mean())
    print("Sparse weighted graph built from training data only:")
    print(f"  Top-k / tau   : {k_neighbors} / {threshold}")
    print(f"  undirected edges: {num_undirected_edges}")
    print(f"  mean degree (without self-loops): {mean_degree:.2f}")

    return {
        "corr_matrix": corr,
        "weighted_adj": torch.tensor(weighted_adj, dtype=torch.float32),
        "adj_norm": torch.tensor(A_hat, dtype=torch.float32),
        "adj_mask": torch.tensor(attention_mask, dtype=torch.float32),
        "connectivity": torch.tensor(connectivity_with_self, dtype=torch.float32),
    }


# =============================================================================
# 4. GLFF layers
# =============================================================================
class GraphMLPLayer(nn.Module):
    """Local graph aggregation over the normalized sparse weighted graph."""

    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj_norm):
        # x: [B,N,d], adj_norm: [N,N]
        x_agg = torch.matmul(adj_norm, x)
        out = self.linear(x_agg)
        out = self.norm(out)
        out = self.act(out)
        return self.dropout(out)


class GraphTransformerLayer(nn.Module):
    """Sample-dependent multi-head attention restricted to sparse graph edges."""

    def __init__(self, dim, num_heads=4, dropout_rate=0.3):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, adj_mask):
        B, N, D = x.shape
        residual = x

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        logits = logits + adj_mask
        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        x1 = self.norm1(residual + self.dropout(out))
        x2 = self.norm2(x1 + self.dropout(self.ffn(x1)))
        return x2


class LocalGlobalGate(nn.Module):
    """
    Dimension-wise local-global adaptive gate.

    The gate uses the two branch representations together with their
    difference and interaction so that the weighting can respond to both
    consistency and discrepancy:

      r = sigmoid(W_r [L || G || |L-G| || L*G] + b_r)
      H = r * L + (1-r) * G

    r has shape [B,N,d].
    """

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 4, dim)

    def forward(self, local_feat, global_feat):
        gate_input = torch.cat(
            [
                local_feat,
                global_feat,
                torch.abs(local_feat - global_feat),
                local_feat * global_feat,
            ],
            dim=-1,
        )
        r = torch.sigmoid(self.gate(gate_input))
        fused = r * local_feat + (1.0 - r) * global_feat
        return fused, r


class GLFFModule(nn.Module):
    def __init__(
        self,
        in_dim=1,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.gates = nn.ModuleList()

        for _ in range(num_layers):
            self.local_layers.append(GraphMLPLayer(hidden_dim, dropout_rate))
            self.global_layers.append(
                GraphTransformerLayer(hidden_dim, num_heads, dropout_rate)
            )
            self.gates.append(LocalGlobalGate(hidden_dim))

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x, adj_norm, adj_mask, return_gates=False):
        x = self.input_proj(x)
        gate_values = []

        for local_layer, global_layer, gate in zip(
            self.local_layers, self.global_layers, self.gates
        ):
            local_feat = local_layer(x, adj_norm)
            global_feat = global_layer(x, adj_mask)
            x, r = gate(local_feat, global_feat)
            gate_values.append(r)

        x = self.output_proj(x)
        if return_gates:
            return x, gate_values
        return x


# =============================================================================
# 5. Cross-view GLFF-LSTM model
# =============================================================================
class GLFFLSTM(nn.Module):
    def __init__(self, num_features, num_classes, adj_norm, adj_mask, cfg: Config):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.cfg = cfg

        self.register_buffer("adj_norm", adj_norm)
        self.register_buffer("adj_mask", adj_mask)

        self.glff = GLFFModule(
            in_dim=1,
            hidden_dim=cfg.gnn_hidden_dim,
            num_layers=cfg.num_glff_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
        )

        # The graph-enhanced nodes are treated as an ordered sequence according
        # to the feature ordering in the input file (semantic-domain order).
        self.lstm_input_dropout = nn.Dropout(cfg.dropout_rate)
        self.lstm = nn.LSTM(
            input_size=cfg.gnn_hidden_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        # GLFF graph-view representation: avg(64) + max(64) = 128.
        graph_view_dim = cfg.gnn_hidden_dim * 2
        assert graph_view_dim == cfg.lstm_hidden_dim, (
            "For element-wise MSE alignment, graph-view and LSTM-view dimensions "
            "must be equal. Current values are "
            f"{graph_view_dim} and {cfg.lstm_hidden_dim}."
        )
        self.graph_view_norm = nn.LayerNorm(graph_view_dim)

        # Classification uses the LSTM representation h, as stated in the paper.
        self.classifier = nn.Sequential(
            nn.Linear(cfg.lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_gates=False):
        # x: [B,N] -> node scalar features [B,N,1]
        node_input = x.unsqueeze(-1)

        if return_gates:
            glff_out, gate_values = self.glff(
                node_input, self.adj_norm, self.adj_mask, return_gates=True
            )
        else:
            glff_out = self.glff(node_input, self.adj_norm, self.adj_mask)
            gate_values = None

        # Graph-view representation g: average + max pooling.
        g_avg = torch.mean(glff_out, dim=1)
        g_max = torch.max(glff_out, dim=1).values
        graph_feat = self.graph_view_norm(torch.cat([g_avg, g_max], dim=1))

        # Sequential view h: ordered GLFF node sequence -> unidirectional LSTM.
        seq_in = self.lstm_input_dropout(glff_out)
        lstm_out, _ = self.lstm(seq_in)
        lstm_feat = lstm_out[:, -1, :]

        logits = self.classifier(lstm_feat)

        if return_gates:
            return logits, graph_feat, lstm_feat, gate_values
        return logits, graph_feat, lstm_feat


# =============================================================================
# 6. Supervised contrastive loss
# =============================================================================
def supervised_contrastive_loss(features, labels, temperature=0.1):
    """
    Supervised contrastive loss over the combined graph/sequence views.

    features: [2B,D], expected to be L2-normalized before this call
    labels  : [2B]

    Because each sample appears in both views, every anchor always has at
    least its cross-view counterpart as a positive example.
    """
    device = features.device
    n = features.shape[0]

    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(n, device=device)
    positive_mask = positive_mask * (1.0 - self_mask)

    exp_logits = torch.exp(logits) * (1.0 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positive_count = positive_mask.sum(dim=1)
    mean_log_prob_pos = (
        (positive_mask * log_prob).sum(dim=1)
        / torch.clamp(positive_count, min=1.0)
    )
    return -mean_log_prob_pos.mean()


# =============================================================================
# 7. Training utilities
# =============================================================================
class EarlyStopping:
    def __init__(self, patience=50, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        return self.counter >= self.patience


def make_loaders(data_dict, cfg: Config, seed: int):
    gen = torch.Generator()
    gen.manual_seed(seed)

    train_ds = FeatureDataset(data_dict["X_train"], data_dict["y_train"])
    val_ds = FeatureDataset(data_dict["X_val"], data_dict["y_val"])
    test_ds = FeatureDataset(data_dict["X_test"], data_dict["y_test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=gen,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader


def compute_joint_loss(logits, graph_feat, lstm_feat, labels, cfg: Config):
    loss_ce = F.cross_entropy(logits, labels)
    loss_align = F.mse_loss(graph_feat, lstm_feat)

    g_norm = F.normalize(graph_feat, p=2, dim=1)
    h_norm = F.normalize(lstm_feat, p=2, dim=1)
    contrast_features = torch.cat([g_norm, h_norm], dim=0)
    contrast_labels = torch.cat([labels, labels], dim=0)

    loss_con = supervised_contrastive_loss(
        contrast_features,
        contrast_labels,
        temperature=cfg.contrast_temperature,
    )

    total = (
        loss_ce
        + cfg.align_loss_weight * loss_align
        + cfg.contrast_loss_weight * loss_con
    )
    return total, loss_ce, loss_align, loss_con


def run_epoch(loader, model, cfg, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "align": 0.0,
        "contrast": 0.0,
        "correct": 0,
        "count": 0,
        "batches": 0,
    }

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            logits, graph_feat, lstm_feat = model(xb)
            total, ce, align, con = compute_joint_loss(
                logits, graph_feat, lstm_feat, yb, cfg
            )

            if is_train:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

            totals["loss"] += total.item()
            totals["ce"] += ce.item()
            totals["align"] += align.item()
            totals["contrast"] += con.item()
            totals["correct"] += (logits.argmax(1) == yb).sum().item()
            totals["count"] += yb.numel()
            totals["batches"] += 1

    b = max(totals["batches"], 1)
    return {
        "loss": totals["loss"] / b,
        "ce": totals["ce"] / b,
        "align": totals["align"] / b,
        "contrast": totals["contrast"] / b,
        "acc": totals["correct"] / max(totals["count"], 1),
    }


def predict(loader, model, device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _, _ = model(xb)
            logits_all.append(logits.cpu())
            y_all.append(yb.cpu())

    logits = torch.cat(logits_all, dim=0)
    y_true = torch.cat(y_all, dim=0).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    return y_true, y_pred, probs


def classification_metrics(y_true, y_pred, probs):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        out["roc_auc_ovr"] = roc_auc_score(
            y_true,
            probs,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        out["roc_auc_ovr"] = np.nan

    return out


# =============================================================================
# 8. One independent run (validation selects the model; test is used once)
# =============================================================================
def train_one_run(seed, data_dict, graph_dict, cfg, device):
    set_seed(seed)
    train_loader, val_loader, test_loader = make_loaders(data_dict, cfg, seed)

    model = GLFFLSTM(
        num_features=data_dict["X_train"].shape[1],
        num_classes=data_dict["num_classes"],
        adj_norm=graph_dict["adj_norm"],
        adj_mask=graph_dict["adj_mask"],
        cfg=cfg,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.scheduler_factor,
        patience=cfg.scheduler_patience,
    )
    early_stopper = EarlyStopping(
        patience=cfg.early_stop_patience,
        delta=cfg.early_stop_delta,
    )

    history = {
        "train": [],
        "val": [],
    }

    best_epoch = 0
    best_val_loss = np.inf

    for epoch in range(1, cfg.epochs + 1):
        train_stats = run_epoch(
            train_loader, model, cfg, device, optimizer=optimizer
        )
        val_stats = run_epoch(val_loader, model, cfg, device, optimizer=None)
        scheduler.step(val_stats["loss"])

        history["train"].append(train_stats)
        history["val"].append(val_stats)

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch

        stop = early_stopper.step(val_stats["loss"], model)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"seed={seed:4d} | epoch={epoch:03d} | "
                f"train={train_stats['loss']:.4f}/{train_stats['acc']:.4f} | "
                f"val={val_stats['loss']:.4f}/{val_stats['acc']:.4f} | "
                f"CE={train_stats['ce']:.4f} | "
                f"Align={train_stats['align']:.4f} | "
                f"Con={train_stats['contrast']:.4f}"
            )

        if stop:
            print(f"seed={seed}: early stopping at epoch {epoch}")
            break

    # Restore the validation-selected model BEFORE touching the test set.
    model.load_state_dict(early_stopper.best_state)

    y_true, y_pred, probs = predict(test_loader, model, device)
    metrics = classification_metrics(y_true, y_pred, probs)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_loss"] = best_val_loss

    return {
        "seed": seed,
        "model_state": copy.deepcopy(model.state_dict()),
        "history": history,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
    }


# =============================================================================
# 9. Five-run summary and SCI-style uncertainty reporting
# =============================================================================
def summarize_runs(run_results, cfg: Config):
    metric_names = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc_ovr",
    ]

    rows = []
    for r in run_results:
        row = {"seed": r["seed"]}
        row.update(r["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(
        os.path.join(cfg.output_dir, "five_run_test_metrics.csv"),
        index=False,
    )

    summary_rows = []
    n = len(df)
    for name in metric_names:
        vals = df[name].dropna().to_numpy(dtype=float)
        mean = vals.mean()
        std = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        summary_rows.append(
            {
                "metric": name,
                "mean": mean,
                "std": std,
                "ci95_half_width": ci95,
                "ci95_lower": mean - ci95,
                "ci95_upper": mean + ci95,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(cfg.output_dir, "five_run_summary.csv"),
        index=False,
    )

    print("\nFive-run independent TEST summary (test never used for model selection):")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['metric']:>16s}: "
            f"{row['mean']:.4f} ± {row['std']:.4f} "
            f"(95% CI {row['ci95_lower']:.4f}–{row['ci95_upper']:.4f})"
        )

    return df, summary_df


# =============================================================================
# 10. Training curves across runs (mean ± 95% CI)
# =============================================================================
def plot_training_curves(run_results, cfg: Config):
    # Curves can have different lengths because of early stopping.
    min_len = min(len(r["history"]["train"]) for r in run_results)

    def stack(which, key):
        return np.asarray(
            [
                [e[key] for e in r["history"][which][:min_len]]
                for r in run_results
            ],
            dtype=float,
        )

    epochs = np.arange(1, min_len + 1)
    train_loss = stack("train", "loss")
    val_loss = stack("val", "loss")
    train_acc = stack("train", "acc")
    val_acc = stack("val", "acc")

    def mean_ci(arr):
        mean = arr.mean(axis=0)
        if arr.shape[0] > 1:
            sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ci = 1.96 * sem
        else:
            ci = np.zeros_like(mean)
        return mean, mean - ci, mean + ci

    tl_m, tl_l, tl_u = mean_ci(train_loss)
    vl_m, vl_l, vl_u = mean_ci(val_loss)
    ta_m, ta_l, ta_u = mean_ci(train_acc)
    va_m, va_l, va_u = mean_ci(val_acc)

    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=300)

    ax = axes[0]
    ax.fill_between(epochs, tl_l, tl_u, alpha=0.15)
    ax.fill_between(epochs, vl_l, vl_u, alpha=0.15)
    ax.plot(epochs, tl_m, linewidth=2.2, label="Training mean")
    ax.plot(epochs, vl_m, linewidth=2.2, label="Validation mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Joint loss")
    ax.grid(ls="--", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.fill_between(epochs, ta_l * 100, ta_u * 100, alpha=0.15)
    ax.fill_between(epochs, va_l * 100, va_u * 100, alpha=0.15)
    ax.plot(epochs, ta_m * 100, linewidth=2.2, label="Training mean")
    ax.plot(epochs, va_m * 100, linewidth=2.2, label="Validation mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(ls="--", alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    path = os.path.join(cfg.output_dir, "training_curves_5runs_mean_95CI.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to: {path}")


# =============================================================================
# 11. Save graph diagnostics
# =============================================================================
def save_graph_diagnostics(graph_dict, feature_names, cfg: Config):
    np.save(
        os.path.join(cfg.output_dir, "pearson_corr_train_only.npy"),
        graph_dict["corr_matrix"],
    )
    np.save(
        os.path.join(cfg.output_dir, "weighted_sparse_adjacency.npy"),
        graph_dict["weighted_adj"].cpu().numpy(),
    )
    np.save(
        os.path.join(cfg.output_dir, "normalized_adjacency.npy"),
        graph_dict["adj_norm"].cpu().numpy(),
    )
    pd.Series(feature_names, name="feature_name").to_csv(
        os.path.join(cfg.output_dir, "feature_names.csv"),
        index=False,
    )


# =============================================================================
# 12. Main
# =============================================================================
def main():
    t0 = time.time()
    set_seed(CFG.split_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dict = load_and_split_data(CFG)

    graph_dict = build_correlation_graph(
        data_dict["X_train"],
        k_neighbors=CFG.k_neighbors,
        threshold=CFG.corr_threshold,
    )
    save_graph_diagnostics(graph_dict, data_dict["feature_names"], CFG)

    # Move fixed graph tensors once; they are then registered as model buffers.
    graph_dict["adj_norm"] = graph_dict["adj_norm"].to(device)
    graph_dict["adj_mask"] = graph_dict["adj_mask"].to(device)

    run_results = []
    for run_idx, seed in enumerate(CFG.seeds, start=1):
        print("\n" + "=" * 72)
        print(f"Independent run {run_idx}/{len(CFG.seeds)} | seed={seed}")
        print("=" * 72)
        result = train_one_run(seed, data_dict, graph_dict, CFG, device)
        run_results.append(result)
        print("Test metrics:", result["metrics"])

    metrics_df, summary_df = summarize_runs(run_results, CFG)
    plot_training_curves(run_results, CFG)

    # Save the validation-best model from the run with the best validation loss.
    # IMPORTANT: selection does NOT use test accuracy.
    representative_idx = int(
        np.argmin([r["metrics"]["best_val_loss"] for r in run_results])
    )
    representative = run_results[representative_idx]
    model_path = os.path.join(CFG.output_dir, "representative_validation_best_model.pth")
    torch.save(representative["model_state"], model_path)
    print(f"Representative model saved to: {model_path}")

    # Detailed report for the representative validation-selected run.
    class_names = [str(x) for x in data_dict["label_encoder"].classes_]
    report = classification_report(
        representative["y_true"],
        representative["y_pred"],
        target_names=class_names,
        zero_division=0,
    )
    with open(
        os.path.join(CFG.output_dir, "representative_classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    cm = confusion_matrix(representative["y_true"], representative["y_pred"])
    np.savetxt(
        os.path.join(CFG.output_dir, "representative_confusion_matrix.csv"),
        cm,
        delimiter=",",
        fmt="%d",
    )

    print(f"\nAll finished in {(time.time() - t0):.1f} s")
    print(f"Outputs: {os.path.abspath(CFG.output_dir)}")


if __name__ == "__main__":
    main()
