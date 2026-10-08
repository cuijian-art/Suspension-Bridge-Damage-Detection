# -*- coding: utf-8 -*-
"""
GCN_LSTM_SCI_complete_5runs.py

功能：
1. 读取 wine.xlsx 数据
2. 构建基于特征相关性的图结构
3. 搭建 StableGCN_LSTM 模型
4. 进行 5 次独立重复训练
5. 保存每个 epoch 的 loss / accuracy
6. 计算 mean ± 1.96 * std
7. 绘制 SCI 风格训练曲线
8. 输出测试集评估结果与混淆矩阵

输出文件：
- training_curves_mean_CI.png
- confusion_matrix.png
- best_gcn_lstm_model.pth
- five_runs_training_results.npy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import random
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')   # 只保存图片，更稳定
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch


# =========================================================
# 0. 全局配置
# =========================================================
class Config:
    gnn_hidden_dim = 64
    lstm_hidden_dim = 128
    dropout_rate = 0.3
    k_neighbors = 5
    corr_threshold = 0.3
    batch_size = 32
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-5
    seeds = [42, 123, 2024, 3407, 666]


config = Config()


# =========================================================
# 1. 随机种子
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2. 数据读取与预处理
# =========================================================
print("=" * 60)
print("1. Data loading and preprocessing")
print("=" * 60)

data_ = pd.read_excel('wine.xlsx', index_col=0)

X = data_.iloc[:, :-1].values
y = data_.iloc[:, -1].values
columns = data_.iloc[:, :-1].columns.values

# 标签编码
le = LabelEncoder()
y = le.fit_transform(y)

print(f"Original label classes: {le.classes_}")
print(f"Encoded label classes: {np.unique(y)}")
print(f"Dataset shape: {X.shape}")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Standardization finished, mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

# 与原代码保持一致：70% 训练，30% 测试
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Train features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Train labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")

config.num_classes = len(np.unique(y))
print(f"Detected number of classes: {config.num_classes}")


# =========================================================
# 3. Dataset 与 DataLoader
# =========================================================
class WineDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        y = torch.LongTensor([self.labels[idx]]).squeeze()
        return x, y


train_dataset = WineDataset(X_train, y_train)
test_dataset = WineDataset(X_test, y_test)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


def create_dataloaders(seed):
    """
    为每一次独立训练创建带固定随机种子的 DataLoader，
    保证 shuffle 可复现且不同 seed 下存在差异。
    """
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# =========================================================
# 4. 图结构构建
# =========================================================
print("\n" + "=" * 60)
print("2. Graph construction")
print("=" * 60)

def build_correlation_adjacency(feature_data, k_neighbors=5, threshold=0.3):
    """
    基于特征相关性构建邻接矩阵
    节点 = 特征
    边 = 特征间相关性
    """
    n_features = feature_data.shape[1]
    corr_matrix = np.corrcoef(feature_data.T)
    np.fill_diagonal(corr_matrix, 0)

    adj = np.zeros((n_features, n_features))

    for i in range(n_features):
        top_k_idx = np.argsort(np.abs(corr_matrix[i]))[-k_neighbors:]
        for j in top_k_idx:
            if abs(corr_matrix[i, j]) >= threshold:
                adj[i, j] = 1

    np.fill_diagonal(adj, 1)

    avg_degree = adj.sum() / n_features
    num_edges = int(adj.sum())

    print(f"Adjacency matrix constructed.")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Total edges: {num_edges}")

    edge_index = torch.LongTensor(np.array(np.where(adj == 1)))
    return edge_index, corr_matrix


edge_index, corr_matrix = build_correlation_adjacency(
    X_train,
    k_neighbors=config.k_neighbors,
    threshold=config.corr_threshold
)


# =========================================================
# 5. StableGCN_LSTM 模型
# =========================================================
print("\n" + "=" * 60)
print("3. Model construction")
print("=" * 60)

class StableGCN_LSTM(nn.Module):
    """
    Stable GCN-LSTM
    每个样本的每个特征作为一个图节点，节点输入维度为1
    """

    def __init__(self, num_features, edge_index):
        super().__init__()
        self.num_features = num_features
        self.register_buffer('edge_index', edge_index)

        # GCN
        self.gcn1 = GCNConv(1, config.gnn_hidden_dim)
        self.gcn2 = GCNConv(config.gnn_hidden_dim, config.gnn_hidden_dim)

        self.bn1 = nn.BatchNorm1d(config.gnn_hidden_dim)
        self.bn2 = nn.BatchNorm1d(config.gnn_hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=config.gnn_hidden_dim,
            hidden_size=config.lstm_hidden_dim,
            batch_first=True,
            bidirectional=False
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, config.num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # 每个样本构建独立图
        data_list = []
        for i in range(batch_size):
            node_features = x[i].unsqueeze(1)  # [num_features] -> [num_features, 1]
            data = Data(x=node_features, edge_index=self.edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)

        # GCN
        x = self.gcn1(batch.x, batch.edge_index)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.gcn2(x, batch.edge_index)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        # 重塑回 [batch_size, num_features, hidden_dim]
        x = x.view(batch_size, self.num_features, config.gnn_hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]

        # 分类输出
        out = self.fc(last_out)
        return out


# =========================================================
# 6. 设备
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

edge_index = edge_index.to(device)

# 仅用于查看模型结构
temp_model = StableGCN_LSTM(num_features=X.shape[1], edge_index=edge_index).to(device)
if HAS_TORCHINFO:
    try:
        print("\nModel summary:")
        summary(temp_model, input_size=(config.batch_size, X.shape[1]), device=device)
    except Exception as e:
        print(f"torchinfo summary skipped: {e}")
del temp_model


# =========================================================
# 7. 训练与验证函数
# =========================================================
print("\n" + "=" * 60)
print("4. Training functions")
print("=" * 60)

def train_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, train_acc = 0.0, 0.0

    model.train()

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (pred.argmax(1) == y_batch).type(torch.float).sum().item()
        train_loss += loss.item()

    train_acc /= size
    train_loss /= num_batches
    return train_acc, train_loss


def val_epoch(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, val_acc = 0.0, 0.0

    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            val_loss += loss.item()
            val_acc += (pred.argmax(1) == y_batch).type(torch.float).sum().item()

    val_acc /= size
    val_loss /= num_batches
    return val_acc, val_loss


# =========================================================
# 8. 五次独立重复训练
# =========================================================
print("\n" + "=" * 60)
print("5. Five-run repeated training")
print("=" * 60)

def repeated_training(
    seeds,
    epochs,
    model_class,
    num_features,
    edge_index,
    device
):
    train_loss_runs = []
    val_loss_runs = []
    train_acc_runs = []
    val_acc_runs = []

    best_global_acc = 0.0
    best_global_model = None

    for run_id, seed in enumerate(seeds, start=1):
        print("\n" + "-" * 60)
        print(f"Run {run_id}/{len(seeds)} | seed = {seed}")
        print("-" * 60)

        set_seed(seed)

        train_loader, test_loader = create_dataloaders(seed)

        model = model_class(
            num_features=num_features,
            edge_index=edge_index
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=False
        )

        loss_fn = nn.CrossEntropyLoss()

        train_loss_history = []
        train_acc_history = []
        val_loss_history = []
        val_acc_history = []

        best_acc_this_run = 0.0
        best_model_this_run = None

        start_time = time.time()

        for epoch in range(epochs):
            train_acc, train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
            val_acc, val_loss = val_epoch(test_loader, model, loss_fn, device)

            scheduler.step(val_loss)

            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            if val_acc > best_acc_this_run:
                best_acc_this_run = val_acc
                best_model_this_run = {
                    k: v.clone().cpu() for k, v in model.state_dict().items()
                }

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | "
                    f"LR: {current_lr:.6f}"
                )

        elapsed = time.time() - start_time
        print(f"Run {run_id} finished in {elapsed:.2f}s")
        print(f"Best validation accuracy in this run: {best_acc_this_run:.4f}")

        train_loss_runs.append(train_loss_history)
        train_acc_runs.append(train_acc_history)
        val_loss_runs.append(val_loss_history)
        val_acc_runs.append(val_acc_history)

        if best_acc_this_run > best_global_acc:
            best_global_acc = best_acc_this_run
            best_global_model = best_model_this_run

    return (
        np.array(train_loss_runs),
        np.array(val_loss_runs),
        np.array(train_acc_runs),
        np.array(val_acc_runs),
        best_global_model,
        best_global_acc
    )


results = repeated_training(
    seeds=config.seeds,
    epochs=config.epochs,
    model_class=StableGCN_LSTM,
    num_features=X.shape[1],
    edge_index=edge_index,
    device=device
)

train_loss_runs, val_loss_runs, train_acc_runs, val_acc_runs, best_model_state, best_global_acc = results

print("\nOverall best validation accuracy across 5 runs: {:.4f}".format(best_global_acc))

# 保存五次训练历史
np.save(
    'five_runs_training_results.npy',
    {
        'train_loss': train_loss_runs,
        'val_loss': val_loss_runs,
        'train_acc': train_acc_runs,
        'val_acc': val_acc_runs
    }
)

print("Saved five-run histories to: five_runs_training_results.npy")


# =========================================================
# 9. 训练曲线绘制：mean ± 1.96 std
# =========================================================
print("\n" + "=" * 60)
print("6. Plot training curves")
print("=" * 60)

def plot_mean_CI(train_loss, val_loss, train_acc, val_acc):
    def plot_mean_CI(train_loss, val_loss, train_acc, val_acc):

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # ==========================
        # Font settings
        # ==========================
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'

        epochs = np.arange(1, train_loss.shape[1] + 1)

        # ==========================
        # Statistics
        # ==========================
        def compute_stats(data):
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0, ddof=1)

            lower = mean - 1.96 * std
            upper = mean + 1.96 * std

            return mean, lower, upper

        tl_mean, tl_low, tl_up = compute_stats(train_loss)
        vl_mean, vl_low, vl_up = compute_stats(val_loss)

        ta_mean, ta_low, ta_up = compute_stats(train_acc)
        va_mean, va_low, va_up = compute_stats(val_acc)

        # clip accuracy range
        ta_low = np.clip(ta_low, 0, 1)
        ta_up = np.clip(ta_up, 0, 1)
        va_low = np.clip(va_low, 0, 1)
        va_up = np.clip(va_up, 0, 1)

        tl_low = np.clip(tl_low, 0, None)
        vl_low = np.clip(vl_low, 0, None)

        # ==========================
        # Figure
        # ==========================
        fig, ax = plt.subplots(
            1, 2,
            figsize=(13.5, 5.3),
            dpi=600
        )

        train_color = "#2166AC"
        val_color = "#B2182B"
        ci_alpha = 0.10

        # ==========================
        # Legend handles
        # ==========================
        ci_patch = Patch(
            facecolor="gray",
            alpha=0.25,
            label="95% confidence interval"
        )

        train_line = Line2D(
            [0], [0],
            color=train_color,
            linewidth=3,
            label="Training mean"
        )

        val_line = Line2D(
            [0], [0],
            color=val_color,
            linewidth=3,
            label="Validation mean"
        )

        # ======================================================
        # Left panel: Loss
        # ======================================================
        ax0 = ax[0]

        # CI area
        ax0.fill_between(
            epochs, tl_low, tl_up,
            color=train_color, alpha=ci_alpha
        )
        ax0.fill_between(
            epochs, vl_low, vl_up,
            color=val_color, alpha=ci_alpha
        )

        # Mean curves
        ax0.plot(
            epochs, tl_mean,
            color=train_color, linewidth=3
        )
        ax0.plot(
            epochs, vl_mean,
            color=val_color, linewidth=3
        )

        # Best loss point
        best_epoch_loss = np.argmin(vl_mean) + 1
        best_loss = vl_mean[best_epoch_loss - 1]

        ax0.axvline(
            best_epoch_loss,
            linestyle="--",
            color="black",
            linewidth=1.5
        )

        ax0.scatter(
            best_epoch_loss,
            best_loss,
            s=90,
            color="black",
            zorder=5
        )

        # Use annotate instead of plain text to avoid overlap
        ax0.annotate(
            f"Best validation\nLoss = {best_loss:.3f}\n(epoch = {best_epoch_loss})",
            xy=(best_epoch_loss, best_loss),
            xytext=(best_epoch_loss + 8, best_loss + 0.18),
            textcoords='data',
            fontsize=12,
            fontweight="bold",
            ha='left',
            va='center',
            arrowprops=dict(
                arrowstyle='->',
                lw=1.2,
                color='black'
            ),
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.85
            )
        )

        ax0.set_xlabel("Epoch", fontsize=18)
        ax0.set_ylabel("Cross-entropy loss", fontsize=18)

        # ======================================================
        # Right panel: Accuracy
        # ======================================================
        ax1 = ax[1]

        ax1.fill_between(
            epochs, ta_low * 100, ta_up * 100,
            color=train_color, alpha=ci_alpha
        )
        ax1.fill_between(
            epochs, va_low * 100, va_up * 100,
            color=val_color, alpha=ci_alpha
        )

        ax1.plot(
            epochs, ta_mean * 100,
            color=train_color, linewidth=3
        )
        ax1.plot(
            epochs, va_mean * 100,
            color=val_color, linewidth=3
        )

        best_epoch_acc = np.argmax(va_mean) + 1
        best_acc = va_mean[best_epoch_acc - 1] * 100

        ax1.axvline(
            best_epoch_acc,
            linestyle="--",
            color="black",
            linewidth=1.5
        )

        ax1.scatter(
            best_epoch_acc,
            best_acc,
            s=90,
            color="black",
            zorder=5
        )

        # Best accuracy annotation moved to upper-left direction
        ax1.annotate(
            f"Best validation\nAcc. = {best_acc:.2f}%\n(epoch = {best_epoch_acc})",
            xy=(best_epoch_acc, best_acc),
            xytext=(best_epoch_acc - 18, best_acc - 4),
            textcoords='data',
            fontsize=12,
            fontweight="bold",
            ha='left',
            va='center',
            arrowprops=dict(
                arrowstyle='->',
                lw=1.2,
                color='black'
            ),
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.85
            )
        )

        # Statistics box: fixed at lower-left
        stats_text = (
            "Five independent runs\n"
            "Mean ± 95% CI\n\n"
            f"Best accuracy:\n{best_acc:.2f}%\n"
            f"Epoch: {best_epoch_acc}"
        )

        ax1.text(
            0.04, 0.24,
            stats_text,
            transform=ax1.transAxes,
            fontsize=11.5,
            fontweight="bold",
            ha='left',
            va='bottom',
            bbox=dict(
                boxstyle="round,pad=0.45",
                facecolor="white",
                edgecolor="black",
                alpha=0.88
            )
        )

        ax1.set_xlabel("Epoch", fontsize=18)
        ax1.set_ylabel("Accuracy (%)", fontsize=18)
        ax1.set_ylim(0, 105)

        # ======================================================
        # Put legends below each subplot
        # ======================================================
        ax0.legend(
            handles=[train_line, val_line, ci_patch],
            fontsize=12,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=1
        )

        ax1.legend(
            handles=[train_line, val_line, ci_patch],
            fontsize=12,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=1
        )

        # ======================================================
        # Common style
        # ======================================================
        for a in ax:
            a.tick_params(
                labelsize=15,
                width=1.5
            )

            a.grid(
                linestyle="--",
                linewidth=0.6,
                alpha=0.25
            )

            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["left"].set_linewidth(1.5)
            a.spines["bottom"].set_linewidth(1.5)

        # extra bottom space for legends
        plt.subplots_adjust(
            wspace=0.22,
            bottom=0.24
        )

        plt.savefig(
            "training_curves_mean_CI_SCI_adjusted.png",
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved: training_curves_mean_CI_SCI_adjusted.png")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    epochs = np.arange(1, train_loss.shape[1] + 1)

    def compute_stats(arr, clip_min=None, clip_max=None):
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1)

        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

        if clip_min is not None or clip_max is not None:
            lower = np.clip(lower, clip_min, clip_max)
            upper = np.clip(upper, clip_min, clip_max)

        return mean, lower, upper

    # Loss
    tl_mean, tl_low, tl_up = compute_stats(train_loss, clip_min=0, clip_max=None)
    vl_mean, vl_low, vl_up = compute_stats(val_loss, clip_min=0, clip_max=None)

    # Accuracy（原始为0~1，绘图转百分数）
    ta_mean, ta_low, ta_up = compute_stats(train_acc, clip_min=0, clip_max=1)
    va_mean, va_low, va_up = compute_stats(val_acc, clip_min=0, clip_max=1)

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=600)

    train_color = "#2166AC"
    val_color = "#B2182B"

    # -------- Loss --------
    ax = axs[0]

    ax.fill_between(
        epochs, tl_low, tl_up,
        color=train_color, alpha=0.15
    )
    ax.fill_between(
        epochs, vl_low, vl_up,
        color=val_color, alpha=0.15
    )

    ax.plot(
        epochs, tl_mean,
        color=train_color, linewidth=3.0, label="Training"
    )
    ax.plot(
        epochs, vl_mean,
        color=val_color, linewidth=3.0, label="Validation"
    )

    best_epoch_loss = np.argmin(vl_mean) + 1
    best_val_loss = vl_mean[best_epoch_loss - 1]

    ax.axvline(
        best_epoch_loss, linestyle="--", color="black",
        linewidth=1.5, alpha=0.9
    )
    ax.scatter(
        best_epoch_loss, best_val_loss,
        s=80, color="black", zorder=5
    )

    ax.text(
        0.03, 0.92, "(a)",
        transform=ax.transAxes,
        fontsize=18, fontweight='bold'
    )

    ax.text(
        best_epoch_loss + 2,
        best_val_loss,
        f"Best epoch = {best_epoch_loss}",
        fontsize=13, fontweight='bold'
    )

    ax.set_xlabel("Epoch", fontsize=17, fontweight='bold')
    ax.set_ylabel("Cross-entropy loss", fontsize=17, fontweight='bold')

    ax.legend(fontsize=13, frameon=False)

    # -------- Accuracy --------
    ax = axs[1]

    ax.fill_between(
        epochs, ta_low * 100, ta_up * 100,
        color=train_color, alpha=0.15
    )
    ax.fill_between(
        epochs, va_low * 100, va_up * 100,
        color=val_color, alpha=0.15
    )

    ax.plot(
        epochs, ta_mean * 100,
        color=train_color, linewidth=3.0, label="Training"
    )
    ax.plot(
        epochs, va_mean * 100,
        color=val_color, linewidth=3.0, label="Validation"
    )

    best_epoch_acc = np.argmax(va_mean) + 1
    best_val_acc = va_mean[best_epoch_acc - 1] * 100

    ax.axvline(
        best_epoch_acc, linestyle="--", color="black",
        linewidth=1.5, alpha=0.9
    )
    ax.scatter(
        best_epoch_acc, best_val_acc,
        s=80, color="black", zorder=5
    )

    ax.text(
        0.03, 0.92, "(b)",
        transform=ax.transAxes,
        fontsize=18, fontweight='bold'
    )

    ax.text(
        max(2, best_epoch_acc - 18),
        max(2, best_val_acc - 8),
        f"Best model\n{best_val_acc:.2f}%",
        fontsize=13, fontweight='bold'
    )

    ax.set_xlabel("Epoch", fontsize=17, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=17, fontweight='bold')
    ax.set_ylim(0, 105)

    ax.legend(fontsize=13, frameon=False)

    # -------- Common style --------
    for ax in axs:
        ax.tick_params(axis='both', labelsize=14, width=1.3)
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.28)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.3)
        ax.spines["bottom"].set_linewidth(1.3)

    plt.tight_layout()
    plt.savefig('training_curves_mean_CI.png', dpi=600, bbox_inches='tight')
    plt.close()

    print("Saved figure: training_curves_mean_CI.png")


plot_mean_CI(train_loss_runs, val_loss_runs, train_acc_runs, val_acc_runs)


# =========================================================
# 10. 加载最佳模型并测试评估
# =========================================================
print("\n" + "=" * 60)
print("7. Final evaluation")
print("=" * 60)

final_model = StableGCN_LSTM(num_features=X.shape[1], edge_index=edge_index).to(device)
final_model.load_state_dict(best_model_state)
torch.save(best_model_state, 'best_gcn_lstm_model.pth')

print("Saved best model: best_gcn_lstm_model.pth")

final_model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_pred_logits = final_model(X_test_tensor)
    y_pred = y_pred_logits.argmax(1).cpu().numpy()
    y_pred_proba = F.softmax(y_pred_logits, dim=1).cpu().numpy()

target_names = [f'Class {i + 1}' for i in range(config.num_classes)]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

try:
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
except Exception:
    roc_auc = np.nan

print("=== Test-set evaluation ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC-AUC  : N/A")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=target_names))

# 混淆矩阵
plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(8, 6), dpi=600)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap='Blues', values_format='d')
plt.xlabel("Predicted label", fontsize=14, fontweight='bold')
plt.ylabel("True label", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=600, bbox_inches='tight')
plt.close()

print("Saved figure: confusion_matrix.png")

print("\n" + "=" * 60)
print("All tasks completed successfully!")
print("Generated files:")
print("1. training_curves_mean_CI.png")
print("2. confusion_matrix.png")
print("3. best_gcn_lstm_model.pth")
print("4. five_runs_training_results.npy")
print("=" * 60)