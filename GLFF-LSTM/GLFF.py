import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import font_manager
from matplotlib.patches import Rectangle


# ========================== 0. 实验配置 ==========================
class Config:
    seed = 42

    # 与原GLFF-LSTM保持一致
    glff_hidden_dim = 64
    glff_num_layers = 2
    glff_num_heads = 4
    dropout_rate = 0.3
    k_neighbors = 5
    corr_threshold = 0.3

    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5

    # 原GLFF-LSTM代码为300轮。
    # 正式比较时，请确保GLFF-LSTM、GLFF-only、GCN-only、LSTM-only使用相同训练轮次。
    epochs = 100

    scheduler_patience = 20
    early_stopping_patience = 50
    early_stopping_delta = 0.001

    test_size = 0.30
    random_state = 42

    num_classes = None


config = Config()

SCRIPT_DIR = Path(__file__).resolve().parent

# 优先使用原GLFF-LSTM代码中的 wine.xlsx。
# 若你的统一消融实验全部使用 wine原始特征顺序.xlsx，可调整顺序。
DATA_FILENAMES = ['wine.xlsx']

MODEL_FILENAME = 'best_glff_only_ablation_model.pth'
METRICS_FILENAME = 'GLFF_only_ablation_metrics.csv'
CLASS_REPORT_FILENAME = 'GLFF_only_ablation_classification_report.csv'
CONFUSION_MATRIX_CSV = 'GLFF_only_ablation_confusion_matrix.csv'
OUTPUT_STEM = 'GLFF_only_ablation_confusion_matrix_TNR_ocean_C1C6'

SHOW_ROW_PERCENT = True


def set_seed(seed):
    """固定随机种子，增强消融实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_data_file():
    searched = []

    for folder in [SCRIPT_DIR, Path.cwd()]:
        for filename in DATA_FILENAMES:
            candidate = folder / filename
            searched.append(str(candidate))

            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        '未找到数据文件，已检查：\n' + '\n'.join(searched)
    )


set_seed(config.seed)


# ========================== 1. Times New Roman与论文配色 ==========================
times_font_candidates = [
    r'C:\Windows\Fonts\times.ttf',
    r'C:\Windows\Fonts\timesbd.ttf',
    r'C:\Windows\Fonts\timesi.ttf',
    r'C:\Windows\Fonts\timesbi.ttf',
]

for font_path in times_font_candidates:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)

try:
    times_regular_path = font_manager.findfont(
        font_manager.FontProperties(family='Times New Roman'),
        fallback_to_default=False,
    )
except ValueError as exc:
    raise RuntimeError(
        '未检测到 Times New Roman，请确认Windows字体目录中存在 times.ttf。'
    ) from exc

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.sans-serif': ['Times New Roman'],
    'font.size': 12,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})

OCEAN_COLORS = [
    '#F5F8FA',
    '#DDECEF',
    '#B8D9D8',
    '#78B8B2',
    '#3E8790',
    '#22566F',
    '#102F4C',
]

PUBLICATION_CMAP = LinearSegmentedColormap.from_list(
    'publication_ocean',
    OCEAN_COLORS,
    N=256,
)

print(f'✅ 绘图字体：Times New Roman ({times_regular_path})')


# ========================== 2. 数据加载与预处理 ==========================
data_path = find_data_file()
data = pd.read_excel(data_path, index_col=0)

X = data.iloc[:, :-1].values
raw_y = data.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(raw_y)

config.num_classes = len(np.unique(y))

print('=' * 72)
print('GLFF-only 消融实验：删除LSTM，仅保留GLFF')
print('=' * 72)
print(f'数据文件：{data_path}')
print(f'样本数：{X.shape[0]}')
print(f'特征数/图节点数：{X.shape[1]}')
print(f'类别数：{config.num_classes}')
print(f'原始标签：{list(label_encoder.classes_)}')

# 与原GLFF-LSTM代码保持一致：
# 先对全体数据标准化，再进行70%/30%分层划分。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=y,
)


# ========================== 3. 数据集与DataLoader ==========================
class BridgeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = torch.as_tensor(
            self.features[index],
            dtype=torch.float32,
        )
        label = torch.as_tensor(
            self.labels[index],
            dtype=torch.long,
        )
        return feature, label


train_dataset = BridgeDataset(X_train, y_train)
test_dataset = BridgeDataset(X_test, y_test)

loader_generator = torch.Generator()
loader_generator.manual_seed(config.seed)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    generator=loader_generator,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
)


# ========================== 4. 相关性邻接矩阵 ==========================
def build_correlation_adjacency(
    feature_data,
    k_neighbors=5,
    threshold=0.3,
):
    """
    与原GLFF-LSTM保持一致：
    节点表示特征，按训练集特征相关性构建邻接矩阵和注意力掩码。
    """
    n_features = feature_data.shape[1]

    corr_matrix = np.corrcoef(feature_data.T)
    np.fill_diagonal(corr_matrix, 0)

    adjacency = np.zeros(
        (n_features, n_features),
        dtype=np.float32,
    )

    for i in range(n_features):
        top_k_indices = np.argsort(
            np.abs(corr_matrix[i])
        )[-k_neighbors:]

        for j in top_k_indices:
            if abs(corr_matrix[i, j]) >= threshold:
                adjacency[i, j] = 1.0

    np.fill_diagonal(adjacency, 1.0)

    adjacency_tensor = torch.as_tensor(
        adjacency,
        dtype=torch.float32,
    )

    adjacency_mask = (
        1.0
        - adjacency_tensor.unsqueeze(0).unsqueeze(0)
    ) * (-1e9)

    print(
        f'邻接矩阵：平均度数={adjacency.sum() / n_features:.2f}，'
        f'边数={int(adjacency.sum())}'
    )

    return adjacency_tensor, adjacency_mask


adjacency, adjacency_mask = build_correlation_adjacency(
    X_train,
    k_neighbors=config.k_neighbors,
    threshold=config.corr_threshold,
)


# ========================== 5. GLFF模块 ==========================
class GraphMLPLayer(nn.Module):
    """GLFF局部分支：基于邻接矩阵进行局部图聚合。"""

    def __init__(
        self,
        in_dim,
        out_dim,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adjacency):
        aggregated = adjacency @ x
        output = self.linear(aggregated)
        output = self.norm(output)
        output = self.activation(output)
        return self.dropout(output)


class GraphTransformerLayer(nn.Module):
    """GLFF全局分支：带图掩码的多头自注意力。"""

    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads=4,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        if self.head_dim * num_heads != out_dim:
            raise ValueError('输出维度必须是注意力头数的整数倍。')

        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x, adjacency_mask):
        batch_size, node_count, _ = x.shape
        residual = x

        query = self.q_proj(x).view(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        key = self.k_proj(x).view(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        value = self.v_proj(x).view(
            batch_size,
            node_count,
            self.num_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)

        attention = (
            query @ key.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        attention = attention + adjacency_mask
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        output = attention @ value

        output = output.permute(
            0,
            2,
            1,
            3,
        ).contiguous().view(
            batch_size,
            node_count,
            -1,
        )

        output = self.out_proj(output)

        x = self.norm1(residual + output)
        x = self.norm2(x + self.mlp(x))

        return self.dropout(x)


class GLFFModule(nn.Module):
    """局部图聚合与全局图注意力融合模块。"""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.input_projection = nn.Linear(
            in_dim,
            hidden_dim,
        )

        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.local_layers.append(
                GraphMLPLayer(
                    hidden_dim,
                    hidden_dim,
                    dropout_rate,
                )
            )

            self.global_layers.append(
                GraphTransformerLayer(
                    hidden_dim,
                    hidden_dim,
                    num_heads,
                    dropout_rate,
                )
            )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        x,
        adjacency,
        adjacency_mask,
    ):
        x = self.input_projection(x)

        for layer_index in range(self.num_layers):
            local_features = self.local_layers[layer_index](
                x,
                adjacency,
            )

            global_features = self.global_layers[layer_index](
                x,
                adjacency_mask,
            )

            x = local_features + global_features

        return self.output_projection(x)


# ========================== 6. GLFF-only消融模型 ==========================
class StableGLFFOnly(nn.Module):
    """
    GLFF-LSTM的GLFF-only消融版本。

    保留：
    1. 两层GLFF；
    2. 局部GraphMLP分支；
    3. 全局Graph Transformer分支；
    4. BatchNorm、Dropout和64维分类头。

    删除：
    1. LSTM层；
    2. LSTM最后时间步输出。

    替代：
    对GLFF输出的全部特征节点做全局平均池化，
    得到64维图级表示，再送入分类头。
    """

    def __init__(
        self,
        num_features,
        adjacency,
        adjacency_mask,
    ):
        super().__init__()

        self.num_features = num_features

        self.register_buffer(
            'adjacency',
            adjacency,
        )
        self.register_buffer(
            'adjacency_mask',
            adjacency_mask,
        )

        self.glff = GLFFModule(
            in_dim=1,
            hidden_dim=config.glff_hidden_dim,
            num_layers=config.glff_num_layers,
            num_heads=config.glff_num_heads,
            dropout_rate=config.dropout_rate,
        )

        self.batch_norm = nn.BatchNorm1d(
            config.glff_hidden_dim
        )
        self.dropout = nn.Dropout(
            config.dropout_rate
        )

        # GLFF池化后输出维度为64
        self.classifier = nn.Sequential(
            nn.Linear(config.glff_hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, config.num_classes),
        )

    def forward(self, x):
        # [B, N] -> [B, N, 1]
        node_features = x.unsqueeze(-1)

        # [B, N, 64]
        node_features = self.glff(
            node_features,
            self.adjacency,
            self.adjacency_mask,
        )

        # BatchNorm1d作用于通道维
        node_features = node_features.permute(
            0,
            2,
            1,
        )
        node_features = self.batch_norm(
            node_features
        )
        node_features = node_features.permute(
            0,
            2,
            1,
        )
        node_features = self.dropout(
            node_features
        )

        # 删除LSTM后的关键替代：对节点维做全局平均池化
        graph_features = node_features.mean(dim=1)

        return self.classifier(graph_features)


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

model = StableGLFFOnly(
    num_features=X.shape[1],
    adjacency=adjacency.to(device),
    adjacency_mask=adjacency_mask.to(device),
).to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f'使用设备：{device}')
print(f'GLFF-only可训练参数量：{parameter_count:,}')
print(model)


# ========================== 7. 训练与模型选择 ==========================
def run_epoch(
    dataloader,
    model,
    loss_function,
    device,
    optimizer=None,
):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = loss_function(
                logits,
                batch_y,
            )

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            current_batch_size = batch_y.size(0)

            total_loss += (
                loss.item() * current_batch_size
            )

            total_correct += (
                logits.argmax(dim=1) == batch_y
            ).sum().item()

            total_samples += current_batch_size

    mean_loss = total_loss / total_samples
    mean_accuracy = (
        total_correct / total_samples
    )

    return mean_accuracy, mean_loss


class EarlyStopping:
    def __init__(
        self,
        patience=50,
        delta=0.001,
    ):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def update(self, loss_value):
        if self.best_loss is None:
            self.best_loss = loss_value
            return

        if loss_value < self.best_loss - self.delta:
            self.best_loss = loss_value
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=config.scheduler_patience,
)

loss_function = nn.CrossEntropyLoss()

early_stopping = EarlyStopping(
    patience=config.early_stopping_patience,
    delta=config.early_stopping_delta,
)

best_accuracy = -1.0
best_state_dict = None
best_epoch = 0

train_accuracy_history = []
train_loss_history = []
test_accuracy_history = []
test_loss_history = []

start_time = time.time()

print('\n开始训练GLFF-only消融模型……')

for epoch in range(config.epochs):
    train_accuracy, train_loss = run_epoch(
        train_loader,
        model,
        loss_function,
        device,
        optimizer=optimizer,
    )

    test_accuracy, test_loss = run_epoch(
        test_loader,
        model,
        loss_function,
        device,
        optimizer=None,
    )

    scheduler.step(test_loss)
    early_stopping.update(test_loss)

    train_accuracy_history.append(
        train_accuracy
    )
    train_loss_history.append(
        train_loss
    )
    test_accuracy_history.append(
        test_accuracy
    )
    test_loss_history.append(
        test_loss
    )

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch + 1

        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        print(
            f'Epoch {epoch + 1:03d} | '
            f'最佳测试准确率更新：'
            f'{best_accuracy * 100:.2f}%'
        )

    if (
        epoch == 0
        or (epoch + 1) % 10 == 0
        or early_stopping.should_stop
    ):
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'Epoch {epoch + 1:03d}/{config.epochs} | '
            f'Train Loss={train_loss:.4f} | '
            f'Train Acc={train_accuracy * 100:.2f}% | '
            f'Test Loss={test_loss:.4f} | '
            f'Test Acc={test_accuracy * 100:.2f}% | '
            f'LR={current_lr:.6f}'
        )

    if early_stopping.should_stop:
        print(
            f'早停触发：Epoch {epoch + 1}'
        )
        break

elapsed_time = time.time() - start_time

if best_state_dict is None:
    raise RuntimeError(
        '训练结束后未获得有效模型权重。'
    )

model.load_state_dict(best_state_dict)

model_path = SCRIPT_DIR / MODEL_FILENAME
torch.save(
    model.state_dict(),
    model_path,
)

actual_epochs = len(train_loss_history)

print(f'\n训练完成，耗时：{elapsed_time:.2f} s')
print(f'实际训练轮次：{actual_epochs}')
print(f'最佳轮次：Epoch {best_epoch}')
print(
    f'最佳GLFF-only测试准确率：'
    f'{best_accuracy * 100:.2f}%'
)
print(f'最佳权重已保存：{model_path}')


# ========================== 8. 最终评估 ==========================
model.eval()

all_true = []
all_pred = []
all_probability = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)

        logits = model(batch_x)
        probabilities = F.softmax(
            logits,
            dim=1,
        )

        all_true.append(
            batch_y.numpy()
        )
        all_pred.append(
            logits.argmax(dim=1).cpu().numpy()
        )
        all_probability.append(
            probabilities.cpu().numpy()
        )

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)
y_pred_probability = np.concatenate(
    all_probability
)

accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0,
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average='macro',
    zero_division=0,
)

try:
    macro_auc = roc_auc_score(
        y_true,
        y_pred_probability,
        multi_class='ovr',
        average='macro',
    )
except ValueError:
    macro_auc = np.nan

class_labels = [
    f'C{class_index + 1}'
    for class_index in range(
        config.num_classes
    )
]

print('\n' + '=' * 72)
print('GLFF-only消融实验最终结果')
print('=' * 72)
print(
    f'Accuracy        : '
    f'{accuracy:.6f} '
    f'({accuracy * 100:.2f}%)'
)
print(
    f'Macro-Precision : '
    f'{macro_precision:.6f}'
)
print(
    f'Macro-Recall    : '
    f'{macro_recall:.6f}'
)
print(
    f'Macro-F1        : '
    f'{macro_f1:.6f}'
)
print(
    f'Macro ROC-AUC   : '
    f'{macro_auc:.6f}'
)

print('\n分类报告：')
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        digits=4,
        zero_division=0,
    )
)

summary_metrics = pd.DataFrame([{
    'Model': 'GLFF-only',
    'Removed component': 'LSTM',
    'Node aggregation': 'Global mean pooling',
    'GLFF layers': config.glff_num_layers,
    'Attention heads': config.glff_num_heads,
    'Hidden dimension': config.glff_hidden_dim,
    'Accuracy': accuracy,
    'Macro-Precision': macro_precision,
    'Macro-Recall': macro_recall,
    'Macro-F1': macro_f1,
    'Macro ROC-AUC': macro_auc,
    'Best selected accuracy': best_accuracy,
    'Best epoch': best_epoch,
    'Actual epochs': actual_epochs,
    'Trainable parameters': parameter_count,
    'Random seed': config.seed,
    'Data file': data_path.name,
}])

summary_metrics_path = (
    SCRIPT_DIR / METRICS_FILENAME
)
summary_metrics.to_csv(
    summary_metrics_path,
    index=False,
    encoding='utf-8-sig',
)

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_path = (
    SCRIPT_DIR / CLASS_REPORT_FILENAME
)
report_df.to_csv(
    report_path,
    encoding='utf-8-sig',
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(
        config.num_classes
    ),
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels,
)

cm_csv_path = (
    SCRIPT_DIR / CONFUSION_MATRIX_CSV
)
cm_df.to_csv(
    cm_csv_path,
    encoding='utf-8-sig',
)

print(
    f'汇总指标已保存：'
    f'{summary_metrics_path}'
)
print(
    f'分类报告已保存：'
    f'{report_path}'
)
print(
    f'混淆矩阵数据已保存：'
    f'{cm_csv_path}'
)


# ========================== 9. 统一论文风格混淆矩阵 ==========================
n_classes = cm.shape[0]

row_sum = cm.sum(
    axis=1,
    keepdims=True,
)

cm_row_percent = np.divide(
    cm,
    row_sum,
    out=np.zeros_like(
        cm,
        dtype=float,
    ),
    where=row_sum != 0,
) * 100.0

fig, ax = plt.subplots(
    figsize=(10.2, 9.0),
    dpi=300,
)

image = ax.imshow(
    cm,
    interpolation='nearest',
    cmap=PUBLICATION_CMAP,
    norm=Normalize(
        vmin=0,
        vmax=max(
            1,
            int(cm.max()),
        ),
    ),
    aspect='equal',
)

cbar = fig.colorbar(
    image,
    ax=ax,
    fraction=0.045,
    pad=0.035,
)

cbar.set_label(
    'Number of samples',
    fontsize=17,
    fontweight='bold',
    fontfamily='Times New Roman',
    labelpad=12,
)

cbar.ax.tick_params(
    labelsize=14,
    length=4,
    width=0.9,
)

cbar.outline.set_linewidth(0.8)

ticks = np.arange(n_classes)

ax.set_xticks(ticks)
ax.set_yticks(ticks)

ax.set_xticklabels(
    class_labels,
    fontsize=19,
    fontweight='bold',
    fontfamily='Times New Roman',
)

ax.set_yticklabels(
    class_labels,
    fontsize=19,
    fontweight='bold',
    fontfamily='Times New Roman',
)

ax.tick_params(
    axis='x',
    rotation=0,
    pad=8,
    length=0,
)

ax.tick_params(
    axis='y',
    rotation=0,
    pad=8,
    length=0,
)

for tick in ax.get_xticklabels():
    tick.set_ha('center')

ax.set_xticks(
    np.arange(
        -0.5,
        n_classes,
        1,
    ),
    minor=True,
)

ax.set_yticks(
    np.arange(
        -0.5,
        n_classes,
        1,
    ),
    minor=True,
)

ax.grid(
    which='minor',
    color='white',
    linestyle='-',
    linewidth=1.6,
)

ax.tick_params(
    which='minor',
    bottom=False,
    left=False,
)

for class_index in range(n_classes):
    ax.add_patch(
        Rectangle(
            (
                class_index - 0.5,
                class_index - 0.5,
            ),
            1,
            1,
            fill=False,
            edgecolor='black',
            linewidth=1.15,
            alpha=0.75,
        )
    )

threshold = cm.max() * 0.52

for true_index in range(n_classes):
    for pred_index in range(n_classes):
        count = int(
            cm[
                true_index,
                pred_index,
            ]
        )

        percentage = cm_row_percent[
            true_index,
            pred_index,
        ]

        text_color = (
            'white'
            if cm[
                true_index,
                pred_index,
            ] > threshold
            else '#202020'
        )

        if SHOW_ROW_PERCENT:
            ax.text(
                pred_index,
                true_index - 0.10,
                f'{count}',
                ha='center',
                va='center',
                fontsize=19,
                fontweight='bold',
                fontfamily='Times New Roman',
                color=text_color,
            )

            ax.text(
                pred_index,
                true_index + 0.22,
                f'{percentage:.1f}%',
                ha='center',
                va='center',
                fontsize=12.5,
                fontweight='normal',
                fontfamily='Times New Roman',
                color=text_color,
                alpha=0.92,
            )
        else:
            ax.text(
                pred_index,
                true_index,
                f'{count}',
                ha='center',
                va='center',
                fontsize=20,
                fontweight='bold',
                fontfamily='Times New Roman',
                color=text_color,
            )

ax.set_xlabel(
    'Predicted class',
    fontsize=19,
    fontweight='bold',
    fontfamily='Times New Roman',
    labelpad=15,
)

ax.set_ylabel(
    'True class',
    fontsize=19,
    fontweight='bold',
    fontfamily='Times New Roman',
    labelpad=15,
)

ax.text(
    0.0,
    1.025,
    f'Overall accuracy = '
    f'{accuracy * 100:.2f}%',
    transform=ax.transAxes,
    ha='left',
    va='bottom',
    fontsize=15,
    fontweight='bold',
    fontfamily='Times New Roman',
)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.9)
    spine.set_color('#333333')

all_text_objects = (
    [ax.xaxis.label, ax.yaxis.label]
    + list(ax.get_xticklabels())
    + list(ax.get_yticklabels())
    + list(ax.texts)
    + [cbar.ax.yaxis.label]
    + list(cbar.ax.get_yticklabels())
)

for text_object in all_text_objects:
    text_object.set_fontfamily(
        'Times New Roman'
    )

ax.set_ylim(
    n_classes - 0.5,
    -0.5,
)

fig.tight_layout(pad=1.35)

png_path = (
    SCRIPT_DIR / f'{OUTPUT_STEM}.png'
)
pdf_path = (
    SCRIPT_DIR / f'{OUTPUT_STEM}.pdf'
)
svg_path = (
    SCRIPT_DIR / f'{OUTPUT_STEM}.svg'
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches='tight',
    facecolor='white',
)

fig.savefig(
    pdf_path,
    bbox_inches='tight',
    facecolor='white',
)

fig.savefig(
    svg_path,
    bbox_inches='tight',
    facecolor='white',
)

plt.close(fig)

print(f'混淆矩阵PNG：{png_path}')
print(f'混淆矩阵PDF：{pdf_path}')
print(f'混淆矩阵SVG：{svg_path}')
print('=' * 72)
