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

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

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

    # 保持与原GCN-LSTM代码一致
    gnn_hidden_dim = 64
    dropout_rate = 0.3
    k_neighbors = 5
    corr_threshold = 0.3

    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    epochs = 100
    test_size = 0.30
    random_state = 42

    num_classes = None


config = Config()

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILENAME = 'wine.xlsx'

MODEL_FILENAME = 'best_gcn_only_ablation_model.pth'
METRICS_FILENAME = 'GCN_only_ablation_metrics.csv'
CLASS_REPORT_FILENAME = 'GCN_only_ablation_classification_report.csv'
CONFUSION_MATRIX_CSV = 'GCN_only_ablation_confusion_matrix.csv'
OUTPUT_STEM = 'GCN_only_ablation_confusion_matrix_TNR_ocean_C1C6'


def set_seed(seed):
    """固定随机种子，增强消融实验的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
data_path = SCRIPT_DIR / DATA_FILENAME
if not data_path.exists():
    data_path = Path.cwd() / DATA_FILENAME

if not data_path.exists():
    raise FileNotFoundError(
        f'未找到 {DATA_FILENAME}，请将数据文件与脚本放在同一目录。'
    )

data = pd.read_excel(data_path, index_col=0)

X = data.iloc[:, :-1].values
raw_y = data.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(raw_y)

config.num_classes = len(np.unique(y))

print('=' * 70)
print('GCN-only 消融实验：删除LSTM，仅保留两层GCN')
print('=' * 70)
print(f'数据文件：{data_path}')
print(f'样本数：{X.shape[0]}')
print(f'特征数/图节点数：{X.shape[1]}')
print(f'类别数：{config.num_classes}')
print(f'原始标签：{list(label_encoder.classes_)}')

# 为了和原GCN-LSTM实验保持完全一致，这里沿用原代码：
# 先对全体数据进行标准化，再进行70%/30%划分。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=config.test_size,
    random_state=config.random_state,
    stratify=y,
)


# ========================== 3. 数据集与图结构 ==========================
class BridgeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        y_value = torch.as_tensor(self.labels[index], dtype=torch.long)
        return x, y_value


train_dataset = BridgeDataset(X_train, y_train)
test_dataset = BridgeDataset(X_test, y_test)

# 使用固定生成器，使训练批次顺序可复现
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


def build_correlation_adjacency(feature_data, k_neighbors=5, threshold=0.3):
    """按原GCN-LSTM代码，用训练集特征相关性构建图。"""
    n_features = feature_data.shape[1]

    corr_matrix = np.corrcoef(feature_data.T)
    np.fill_diagonal(corr_matrix, 0)

    adjacency = np.zeros((n_features, n_features), dtype=np.float32)

    for i in range(n_features):
        top_k_indices = np.argsort(np.abs(corr_matrix[i]))[-k_neighbors:]

        for j in top_k_indices:
            if abs(corr_matrix[i, j]) >= threshold:
                adjacency[i, j] = 1.0

    np.fill_diagonal(adjacency, 1.0)

    edge_index = torch.as_tensor(
        np.array(np.where(adjacency == 1)),
        dtype=torch.long,
    )

    print(
        f'邻接矩阵：平均度数={adjacency.sum() / n_features:.2f}，'
        f'边数={edge_index.shape[1]}'
    )

    return edge_index


edge_index = build_correlation_adjacency(
    X_train,
    k_neighbors=config.k_neighbors,
    threshold=config.corr_threshold,
)


# ========================== 4. GCN-only消融模型 ==========================
class StableGCNOnly(nn.Module):
    """
    GCN-LSTM的消融版本。

    保留：
    1. 两层GCNConv；
    2. BatchNorm；
    3. ReLU与Dropout；
    4. 64维分类头。

    删除：
    1. LSTM层；
    2. 最后时间步特征提取。

    替代：
    对每个样本图的全部特征节点做全局平均池化，
    得到一个64维图级表示，再送入分类头。
    """

    def __init__(self, num_features, edge_index):
        super().__init__()

        self.num_features = num_features
        self.register_buffer('edge_index', edge_index)

        self.gcn1 = GCNConv(1, config.gnn_hidden_dim)
        self.gcn2 = GCNConv(
            config.gnn_hidden_dim,
            config.gnn_hidden_dim,
        )

        self.bn1 = nn.BatchNorm1d(config.gnn_hidden_dim)
        self.bn2 = nn.BatchNorm1d(config.gnn_hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

        # 删除LSTM后，池化输出维度为gnn_hidden_dim=64
        self.fc = nn.Sequential(
            nn.Linear(config.gnn_hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(config.dropout_rate),
            nn.Linear(64, config.num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        graph_list = []
        for sample_index in range(batch_size):
            node_features = x[sample_index].unsqueeze(1)

            graph = Data(
                x=node_features,
                edge_index=self.edge_index,
            )
            graph_list.append(graph)

        batch_graph = Batch.from_data_list(graph_list)

        node_features = self.gcn1(
            batch_graph.x,
            batch_graph.edge_index,
        )
        node_features = F.relu(node_features)
        node_features = self.bn1(node_features)
        node_features = self.dropout(node_features)

        node_features = self.gcn2(
            node_features,
            batch_graph.edge_index,
        )
        node_features = F.relu(node_features)
        node_features = self.bn2(node_features)
        node_features = self.dropout(node_features)

        # [batch_size * num_features, hidden_dim]
        # -> [batch_size, num_features, hidden_dim]
        node_features = node_features.view(
            batch_size,
            self.num_features,
            config.gnn_hidden_dim,
        )

        # 删除LSTM后的关键替代：节点维全局平均池化
        graph_features = node_features.mean(dim=1)

        return self.fc(graph_features)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
edge_index = edge_index.to(device)

model = StableGCNOnly(
    num_features=X.shape[1],
    edge_index=edge_index,
).to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f'使用设备：{device}')
print(f'GCN-only可训练参数量：{parameter_count:,}')
print(model)


# ========================== 5. 训练与模型选择 ==========================
def run_epoch(dataloader, model, loss_fn, device, optimizer=None):
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                logits.argmax(dim=1) == batch_y
            ).sum().item()
            total_samples += batch_size

    mean_loss = total_loss / total_samples
    mean_accuracy = total_correct / total_samples

    return mean_accuracy, mean_loss


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=20,
)

loss_fn = nn.CrossEntropyLoss()

best_accuracy = -1.0
best_state_dict = None

train_accuracy_history = []
train_loss_history = []
test_accuracy_history = []
test_loss_history = []

start_time = time.time()

print('\n开始训练GCN-only消融模型……')

for epoch in range(config.epochs):
    train_accuracy, train_loss = run_epoch(
        train_loader,
        model,
        loss_fn,
        device,
        optimizer=optimizer,
    )

    test_accuracy, test_loss = run_epoch(
        test_loader,
        model,
        loss_fn,
        device,
        optimizer=None,
    )

    scheduler.step(test_loss)

    train_accuracy_history.append(train_accuracy)
    train_loss_history.append(train_loss)
    test_accuracy_history.append(test_accuracy)
    test_loss_history.append(test_loss)

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        print(
            f'Epoch {epoch + 1:02d} | '
            f'最佳测试准确率更新：{best_accuracy * 100:.2f}%'
        )

    print(
        f'Epoch {epoch + 1:02d}/{config.epochs} | '
        f'Train Loss={train_loss:.4f} | '
        f'Train Acc={train_accuracy * 100:.2f}% | '
        f'Test Loss={test_loss:.4f} | '
        f'Test Acc={test_accuracy * 100:.2f}%'
    )

elapsed_time = time.time() - start_time

if best_state_dict is None:
    raise RuntimeError('训练结束后未获得有效模型权重。')

model.load_state_dict(best_state_dict)

model_path = SCRIPT_DIR / MODEL_FILENAME
torch.save(model.state_dict(), model_path)

print(f'\n训练完成，耗时：{elapsed_time:.2f} s')
print(f'最佳GCN-only测试准确率：{best_accuracy * 100:.2f}%')
print(f'最佳权重已保存：{model_path}')


# ========================== 6. 最终评估 ==========================
model.eval()

all_true = []
all_pred = []
all_probability = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)

        logits = model(batch_x)
        probabilities = F.softmax(logits, dim=1)

        all_true.append(batch_y.numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_probability.append(probabilities.cpu().numpy())

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)
y_pred_probability = np.concatenate(all_probability)

accuracy = accuracy_score(y_true, y_pred)
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
    for class_index in range(config.num_classes)
]

print('\n' + '=' * 70)
print('GCN-only消融实验最终结果')
print('=' * 70)
print(f'Accuracy        : {accuracy:.6f} ({accuracy * 100:.2f}%)')
print(f'Macro-Precision : {macro_precision:.6f}')
print(f'Macro-Recall    : {macro_recall:.6f}')
print(f'Macro-F1        : {macro_f1:.6f}')
print(f'Macro ROC-AUC   : {macro_auc:.6f}')

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
    'Model': 'GCN-only',
    'Removed component': 'LSTM',
    'Pooling': 'Global mean pooling',
    'Accuracy': accuracy,
    'Macro-Precision': macro_precision,
    'Macro-Recall': macro_recall,
    'Macro-F1': macro_f1,
    'Macro ROC-AUC': macro_auc,
    'Best selected accuracy': best_accuracy,
    'Trainable parameters': parameter_count,
    'Epochs': config.epochs,
    'Random seed': config.seed,
}])

summary_metrics_path = SCRIPT_DIR / METRICS_FILENAME
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

report_df = pd.DataFrame(report_dict).transpose()
report_path = SCRIPT_DIR / CLASS_REPORT_FILENAME
report_df.to_csv(
    report_path,
    encoding='utf-8-sig',
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(config.num_classes),
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels,
)
cm_csv_path = SCRIPT_DIR / CONFUSION_MATRIX_CSV
cm_df.to_csv(
    cm_csv_path,
    encoding='utf-8-sig',
)

print(f'汇总指标已保存：{summary_metrics_path}')
print(f'分类报告已保存：{report_path}')
print(f'混淆矩阵数据已保存：{cm_csv_path}')


# ========================== 7. 统一论文风格混淆矩阵 ==========================
n_classes = cm.shape[0]

row_sum = cm.sum(axis=1, keepdims=True)
cm_row_percent = np.divide(
    cm,
    row_sum,
    out=np.zeros_like(cm, dtype=float),
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
        vmax=max(1, int(cm.max())),
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
    np.arange(-0.5, n_classes, 1),
    minor=True,
)
ax.set_yticks(
    np.arange(-0.5, n_classes, 1),
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
            (class_index - 0.5, class_index - 0.5),
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
        count = int(cm[true_index, pred_index])
        percentage = cm_row_percent[
            true_index,
            pred_index,
        ]

        text_color = (
            'white'
            if cm[true_index, pred_index] > threshold
            else '#202020'
        )

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
    f'Overall accuracy = {accuracy * 100:.2f}%',
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
    text_object.set_fontfamily('Times New Roman')

ax.set_ylim(n_classes - 0.5, -0.5)
fig.tight_layout(pad=1.35)

png_path = SCRIPT_DIR / f'{OUTPUT_STEM}.png'
pdf_path = SCRIPT_DIR / f'{OUTPUT_STEM}.pdf'
svg_path = SCRIPT_DIR / f'{OUTPUT_STEM}.svg'

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
print('=' * 70)
