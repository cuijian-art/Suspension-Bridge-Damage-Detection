import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from data_process import load_data, CLASS_LIST
from models import MultiLevelFusionModel

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ====================== 你原有的训练参数完全保留 ======================
EPOCHS = 500
LR = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL_PATH = "./best_paper_model2-99%.pth"
# 新增：DDC损失权重，类别不平衡可调到1.0
ALPHA = 0.5


# ======================================================

# ====================== 新增：DDC深度散度聚类损失 ======================
class DDCLoss(nn.Module):
    def __init__(self, num_classes, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.eps = 1e-8

    def compute_kernel(self, x):
        B, D = x.shape
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        dist_sq = x_norm + x_norm.T - 2 * torch.mm(x, x.T)
        bandwidth = torch.median(dist_sq[dist_sq > 0])
        kernel = torch.exp(-dist_sq / (bandwidth + self.eps))
        return kernel

    def forward(self, features, logits, labels):
        B, D = features.shape
        soft_assign = F.softmax(logits, dim=-1)
        kernel = self.compute_kernel(features)

        # 类间可分性
        class_weights = soft_assign / (soft_assign.sum(dim=0, keepdim=True) + self.eps)
        class_kernel = class_weights.T @ kernel @ class_weights
        dcs = 0.0
        count = 0
        for i in range(self.num_classes):
            for j in range(i + 1, self.num_classes):
                cs_dist = class_kernel[i, j] / torch.sqrt(class_kernel[i, i] * class_kernel[j, j] + self.eps)
                dcs += -torch.log(cs_dist + self.eps)
                count += 1
        dcs = dcs / count if count > 0 else 0.0

        # 类内紧凑性
        dgs = 0.0
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() <= 1:
                continue
            class_feat = features[mask]
            class_mean = class_feat.mean(dim=0, keepdim=True)
            dgs += torch.mean(torch.norm(class_feat - class_mean, p=2, dim=-1))
        dgs = dgs / self.num_classes

        # 类别正交性
        assign_corr = soft_assign.T @ soft_assign
        deo = torch.sum(torch.triu(assign_corr, diagonal=1))

        total_ddc = dcs + 0.5 * dgs + 0.1 * deo
        return total_ddc


class TotalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, device='cuda'):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.ddc_loss = DDCLoss(num_classes, device)
        self.alpha = alpha

    def forward(self, logits, labels, features):
        ce = self.ce_loss(logits, labels)
        ddc = self.ddc_loss(features, logits, labels) if self.alpha > 0 else 0.0
        total = ce + self.alpha * ddc
        return total, ce, ddc


# ====================== 原有熵决策融合逻辑完全保留 ======================
def entropy_decision_fusion(cnn_logits, gcn_logits, num_classes):
    batch_size = cnn_logits.shape[0]
    n_sensor = cnn_logits.shape[1]
    all_logits = torch.cat([cnn_logits, gcn_logits], dim=1)
    all_probs = F.softmax(all_logits, dim=-1)

    entropy = -torch.sum(all_probs * torch.log(all_probs + 1e-8), dim=-1)
    weight = F.softmax(1 / (entropy + 1e-8), dim=-1).unsqueeze(-1)
    final_probs = torch.sum(weight * all_probs, dim=1)
    return final_probs, all_logits


def train():
    x, y, edge_index, edge_weight, train_mask, val_mask, test_mask, n_per_sensor, n_sensor, num_classes = load_data()
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)
    sample_y = y.to(DEVICE).long()

    model = MultiLevelFusionModel(n_sensor=n_sensor, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=15, factor=0.5)
    # 替换原有交叉熵损失为带DDC的总损失
    criterion = TotalLoss(num_classes=num_classes, alpha=ALPHA, device=DEVICE)

    best_val_acc = 0
    early_stop_count = 0
    train_loss_list = []
    val_acc_list = []

    print(f"开始训练，设备：{DEVICE}，总样本数：{len(x)}")
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        optimizer.zero_grad()
        fused_feat, cnn_logits, gcn_logits, _ = model(x, edge_index, edge_weight, n_per_sensor)

        final_probs, all_logits = entropy_decision_fusion(cnn_logits, gcn_logits, num_classes)
        # 替换原有损失计算，保留辅助损失
        loss_final, ce_loss, ddc_loss = criterion(
            logits=final_probs[train_mask],
            labels=sample_y[train_mask],
            features=fused_feat[train_mask]
        )
        loss_cnn = torch.mean(torch.stack(
            [criterion.ce_loss(cnn_logits[train_mask, s, :], sample_y[train_mask]) for s in range(n_sensor)]))
        loss_gcn = torch.mean(torch.stack(
            [criterion.ce_loss(gcn_logits[train_mask, s, :], sample_y[train_mask]) for s in range(n_sensor)]))
        loss = loss_final + 0.3 * loss_cnn + 0.3 * loss_gcn

        loss.backward()
        optimizer.step()

        # 验证逻辑完全保留
        model.eval()
        with torch.no_grad():
            fused_feat_val, cnn_logits_val, gcn_logits_val, _ = model(x, edge_index, edge_weight, n_per_sensor)
            final_probs_val, _ = entropy_decision_fusion(cnn_logits_val, gcn_logits_val, num_classes)
            val_pred = final_probs_val.argmax(dim=1)[val_mask]
            val_true = sample_y[val_mask]
            val_acc = accuracy_score(val_true.cpu().numpy(), val_pred.cpu().numpy())
        scheduler.step(val_acc)

        train_loss_list.append(loss.item())
        val_acc_list.append(val_acc)

        # 保留：验证精度更高时，保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
        # 删除：所有早停相关逻辑（不计数、不提前停止）

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch:03d} | 总损失：{loss.item():.4f} | CE损失：{ce_loss.item():.4f} | DDC损失：{ddc_loss.item():.4f} | 验证精度：{val_acc:.4f}")

    # 测试逻辑完全保留
    print("开始测试...")
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        fused_feat_test, cnn_logits_test, gcn_logits_test, attn_weight = model(x, edge_index, edge_weight, n_per_sensor)
        final_probs_test, _ = entropy_decision_fusion(cnn_logits_test, gcn_logits_test, num_classes)
        test_pred = final_probs_test.argmax(dim=1)[test_mask].cpu().numpy()
        test_true = sample_y[test_mask].cpu().numpy()

        acc = accuracy_score(test_true, test_pred)
        f1 = f1_score(test_true, test_pred, average='macro')
        print(f"\n===== 最终融合结果 =====")
        print(f"精度：{acc:.4f} | F1：{f1:.4f}")

        print(f"\n===== 各传感器单独CNN结果 =====")
        cnn_acc_list = []
        for s in range(n_sensor):
            s_pred = cnn_logits_test[:, s, :].argmax(dim=1)[test_mask].cpu().numpy()
            s_acc = accuracy_score(test_true, s_pred)
            cnn_acc_list.append(s_acc)
            print(f"传感器{s + 1} CNN精度：{s_acc:.4f}")
        print(f"\n===== 各传感器单独GCN结果 =====")
        gcn_acc_list = []
        for s in range(n_sensor):
            s_pred = gcn_logits_test[:, s, :].argmax(dim=1)[test_mask].cpu().numpy()
            s_acc = accuracy_score(test_true, s_pred)
            gcn_acc_list.append(s_acc)
            print(f"传感器{s + 1} GCN精度：{s_acc:.4f}")

    plt.figure(figsize=(12, 6))
    x_ax = np.arange(n_sensor)
    width = 0.35
    plt.bar(x_ax - width / 2, cnn_acc_list, width, label='单传感器CNN', color='#1f77b4')
    plt.bar(x_ax + width / 2, gcn_acc_list, width, label='单传感器GCN', color='#ff7f0e')
    plt.axhline(y=acc, color='#2ca02c', linestyle='--', linewidth=2, label=f'多传感器融合最终精度={acc:.4f}')
    plt.xticks(x_ax, [f'传感器{i + 1}' for i in range(n_sensor)])
    plt.ylabel("分类精度")
    plt.title("单传感器 vs 多传感器融合精度对比")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig("./sensor_acc_compare_paper.png", dpi=300, bbox_inches='tight')
    print("\n精度对比图已保存为：sensor_acc_compare_paper.png")

    cm = confusion_matrix(test_true, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LIST, yticklabels=CLASS_LIST)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("融合结果混淆矩阵")
    plt.savefig("./confusion_matrix_paper.png", dpi=300, bbox_inches='tight')
    print("混淆矩阵已保存为：confusion_matrix_paper.png")


if __name__ == '__main__':
    train()