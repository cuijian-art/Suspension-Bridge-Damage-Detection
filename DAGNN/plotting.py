from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def save_history(history, output_dir):
    path = Path(output_dir) / "training_history.csv"
    keys = list(history.keys())
    row_count = max((len(history[key]) for key in keys), default=0)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", *keys])
        writer.writeheader()
        for index in range(row_count):
            row = {"epoch": index}
            row.update({key: history[key][index] if index < len(history[key]) else "" for key in keys})
            writer.writerow(row)


def plot_training_curves(history, output_dir):
    output_dir = Path(output_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], lw=2.2, color="#264653", label="Training")
    axes[0].plot(epochs, history["target_validation_loss"], lw=2.2, color="#E76F51", label="Target validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes[1].plot(epochs, history["train_accuracy"], lw=2.2, color="#2A9D8F", label="Source training")
    axes[1].plot(epochs, history["source_validation_accuracy"], lw=2.2, color="#457B9D", label="Source validation")
    axes[1].plot(epochs, history["target_validation_accuracy"], lw=2.2, color="#E76F51", label="Target validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Classification accuracy")
    for axis in axes:
        axis.grid(True, color="#D7DCE2", linestyle="--", linewidth=0.8)
        axis.legend(frameon=False)
        axis.tick_params(labelsize=11)
    fig.savefig(output_dir / "training_curves.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_adaptation_curves(history, output_dir):
    if not history["domain_loss"] and not history["mmd_loss"]:
        return
    output_dir = Path(output_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    if history["domain_loss"]:
        axes[0].plot(epochs, history["domain_loss"], lw=2.2, color="#7B2CBF", label="Domain loss")
        axes[0].plot(epochs, history["domain_accuracy"], lw=2.2, color="#2A9D8F", label="Domain accuracy (%)")
    if history["mmd_loss"]:
        axes[1].plot(epochs, history["mmd_loss"], lw=2.2, color="#D1495B", label="MMD")
    axes[0].set(xlabel="Epoch", title="Domain-adversarial learning")
    axes[1].set(xlabel="Epoch", ylabel="MMD", title="Distribution discrepancy")
    for axis in axes:
        axis.grid(True, color="#D7DCE2", linestyle="--", linewidth=0.8)
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False)
        axis.tick_params(labelsize=11)
    fig.savefig(output_dir / "adaptation_curves.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "adaptation_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, class_names, output_dir):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))).astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    size = max(7.0, 0.72 * len(class_names) + 3.0)
    fig, axis = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    sns.heatmap(
        matrix,
        ax=axis,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        square=True,
        annot=True,
        fmt=".2f",
        linewidths=0.45,
        linecolor="white",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized proportion", "shrink": 0.82, "pad": 0.03},
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Target-domain confusion matrix")
    axis.tick_params(axis="x", rotation=45, labelsize=10)
    axis.tick_params(axis="y", rotation=0, labelsize=10)
    output_dir = Path(output_dir)
    fig.savefig(output_dir / "confusion_matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "confusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tsne(source_features, target_features, stage, output_dir, seed=42):
    source = np.asarray(source_features)
    target = np.asarray(target_features)
    combined = np.concatenate((source, target), axis=0)
    if combined.shape[0] < 4:
        return
    perplexity = min(30, max(2, (combined.shape[0] - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        n_iter=1000,
    ).fit_transform(combined)
    fig, axis = plt.subplots(figsize=(8.4, 7.0), constrained_layout=True)
    axis.set_facecolor("#F2F3F5")
    source_count = len(source)
    axis.scatter(
        embedding[:source_count, 0],
        embedding[:source_count, 1],
        s=42,
        alpha=0.72,
        color="#31688E",
        marker="o",
        edgecolors="white",
        linewidths=0.25,
        label="Source domain",
    )
    axis.scatter(
        embedding[source_count:, 0],
        embedding[source_count:, 1],
        s=46,
        alpha=0.72,
        color="#D95F59",
        marker="^",
        edgecolors="white",
        linewidths=0.25,
        label="Target domain",
    )
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    axis.set_title(f"Feature distribution: {stage}")
    axis.grid(True, color="white", linewidth=1.0)
    axis.legend(frameon=True, facecolor="white", edgecolor="#B7BCC5")
    axis.tick_params(labelsize=11)
    output_dir = Path(output_dir)
    fig.savefig(output_dir / f"tsne_{stage}.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"tsne_{stage}.pdf", bbox_inches="tight")
    plt.close(fig)
