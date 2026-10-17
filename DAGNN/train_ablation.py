from __future__ import annotations

import argparse
import csv
import gc
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cli import add_training_arguments, build_config, configure_logging
from trainer import Trainer


VARIANTS = (
    {"slug": "full_dagnn", "name": "Full DAGNN", "use_gcn": True, "use_adv": True, "use_mmd": True},
    {"slug": "without_adv", "name": "w/o ADV", "use_gcn": True, "use_adv": False, "use_mmd": True},
    {"slug": "without_mmd", "name": "w/o MMD", "use_gcn": True, "use_adv": True, "use_mmd": False},
    {"slug": "without_gcn", "name": "w/o GCN", "use_gcn": False, "use_adv": True, "use_mmd": True},
    {"slug": "without_adv_mmd", "name": "w/o ADV & MMD", "use_gcn": True, "use_adv": False, "use_mmd": False},
)


def save_summary(results, output_dir):
    output_dir = Path(output_dir)
    with (output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    labels = [row["variant"] for row in results]
    values = np.asarray([row["best_target_validation_accuracy_percent"] for row in results])
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(9.4, 5.5), constrained_layout=True)
    bars = axis.bar(np.arange(len(values)), values, width=0.64, color="#3D5A80")
    axis.set_xticks(np.arange(len(values)), labels, rotation=12, ha="right")
    axis.set_ylabel("Best target-domain accuracy (%)")
    axis.set_title("DAGNN ablation study")
    axis.set_ylim(max(0.0, float(values.min()) - 8.0), 101.5)
    axis.grid(axis="y", color="#D7DCE2", linestyle="--", linewidth=0.8)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontweight="bold")
    fig.savefig(output_dir / "ablation_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "ablation_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = add_training_arguments(argparse.ArgumentParser())
    parser.set_defaults(target="MovingTrolleyExcitation")
    parser.add_argument("--only", nargs="*", choices=[variant["slug"] for variant in VARIANTS])
    args = parser.parse_args()
    selected = [variant for variant in VARIANTS if not args.only or variant["slug"] in args.only]
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    experiment_dir = Path(args.output_dir) / f"DAGNN_ablation_{args.source}_to_{args.target}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for variant in selected:
        task_dir = experiment_dir / variant["slug"]
        configure_logging(experiment_dir / f"{variant['slug']}.log")
        config = build_config(
            args,
            args.source,
            args.target,
            task_dir,
            variant_name=variant["name"],
            use_gcn=variant["use_gcn"],
            use_adv=variant["use_adv"],
            use_mmd=variant["use_mmd"],
        )
        trainer = Trainer(config)
        trainer.setup()
        results.append(trainer.train())
        save_summary(results, experiment_dir)
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(experiment_dir.resolve())


if __name__ == "__main__":
    main()
