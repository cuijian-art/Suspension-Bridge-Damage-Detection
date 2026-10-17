from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_loader import DOMAIN_NAMES
from trainer import TrainingConfig


def add_training_arguments(parser, include_domains=True):
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./checkpoint")
    if include_domains:
        parser.add_argument("--source", choices=DOMAIN_NAMES, default=DOMAIN_NAMES[0])
        parser.add_argument("--target", choices=DOMAIN_NAMES, default=DOMAIN_NAMES[1])
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--normalization", choices=["mean-std", "0-1", "-1-1", "none"], default="mean-std")
    parser.add_argument("--signal_key", type=str, default="")
    parser.add_argument("--channel_index", type=int, default=0)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--scheduler", choices=["step", "exponential", "fixed"], default="step")
    parser.add_argument("--scheduler_steps", type=str, default="50,80")
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--bottleneck_size", type=int, default=256)
    parser.add_argument("--discriminator_hidden_size", type=int, default=1024)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--adaptation_schedule", choices=["constant", "progressive"], default="constant")
    parser.add_argument("--edge_dropout", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def build_config(args, source, target, output_dir, **overrides):
    values = {
        "data_dir": args.data_dir,
        "source_domain": source,
        "target_domain": target,
        "output_dir": str(output_dir),
        "window_size": args.window_size,
        "stride": args.stride,
        "normalization": args.normalization,
        "signal_key": args.signal_key,
        "channel_index": args.channel_index,
        "validation_ratio": args.validation_ratio,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "scheduler_steps": args.scheduler_steps,
        "scheduler_gamma": args.scheduler_gamma,
        "bottleneck_size": args.bottleneck_size,
        "discriminator_hidden_size": args.discriminator_hidden_size,
        "lambda_adv": args.lambda_adv,
        "lambda_mmd": args.lambda_mmd,
        "adaptation_schedule": args.adaptation_schedule,
        "edge_dropout": args.edge_dropout,
        "device": args.device,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def configure_logging(path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(message)s", "%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(Path(path), encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
