from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_loader import SuspensionBridgeData
from losses import multi_kernel_mmd
from models import DAGNNFeatureExtractor, DomainDiscriminator
from plotting import (
    plot_adaptation_curves,
    plot_confusion,
    plot_training_curves,
    plot_tsne,
    save_history,
)


@dataclass
class TrainingConfig:
    data_dir: str
    source_domain: str
    target_domain: str
    output_dir: str
    window_size: int = 1024
    stride: int = 1024
    normalization: str = "mean-std"
    signal_key: str = ""
    channel_index: int = 0
    validation_ratio: float = 0.2
    split_seed: int = 42
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 50
    warmup_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9
    optimizer: str = "adam"
    scheduler: str = "step"
    scheduler_steps: str = "50,80"
    scheduler_gamma: float = 0.1
    bottleneck_size: int = 256
    discriminator_hidden_size: int = 1024
    lambda_adv: float = 1.0
    lambda_mmd: float = 1.0
    adaptation_schedule: str = "constant"
    edge_dropout: float = 0.5
    use_gcn: bool = True
    use_adv: bool = True
    use_mmd: bool = True
    variant_name: str = "Full DAGNN"
    device: str = "cuda:0"


def set_random_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "source_validation_loss": [],
            "source_validation_accuracy": [],
            "target_validation_loss": [],
            "target_validation_accuracy": [],
            "domain_loss": [],
            "domain_accuracy": [],
            "mmd_loss": [],
        }

    def setup(self):
        set_random_seed(self.config.seed)
        data = SuspensionBridgeData(
            data_dir=self.config.data_dir,
            source_domain=self.config.source_domain,
            target_domain=self.config.target_domain,
            window_size=self.config.window_size,
            stride=self.config.stride,
            normalization=self.config.normalization,
            signal_key=self.config.signal_key,
            channel_index=self.config.channel_index,
            validation_ratio=self.config.validation_ratio,
            split_seed=self.config.split_seed,
        )
        self.class_names = data.class_names
        self.datasets = data.build()
        generator = torch.Generator().manual_seed(self.config.seed)
        pin_memory = self.device.type == "cuda"
        self.loaders = {
            name: DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=name.endswith("train"),
                num_workers=self.config.num_workers,
                pin_memory=pin_memory,
                drop_last=False,
                generator=generator,
            )
            for name, dataset in self.datasets.items()
        }
        self.feature_extractor = DAGNNFeatureExtractor(
            use_gcn=self.config.use_gcn,
            edge_dropout=self.config.edge_dropout,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(256, self.config.bottleneck_size),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.classifier = nn.Linear(self.config.bottleneck_size, len(self.class_names))
        maximum_iterations = max(
            1,
            len(self.loaders["source_train"])
            * max(self.config.epochs - self.config.warmup_epochs, 1),
        )
        self.discriminator = (
            DomainDiscriminator(
                input_features=self.config.bottleneck_size,
                hidden_size=self.config.discriminator_hidden_size,
                max_iterations=maximum_iterations,
            )
            if self.config.use_adv
            else None
        )
        self.prediction_model = nn.Sequential(
            self.feature_extractor,
            self.bottleneck,
            self.classifier,
        ).to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)
        parameters = list(self.prediction_model.parameters())
        if self.discriminator is not None:
            parameters.extend(self.discriminator.parameters())
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        if self.config.scheduler == "step":
            milestones = [int(value) for value in self.config.scheduler_steps.split(",") if value]
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=milestones,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.config.scheduler_gamma,
            )
        elif self.config.scheduler == "fixed":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
        self.classification_loss = nn.CrossEntropyLoss()
        self.domain_loss = nn.BCELoss()
        self.best_accuracy = -1.0
        self.best_epoch = -1
        self.checkpoint_path = self.output_dir / "best_model.pth"
        self.initial_source_features, _ = self.extract_features(self.loaders["source_train"])
        self.initial_target_features, _ = self.extract_features(self.loaders["target_validation"])
        (self.output_dir / "configuration.json").write_text(
            json.dumps(asdict(self.config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _set_training(self, training):
        self.prediction_model.train(training)
        if self.discriminator is not None:
            self.discriminator.train(training)

    def _features(self, inputs):
        return self.bottleneck(self.feature_extractor(inputs))

    def _adaptation_factor(self, epoch):
        if self.config.adaptation_schedule == "constant":
            return 1.0
        if self.config.adaptation_schedule == "progressive":
            denominator = max(self.config.epochs - self.config.warmup_epochs, 1)
            progress = max(epoch - self.config.warmup_epochs, 0) / denominator
            return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        raise ValueError(f"Unsupported adaptation schedule: {self.config.adaptation_schedule}")

    def extract_features(self, loader):
        self._set_training(False)
        features = []
        labels = []
        with torch.no_grad():
            for inputs, batch_labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                features.append(self._features(inputs).cpu())
                labels.append(batch_labels.cpu())
        return torch.cat(features).numpy(), torch.cat(labels).numpy()

    def _evaluate(self, loader):
        self._set_training(False)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        true_labels = []
        predictions = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.classifier(self._features(inputs))
                loss = self.classification_loss(logits, labels)
                predicted = logits.argmax(dim=1)
                total_loss += loss.item() * labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += labels.size(0)
                true_labels.append(labels.cpu())
                predictions.append(predicted.cpu())
        return (
            total_loss / total_samples,
            100.0 * total_correct / total_samples,
            torch.cat(true_labels).numpy(),
            torch.cat(predictions).numpy(),
        )

    def _train_epoch(self, epoch):
        self._set_training(True)
        target_iterator = iter(self.loaders["target_train"])
        adaptation_active = epoch >= self.config.warmup_epochs and (
            self.config.use_adv or self.config.use_mmd
        )
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        domain_loss_sum = 0.0
        domain_correct = 0
        domain_samples = 0
        mmd_sum = 0.0
        mmd_batches = 0
        factor = self._adaptation_factor(epoch)
        for source_inputs, source_labels in self.loaders["source_train"]:
            if adaptation_active:
                try:
                    target_inputs, _ = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(self.loaders["target_train"])
                    target_inputs, _ = next(target_iterator)
                inputs = torch.cat((source_inputs, target_inputs), dim=0)
            else:
                inputs = source_inputs
            inputs = inputs.to(self.device, non_blocking=True)
            source_labels = source_labels.to(self.device, non_blocking=True)
            source_count = source_labels.size(0)
            features = self._features(inputs)
            source_logits = self.classifier(features[:source_count])
            classification = self.classification_loss(source_logits, source_labels)
            loss = classification
            if adaptation_active and self.config.use_adv:
                target_count = features.size(0) - source_count
                domain_labels = torch.cat(
                    (
                        torch.ones(source_count, device=self.device),
                        torch.zeros(target_count, device=self.device),
                    )
                )
                domain_output = self.discriminator(features)
                adversarial = self.domain_loss(domain_output, domain_labels)
                loss = loss + factor * self.config.lambda_adv * adversarial
                domain_loss_sum += adversarial.item() * domain_labels.numel()
                domain_correct += (domain_output >= 0.5).eq(domain_labels.bool()).sum().item()
                domain_samples += domain_labels.numel()
            if adaptation_active and self.config.use_mmd:
                discrepancy = multi_kernel_mmd(features[:source_count], features[source_count:])
                loss = loss + factor * self.config.lambda_mmd * discrepancy
                mmd_sum += discrepancy.item()
                mmd_batches += 1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            predicted = source_logits.argmax(dim=1)
            total_loss += loss.item() * source_count
            total_correct += predicted.eq(source_labels).sum().item()
            total_samples += source_count
        return {
            "loss": total_loss / total_samples,
            "accuracy": 100.0 * total_correct / total_samples,
            "domain_loss": domain_loss_sum / domain_samples if domain_samples else 0.0,
            "domain_accuracy": 100.0 * domain_correct / domain_samples if domain_samples else 0.0,
            "mmd_loss": mmd_sum / mmd_batches if mmd_batches else 0.0,
        }

    def _save_best(self, epoch, accuracy):
        self.best_epoch = epoch
        self.best_accuracy = accuracy
        torch.save(
            {
                "epoch": epoch,
                "target_validation_accuracy": accuracy,
                "class_names": self.class_names,
                "configuration": asdict(self.config),
                "model_state_dict": self.prediction_model.state_dict(),
            },
            self.checkpoint_path,
        )

    def train(self):
        logging.info("Source domain: %s", self.config.source_domain)
        logging.info("Target domain: %s", self.config.target_domain)
        logging.info("Classes: %s", self.class_names)
        for epoch in range(self.config.epochs):
            training = self._train_epoch(epoch)
            source_loss, source_accuracy, _, _ = self._evaluate(self.loaders["source_validation"])
            target_loss, target_accuracy, _, _ = self._evaluate(self.loaders["target_validation"])
            self.history["train_loss"].append(training["loss"])
            self.history["train_accuracy"].append(training["accuracy"])
            self.history["source_validation_loss"].append(source_loss)
            self.history["source_validation_accuracy"].append(source_accuracy)
            self.history["target_validation_loss"].append(target_loss)
            self.history["target_validation_accuracy"].append(target_accuracy)
            if self.config.use_adv:
                self.history["domain_loss"].append(training["domain_loss"])
                self.history["domain_accuracy"].append(training["domain_accuracy"])
            if self.config.use_mmd:
                self.history["mmd_loss"].append(training["mmd_loss"])
            if target_accuracy > self.best_accuracy:
                self._save_best(epoch, target_accuracy)
            logging.info(
                "Epoch %d/%d train_loss=%.4f train_acc=%.2f source_val_acc=%.2f target_val_acc=%.2f",
                epoch + 1,
                self.config.epochs,
                training["loss"],
                training["accuracy"],
                source_accuracy,
                target_accuracy,
            )
            if self.scheduler is not None:
                self.scheduler.step()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.prediction_model.load_state_dict(checkpoint["model_state_dict"])
        _, _, true_labels, predictions = self._evaluate(self.loaders["target_validation"])
        final_source_features, _ = self.extract_features(self.loaders["source_train"])
        final_target_features, _ = self.extract_features(self.loaders["target_validation"])
        save_history(self.history, self.output_dir)
        plot_training_curves(self.history, self.output_dir)
        plot_adaptation_curves(self.history, self.output_dir)
        plot_confusion(true_labels, predictions, self.class_names, self.output_dir)
        plot_tsne(
            self.initial_source_features,
            self.initial_target_features,
            "before_training",
            self.output_dir,
            self.config.seed,
        )
        plot_tsne(
            final_source_features,
            final_target_features,
            "best_checkpoint",
            self.output_dir,
            self.config.seed,
        )
        summary = {
            "variant": self.config.variant_name,
            "source_domain": self.config.source_domain,
            "target_domain": self.config.target_domain,
            "class_count": len(self.class_names),
            "class_names": self.class_names,
            "best_epoch": self.best_epoch,
            "best_target_validation_accuracy_percent": self.best_accuracy,
            "source_train_samples": len(self.datasets["source_train"]),
            "source_validation_samples": len(self.datasets["source_validation"]),
            "target_train_samples": len(self.datasets["target_train"]),
            "target_validation_samples": len(self.datasets["target_validation"]),
            "checkpoint": str(self.checkpoint_path.resolve()),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
