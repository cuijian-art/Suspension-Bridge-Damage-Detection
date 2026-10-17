from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


DOMAIN_NAMES = (
    "HammerExcitation",
    "ShakerExcitation",
    "MovingTrolleyExcitation",
)
SUPPORTED_SUFFIXES = {".mat", ".npy", ".npz", ".csv", ".txt"}


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


class SignalDataset(Dataset):
    def __init__(self, signals, labels, normalization="mean-std"):
        self.signals = np.asarray(signals, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.normalization = normalization

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        signal = self.signals[index].astype(np.float32, copy=True)
        if self.normalization == "mean-std":
            signal = (signal - signal.mean()) / max(float(signal.std()), 1e-8)
        elif self.normalization == "0-1":
            signal = (signal - signal.min()) / max(float(signal.max() - signal.min()), 1e-8)
        elif self.normalization == "-1-1":
            signal = 2.0 * (signal - signal.min()) / max(float(signal.max() - signal.min()), 1e-8) - 1.0
        elif self.normalization != "none":
            raise ValueError(f"Unsupported normalization: {self.normalization}")
        return torch.from_numpy(signal[None, :]), torch.tensor(self.labels[index], dtype=torch.long)


class SuspensionBridgeData:
    def __init__(
        self,
        data_dir,
        source_domain,
        target_domain,
        window_size=1024,
        stride=1024,
        normalization="mean-std",
        signal_key="",
        channel_index=0,
        validation_ratio=0.2,
        split_seed=42,
    ):
        self.data_dir = Path(data_dir)
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.window_size = int(window_size)
        self.stride = int(stride) if int(stride) > 0 else int(window_size)
        self.normalization = normalization
        self.signal_key = signal_key
        self.channel_index = int(channel_index)
        self.validation_ratio = float(validation_ratio)
        self.split_seed = int(split_seed)
        self.class_names = self._discover_classes()
        self.num_classes = len(self.class_names)

    def _domain_path(self, domain):
        if domain not in DOMAIN_NAMES:
            raise ValueError(f"Unknown excitation domain: {domain}")
        path = self.data_dir / domain
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    def _discover_classes(self):
        source_classes = {
            path.name for path in self._domain_path(self.source_domain).iterdir() if path.is_dir()
        }
        target_classes = {
            path.name for path in self._domain_path(self.target_domain).iterdir() if path.is_dir()
        }
        if source_classes != target_classes:
            raise ValueError(
                f"Class folders differ between {self.source_domain} and {self.target_domain}: "
                f"source={sorted(source_classes)}, target={sorted(target_classes)}"
            )
        if len(source_classes) < 2:
            raise ValueError("At least two class folders are required in each excitation domain")
        return sorted(source_classes, key=natural_key)

    def _numeric_array(self, path):
        suffix = path.suffix.lower()
        if suffix == ".mat":
            content = loadmat(path)
            if self.signal_key:
                if self.signal_key not in content:
                    raise KeyError(f"{self.signal_key} is not present in {path}")
                array = content[self.signal_key]
            else:
                candidates = [
                    value
                    for key, value in content.items()
                    if not key.startswith("__") and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
                ]
                if not candidates:
                    raise ValueError(f"No numeric signal was found in {path}")
                array = max(candidates, key=lambda value: value.size)
        elif suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            content = np.load(path, allow_pickle=False)
            if self.signal_key:
                if self.signal_key not in content.files:
                    raise KeyError(f"{self.signal_key} is not present in {path}")
                array = content[self.signal_key]
            else:
                if not content.files:
                    raise ValueError(f"No array was found in {path}")
                array = max((content[key] for key in content.files), key=lambda value: value.size)
        elif suffix == ".csv":
            array = np.loadtxt(path, delimiter=",")
        elif suffix == ".txt":
            array = np.loadtxt(path)
        else:
            raise ValueError(f"Unsupported file: {path}")
        array = np.asarray(array)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Signal data must be numeric: {path}")
        return np.real(array).astype(np.float32, copy=False)

    def _continuous_signal(self, array, path):
        array = np.squeeze(array)
        if array.ndim == 1:
            return array
        if array.ndim != 2:
            raise ValueError(f"Expected one- or two-dimensional signal data in {path}, got {array.shape}")
        time_axis = int(np.argmax(array.shape))
        channel_axis = 1 - time_axis
        channel_count = array.shape[channel_axis]
        if not 0 <= self.channel_index < channel_count:
            raise IndexError(
                f"channel_index={self.channel_index} is outside the available range 0..{channel_count - 1} in {path}"
            )
        return np.take(array, self.channel_index, axis=channel_axis).reshape(-1)

    def _windows_from_file(self, path):
        signal = self._continuous_signal(self._numeric_array(path), path)
        if signal.size < self.window_size:
            return []
        return [
            signal[start : start + self.window_size]
            for start in range(0, signal.size - self.window_size + 1, self.stride)
        ]

    def _load_domain(self, domain):
        signals = []
        labels = []
        domain_path = self._domain_path(domain)
        for label, class_name in enumerate(self.class_names):
            class_path = domain_path / class_name
            files = sorted(
                (
                    path
                    for path in class_path.rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
                ),
                key=lambda path: natural_key(str(path.relative_to(class_path))),
            )
            if not files:
                raise FileNotFoundError(f"No supported signal files were found in {class_path}")
            class_count = 0
            for path in files:
                windows = self._windows_from_file(path)
                signals.extend(windows)
                labels.extend([label] * len(windows))
                class_count += len(windows)
            if class_count < 2:
                raise ValueError(f"Not enough windows were extracted from {class_path}")
        return np.stack(signals), np.asarray(labels, dtype=np.int64)

    def _split(self, signals, labels):
        indices = np.arange(len(labels))
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=self.validation_ratio,
            random_state=self.split_seed,
            shuffle=True,
            stratify=labels,
        )
        return (
            SignalDataset(signals[train_indices], labels[train_indices], self.normalization),
            SignalDataset(signals[validation_indices], labels[validation_indices], self.normalization),
        )

    def build(self):
        source_signals, source_labels = self._load_domain(self.source_domain)
        target_signals, target_labels = self._load_domain(self.target_domain)
        source_train, source_validation = self._split(source_signals, source_labels)
        target_train, target_validation = self._split(target_signals, target_labels)
        return {
            "source_train": source_train,
            "source_validation": source_validation,
            "target_train": target_train,
            "target_validation": target_validation,
        }
