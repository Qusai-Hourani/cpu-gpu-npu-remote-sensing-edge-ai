from __future__ import annotations

r"""

Purpose:
    GPU training script for the strengthened CPU/GPU/NPU remote-sensing study.
    This version replaces the old fixed-epoch 3/5-epoch setup with max-epoch
    training, early stopping, and best-validation checkpoint selection.


Environment:
    ai_env


Typical commands:
    python main_scripts\benchmark_train_gpu_v4_convergence.py --dataset eurosat --models resnet18 --max-epochs 2 --smoke-test

    python main_scripts\benchmark_train_gpu_v4_convergence.py --dataset eurosat --models all --max-epochs 30 --patience 5
    python main_scripts\benchmark_train_gpu_v4_convergence.py --dataset aid --models all --max-epochs 50 --patience 5
    python main_scripts\benchmark_train_gpu_v4_convergence_resisc45.py --dataset resisc45 --models all --max-epochs 50 --patience 5

Outputs:
    models_converged\<model>_<dataset>_converged.pth
    results\gpu_benchmark_results_converged.csv
    results\training_logs_converged\<dataset>_<model>_history.csv
"""

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18,
    mobilenet_v2,
    mobilenet_v3_small,
    mobilenet_v3_large,
    efficientnet_b0,
    efficientnet_v2_s,
    shufflenet_v2_x1_0,
    convnext_tiny,
)

try:
    from torchvision.models import (
        ResNet18_Weights,
        MobileNet_V2_Weights,
        MobileNet_V3_Small_Weights,
        MobileNet_V3_Large_Weights,
        EfficientNet_B0_Weights,
        EfficientNet_V2_S_Weights,
        ShuffleNet_V2_X1_0_Weights,
        ConvNeXt_Tiny_Weights,
    )
    HAS_TORCHVISION_WEIGHTS_API = True
except Exception:
    HAS_TORCHVISION_WEIGHTS_API = False

try:
    import timm
    HAS_TIMM = True
except Exception:
    timm = None
    HAS_TIMM = False



# Fixed methodology settings

SEED = 42
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_LATENCY_REPEATS = 3
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 1e-4

ALL_MODELS = [
    "resnet18",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_v2_s",
    "shufflenet_v2_x1_0",
    "convnext_tiny",
    "repvit_m0_9",
    "starnet_s4",
    "shvit_s4",
    "mambaout_kobe",
]

DATASET_DEFAULT_MAX_EPOCHS = {
    "eurosat": 30,
    "aid": 50,
    "resisc45": 50,
}

DATASET_DISPLAY_NAMES = {
    "eurosat": "EuroSAT",
    "aid": "AID",
    "resisc45": "RESISC45",
}

CSV_COLUMNS = [
    "run_version",
    "dataset",
    "model",
    "model_family",
    "accelerator",
    "runtime",
    "num_classes",
    "max_epochs",
    "epochs_completed",
    "best_epoch",
    "early_stopped",
    "patience",
    "min_delta",
    "batch_size",
    "learning_rate",
    "params_million",
    "pth_size_mb",
    "best_val_accuracy_percent",
    "best_val_f1_macro",
    "test_accuracy_percent",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "fps",
    "latency_repeats",
    "warmup_batches",
    "repeat_mean_latencies_ms",
    "gpu_peak_vram_mb",
    "checkpoint_path",
    "history_csv",
    "notes",
]


@dataclass
class DatasetBundle:
    name: str
    display_name: str
    classes: List[str]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    train_size: int
    val_size: int
    test_size: int


@dataclass
class EvalMetrics:
    accuracy_percent: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


@dataclass
class LatencyMetrics:
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    fps: float
    repeat_mean_latencies_ms: List[float]
    gpu_peak_vram_mb: float



# Reproducibility and device

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    print("[WARN] CUDA is not available. CPU fallback results must NOT be used as GPU benchmark results.")
    return torch.device("cpu")



# Dataset utilities

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.subset)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_base_dataset(dataset_name: str, data_root: Path):
    dataset_name = dataset_name.lower()
    if dataset_name == "eurosat":
        print("[INFO] Loading EuroSAT from torchvision.datasets.EuroSAT ...")
        return datasets.EuroSAT(root=str(data_root), transform=None, download=False)

    if dataset_name == "aid":
        aid_root = data_root / "AID"
        print(f"[INFO] Loading AID from ImageFolder: {aid_root}")
        if not aid_root.exists():
            raise FileNotFoundError(
                f"AID folder not found: {aid_root}. Expected data/AID/<class_name>/<images>."
            )
        return datasets.ImageFolder(root=str(aid_root), transform=None)

    if dataset_name == "resisc45":
        candidates = [
            data_root / "RESISC45",
            data_root / "NWPU-RESISC45",
            data_root / "NWPU_RESISC45",
            data_root / "resisc45",
            data_root / "nwpu-resisc45",
        ]

        for resisc_root in candidates:
            if resisc_root.exists() and resisc_root.is_dir():
                class_dirs = [p for p in resisc_root.iterdir() if p.is_dir()]
                if len(class_dirs) >= 40:
                    print(f"[INFO] Loading RESISC45 from ImageFolder: {resisc_root}")
                    return datasets.ImageFolder(root=str(resisc_root), transform=None)

        searched = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "RESISC45 folder not found or does not contain enough class folders.\n"
            "Expected ImageFolder layout such as data/RESISC45/<class_name>/<images>.\n"
            f"Searched:\n{searched}"
        )

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def create_dataset_bundle(
    dataset_name: str,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    smoke_test: bool = False,
) -> DatasetBundle:
    dataset_name = dataset_name.lower()
    base_dataset = load_base_dataset(dataset_name, data_root)
    classes = list(base_dataset.classes)
    num_classes = len(classes)

    total_size = len(base_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    if smoke_test:
        print("[WARN] Smoke-test mode enabled: using small subsets only.")
        train_subset = Subset(train_subset, list(range(min(256, len(train_subset)))))
        val_subset = Subset(val_subset, list(range(min(128, len(val_subset)))))
        test_subset = Subset(test_subset, list(range(min(128, len(test_subset)))))

    train_transform, eval_transform = build_transforms()
    train_dataset = TransformedSubset(train_subset, transform=train_transform)
    val_dataset = TransformedSubset(val_subset, transform=eval_transform)
    test_dataset = TransformedSubset(test_subset, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\n" + "=" * 80)
    print(f"DATASET: {DATASET_DISPLAY_NAMES[dataset_name]}")
    print("=" * 80)
    print(f"[INFO] Total images: {total_size}")
    print(f"[INFO] Classes: {num_classes}")
    print(f"[INFO] Train/Val/Test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    return DatasetBundle(
        name=dataset_name,
        display_name=DATASET_DISPLAY_NAMES[dataset_name],
        classes=classes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )



# Model factory

def model_family(model_name: str) -> str:
    mapping = {
        "resnet18": "Residual CNN",
        "mobilenet_v2": "Mobile CNN",
        "mobilenet_v3_small": "Mobile CNN",
        "mobilenet_v3_large": "Mobile CNN",
        "efficientnet_b0": "Efficient CNN",
        "efficientnet_v2_s": "Efficient CNN",
        "shufflenet_v2_x1_0": "Lightweight CNN",
        "convnext_tiny": "Modern CNN",
        "repvit_m0_9": "Recent Lightweight Vision Model",
        "starnet_s4": "Recent Lightweight CNN",
        "shvit_s4": "Recent Efficient Vision Model",
        "mambaout_kobe": "Recent Mamba-inspired Vision Model",
    }
    return mapping.get(model_name, "Unknown")


def weights_or_none(weights_cls):
    if HAS_TORCHVISION_WEIGHTS_API:
        return weights_cls.DEFAULT
    return None


def create_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()

    if name == "resnet18":
        model = resnet18(weights=weights_or_none(ResNet18_Weights)) if HAS_TORCHVISION_WEIGHTS_API else resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v2":
        model = mobilenet_v2(weights=weights_or_none(MobileNet_V2_Weights)) if HAS_TORCHVISION_WEIGHTS_API else mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=weights_or_none(MobileNet_V3_Small_Weights)) if HAS_TORCHVISION_WEIGHTS_API else mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if name == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=weights_or_none(MobileNet_V3_Large_Weights)) if HAS_TORCHVISION_WEIGHTS_API else mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = efficientnet_b0(weights=weights_or_none(EfficientNet_B0_Weights)) if HAS_TORCHVISION_WEIGHTS_API else efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "efficientnet_v2_s":
        model = efficientnet_v2_s(weights=weights_or_none(EfficientNet_V2_S_Weights)) if HAS_TORCHVISION_WEIGHTS_API else efficientnet_v2_s(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if name == "shufflenet_v2_x1_0":
        model = shufflenet_v2_x1_0(weights=weights_or_none(ShuffleNet_V2_X1_0_Weights)) if HAS_TORCHVISION_WEIGHTS_API else shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "convnext_tiny":
        model = convnext_tiny(weights=weights_or_none(ConvNeXt_Tiny_Weights)) if HAS_TORCHVISION_WEIGHTS_API else convnext_tiny(pretrained=True)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if name == "repvit_m0_9":
        if not HAS_TIMM:
            raise ImportError("RepViT requires timm. Install with: pip install timm")
        return timm.create_model(
            "repvit_m0_9.dist_450e_in1k",
            pretrained=True,
            num_classes=num_classes,
        )

    if name == "starnet_s4":
        if not HAS_TIMM:
            raise ImportError("StarNet requires timm. Install with: pip install timm")
        return timm.create_model(
            "starnet_s4",
            pretrained=True,
            num_classes=num_classes,
        )

    if name == "shvit_s4":
        if not HAS_TIMM:
            raise ImportError("SHViT requires timm. Install with: pip install timm")
        return timm.create_model(
            "shvit_s4",
            pretrained=True,
            num_classes=num_classes,
        )

    if name == "mambaout_kobe":
        if not HAS_TIMM:
            raise ImportError("MambaOut requires timm. Install with: pip install timm")
        return timm.create_model(
            "mambaout_kobe",
            pretrained=True,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)



# Metrics

def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_classes: int,
) -> EvalMetrics:
    y_true_arr = np.asarray(list(y_true), dtype=np.int64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.int64)

    accuracy = 100.0 * float((y_true_arr == y_pred_arr).sum()) / max(1, len(y_true_arr))

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = np.logical_and(y_pred_arr == c, y_true_arr == c).sum()
        fp = np.logical_and(y_pred_arr == c, y_true_arr != c).sum()
        fn = np.logical_and(y_pred_arr != c, y_true_arr == c).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))

    return EvalMetrics(
        accuracy_percent=accuracy,
        precision_macro=100.0 * float(np.mean(precisions)),
        recall_macro=100.0 * float(np.mean(recalls)),
        f1_macro=100.0 * float(np.mean(f1s)),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> EvalMetrics:
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    return compute_classification_metrics(all_targets, all_preds, num_classes=num_classes)



# Training and latency

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    max_epochs: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.perf_counter()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            loss_so_far = running_loss / max(1, total)
            acc_so_far = 100.0 * correct / max(1, total)
            print(
                f"  [Epoch {epoch_idx}/{max_epochs}] "
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss {loss_so_far:.4f} | Train Acc {acc_so_far:.2f}%"
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)
    print(f"[INFO] Epoch time: {elapsed:.2f}s")
    return epoch_loss, epoch_acc


@torch.no_grad()
def run_warmup_batches(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int,
) -> None:
    model.eval()
    completed = 0
    for inputs, _ in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        completed += 1
        if completed >= warmup_batches:
            break
    print(f"[INFO] Completed warmup batches: {completed}")


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    repeats: int = DEFAULT_LATENCY_REPEATS,
) -> LatencyMetrics:
    model.eval()
    all_batch_ms_per_image: List[float] = []
    repeat_mean_latencies: List[float] = []
    total_timed_seconds = 0.0
    total_timed_images = 0

    gpu_peak_vram_mb = 0.0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(
        f"[INFO] Measuring latency: warmup_batches={warmup_batches}, "
        f"repeats={repeats}, one sample per timed batch."
    )

    run_warmup_batches(model, dataloader, device, warmup_batches=warmup_batches)

    for repeat_idx in range(1, repeats + 1):
        repeat_time = 0.0
        repeat_images = 0

        print(f"[INFO] Latency repeat {repeat_idx}/{repeats} ...")

        for inputs, _ in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            batch_size = inputs.size(0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(inputs)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            batch_time = end - start
            batch_ms_per_image = (batch_time / batch_size) * 1000.0

            all_batch_ms_per_image.append(batch_ms_per_image)
            repeat_time += batch_time
            repeat_images += batch_size

        repeat_mean = (repeat_time / repeat_images) * 1000.0 if repeat_images > 0 else 0.0
        repeat_mean_latencies.append(float(repeat_mean))
        total_timed_seconds += repeat_time
        total_timed_images += repeat_images

        print(
            f"[INFO] Repeat {repeat_idx}/{repeats}: "
            f"mean={repeat_mean:.6f} ms/image over {repeat_images} images"
        )

    if device.type == "cuda":
        gpu_peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    lat = np.asarray(all_batch_ms_per_image, dtype=np.float64)
    mean_latency_ms = float(np.mean(lat)) if len(lat) else 0.0
    median_latency_ms = float(np.median(lat)) if len(lat) else 0.0
    p95_latency_ms = float(np.percentile(lat, 95)) if len(lat) else 0.0
    fps = float(total_timed_images / total_timed_seconds) if total_timed_seconds > 0 else 0.0

    return LatencyMetrics(
        mean_latency_ms=mean_latency_ms,
        median_latency_ms=median_latency_ms,
        p95_latency_ms=p95_latency_ms,
        fps=fps,
        repeat_mean_latencies_ms=repeat_mean_latencies,
        gpu_peak_vram_mb=gpu_peak_vram_mb,
    )


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def append_result_csv(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_history_csv(history_path: Path, rows: List[Dict[str, object]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy_percent",
        "val_accuracy_percent",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "is_best",
    ]
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



# Main benchmark loop

def run_one_model(
    model_name: str,
    dataset: DatasetBundle,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    batch_size: int,
    results_path: Path,
    models_dir: Path,
    history_dir: Path,
    smoke_test: bool,
    warmup_batches: int,
    latency_repeats: int,
) -> None:
    print("\n" + "=" * 100)
    print(f"MODEL: {model_name} | DATASET: {dataset.display_name}")
    print("=" * 100)

    model = create_model(model_name, num_classes=dataset.num_classes)
    model = model.to(device)

    params_million = count_parameters(model) / 1_000_000.0
    print(f"[INFO] Parameter count: {params_million:.3f}M")
    print(f"[INFO] Model family: {model_family(model_name)}")
    print(f"[INFO] Max epochs: {max_epochs} | patience: {patience} | min_delta: {min_delta}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    early_stopped = False
    history_rows: List[Dict[str, object]] = []

    for epoch in range(1, max_epochs + 1):
        print("\n" + "-" * 80)
        print(f"[INFO] Training {model_name} on {dataset.display_name}: epoch {epoch}/{max_epochs}")
        print("-" * 80)

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=dataset.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            max_epochs=max_epochs,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=dataset.val_loader,
            device=device,
            num_classes=dataset.num_classes,
        )

        improved = val_metrics.accuracy_percent > (best_val_acc + min_delta)

        if improved:
            best_val_acc = val_metrics.accuracy_percent
            best_val_f1 = val_metrics.f1_macro
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"[INFO] New best validation accuracy: {best_val_acc:.4f}% at epoch {best_epoch}")
        else:
            epochs_without_improvement += 1
            print(
                f"[INFO] No validation improvement. "
                f"epochs_without_improvement={epochs_without_improvement}/{patience}"
            )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy_percent": f"{train_acc:.6f}",
                "val_accuracy_percent": f"{val_metrics.accuracy_percent:.6f}",
                "val_precision_macro": f"{val_metrics.precision_macro:.6f}",
                "val_recall_macro": f"{val_metrics.recall_macro:.6f}",
                "val_f1_macro": f"{val_metrics.f1_macro:.6f}",
                "is_best": "yes" if improved else "no",
            }
        )

        print(
            f"[EPOCH RESULT] {model_name} | {dataset.display_name} | "
            f"Epoch {epoch}/{max_epochs} | Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.2f}% | Val Acc {val_metrics.accuracy_percent:.2f}% | "
            f"Val Macro F1 {val_metrics.f1_macro:.2f}%"
        )

        if epoch >= 2 and epochs_without_improvement >= patience:
            early_stopped = True
            print(f"[INFO] Early stopping triggered at epoch {epoch}.")
            break

    epochs_completed = len(history_rows)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"[INFO] Loaded best validation checkpoint: epoch={best_epoch}, val_acc={best_val_acc:.4f}%")
    else:
        print("[WARN] No best_state was stored. Using final epoch state.")

    checkpoint_name = f"{model_name}_{dataset.name}_converged.pth"
    if smoke_test:
        checkpoint_name = f"SMOKE_{checkpoint_name}"
    checkpoint_path = models_dir / checkpoint_name
    save_checkpoint(model, checkpoint_path)
    pth_size = file_size_mb(checkpoint_path)
    print(f"[INFO] Saved best checkpoint: {checkpoint_path}")
    print(f"[INFO] PTH size: {pth_size:.2f} MB")

    history_path = history_dir / f"{dataset.name}_{model_name}_history.csv"
    if smoke_test:
        history_path = history_dir / f"SMOKE_{dataset.name}_{model_name}_history.csv"
    write_history_csv(history_path, history_rows)
    print(f"[INFO] Wrote training history: {history_path}")

    print("\n[INFO] Evaluating final test metrics using best validation checkpoint ...")
    test_metrics = evaluate_model(
        model=model,
        dataloader=dataset.test_loader,
        device=device,
        num_classes=dataset.num_classes,
    )

    print("[INFO] Measuring final GPU latency/FPS/VRAM ...")
    latency_metrics = measure_latency(
        model=model,
        dataloader=dataset.test_loader,
        device=device,
        warmup_batches=warmup_batches,
        repeats=latency_repeats,
    )

    print("\n" + "=" * 100)
    print(f"FINAL CONVERGENCE RESULT | {dataset.display_name} | {model_name}")
    print("=" * 100)
    print(f"Best epoch:      {best_epoch}/{max_epochs}")
    print(f"Early stopped:   {early_stopped}")
    print(f"Best Val Acc:    {best_val_acc:.2f}%")
    print(f"Best Val F1:     {best_val_f1:.2f}%")
    print(f"Test Accuracy:   {test_metrics.accuracy_percent:.2f}%")
    print(f"Macro Prec.:     {test_metrics.precision_macro:.2f}%")
    print(f"Macro Recall:    {test_metrics.recall_macro:.2f}%")
    print(f"Macro F1:        {test_metrics.f1_macro:.2f}%")
    print(f"Mean latency:    {latency_metrics.mean_latency_ms:.4f} ms/image")
    print(f"Median latency:  {latency_metrics.median_latency_ms:.4f} ms/image")
    print(f"P95 latency:     {latency_metrics.p95_latency_ms:.4f} ms/image")
    print(f"FPS:             {latency_metrics.fps:.2f} images/s")
    print(f"Peak GPU VRAM:   {latency_metrics.gpu_peak_vram_mb:.2f} MB")
    print(f"Repeat means:    {latency_metrics.repeat_mean_latencies_ms}")

    notes = "V4 convergence PyTorch CUDA GPU benchmark; early stopping; best validation checkpoint"
    if smoke_test:
        notes = "SMOKE TEST ONLY - not for paper results"
    elif device.type != "cuda":
        notes = "CPU fallback - not valid as GPU benchmark"

    row = {
        "run_version": "v4_convergence",
        "dataset": dataset.display_name,
        "model": model_name,
        "model_family": model_family(model_name),
        "accelerator": "GPU" if device.type == "cuda" else "CPU_FALLBACK",
        "runtime": "PyTorch CUDA" if device.type == "cuda" else "PyTorch CPU",
        "num_classes": dataset.num_classes,
        "max_epochs": max_epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "early_stopped": early_stopped,
        "patience": patience,
        "min_delta": min_delta,
        "batch_size": batch_size,
        "learning_rate": LEARNING_RATE,
        "params_million": f"{params_million:.6f}",
        "pth_size_mb": f"{pth_size:.6f}",
        "best_val_accuracy_percent": f"{best_val_acc:.6f}",
        "best_val_f1_macro": f"{best_val_f1:.6f}",
        "test_accuracy_percent": f"{test_metrics.accuracy_percent:.6f}",
        "precision_macro": f"{test_metrics.precision_macro:.6f}",
        "recall_macro": f"{test_metrics.recall_macro:.6f}",
        "f1_macro": f"{test_metrics.f1_macro:.6f}",
        "mean_latency_ms": f"{latency_metrics.mean_latency_ms:.6f}",
        "median_latency_ms": f"{latency_metrics.median_latency_ms:.6f}",
        "p95_latency_ms": f"{latency_metrics.p95_latency_ms:.6f}",
        "fps": f"{latency_metrics.fps:.6f}",
        "latency_repeats": latency_repeats,
        "warmup_batches": warmup_batches,
        "repeat_mean_latencies_ms": ";".join(f"{x:.6f}" for x in latency_metrics.repeat_mean_latencies_ms),
        "gpu_peak_vram_mb": f"{latency_metrics.gpu_peak_vram_mb:.6f}",
        "checkpoint_path": str(checkpoint_path),
        "history_csv": str(history_path),
        "notes": notes,
    }
    append_result_csv(results_path, row)
    print(f"[INFO] Appended result to: {results_path}")



# CLI

def parse_models(raw_models: List[str]) -> List[str]:
    if len(raw_models) == 1 and raw_models[0].lower() == "all":
        return ALL_MODELS.copy()

    selected = []
    for m in raw_models:
        m_lower = m.lower()
        if m_lower not in ALL_MODELS:
            raise ValueError(f"Unknown model '{m}'. Allowed: {ALL_MODELS} or all")
        selected.append(m_lower)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 convergence GPU training benchmark for EuroSAT/AID/RESISC45.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["eurosat", "aid", "resisc45"],
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to run, or 'all'.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum epochs. Defaults: EuroSAT=30, AID=50, RESISC45=50.",
    )
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--models-dir", type=str, default="models_converged")
    parser.add_argument("--results-path", type=str, default="results/gpu_benchmark_results_converged.csv")
    parser.add_argument("--history-dir", type=str, default="results/training_logs_converged")
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use tiny subsets for checking script flow. Results are not valid for the paper.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Delete the output results CSV before this run starts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    configure_torch()

    selected_models = parse_models(args.models)
    max_epochs = args.max_epochs if args.max_epochs is not None else DATASET_DEFAULT_MAX_EPOCHS[args.dataset]
    results_path = Path(args.results_path)

    if args.overwrite_results and results_path.exists():
        results_path.unlink()
        print(f"[INFO] Deleted old results CSV: {results_path}")

    print("\n" + "=" * 100)
    print("GPU TRAINING BENCHMARK V4: CONVERGENCE + EARLY STOPPING")
    print("=" * 100)
    print(f"[INFO] Working directory: {Path.cwd()}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Models: {selected_models}")
    print(f"[INFO] Max epochs: {max_epochs}")
    print(f"[INFO] Patience: {args.patience}")
    print(f"[INFO] Min delta: {args.min_delta}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")
    print(f"[INFO] Seed: {SEED}")
    print(f"[INFO] Warmup batches: {args.warmup_batches}")
    print(f"[INFO] Latency repeats: {args.latency_repeats}")
    print(f"[INFO] Output models dir: {Path(args.models_dir)}")
    print(f"[INFO] Output results CSV: {results_path}")
    print(f"[INFO] Output history dir: {Path(args.history_dir)}")
    print("[IMPORTANT] Real benchmark runs must be in Turbo/performance mode because latency/FPS/VRAM are measured.")

    if args.smoke_test:
        print("[WARN] Smoke-test mode enabled. Results are not valid for the paper.")

    device = get_device()
    dataset_bundle = create_dataset_bundle(
        dataset_name=args.dataset,
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
    )

    for model_name in selected_models:
        run_one_model(
            model_name=model_name,
            dataset=dataset_bundle,
            device=device,
            max_epochs=max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            batch_size=args.batch_size,
            results_path=results_path,
            models_dir=Path(args.models_dir),
            history_dir=Path(args.history_dir),
            smoke_test=args.smoke_test,
            warmup_batches=args.warmup_batches,
            latency_repeats=args.latency_repeats,
        )

    print("\n" + "=" * 100)
    print("[OK] V4 convergence GPU benchmark script finished.")
    print(f"[INFO] Results CSV: {results_path}")
    print(f"[INFO] Checkpoints folder: {Path(args.models_dir)}")
    print(f"[INFO] History folder: {Path(args.history_dir)}")
    print("=" * 100)


if __name__ == "__main__":
    main()


