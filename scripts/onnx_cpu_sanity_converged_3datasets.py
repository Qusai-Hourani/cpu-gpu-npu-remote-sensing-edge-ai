from __future__ import annotations

"""

Correctness check for the 36 converged ONNX models.
Compares PyTorch CPU vs ONNX Runtime CPU on the same deterministic test loader.
This is NOT a latency benchmark and does NOT modify checkpoints or result CSVs.


"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPU_RESULTS_CSV = PROJECT_ROOT / "results" / "gpu_benchmark_results_converged_3datasets.csv"
DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_models_converged_3datasets"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "onnx_cpu_sanity_results_converged_3datasets.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

IMAGE_SIZE = 224
SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
DEFAULT_THRESHOLD_PP = 0.25
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_COLUMNS = [
    "dataset", "model", "num_classes", "checkpoint_path", "onnx_path",
    "gpu_csv_test_accuracy_percent", "pytorch_cpu_accuracy_percent", "onnx_cpu_accuracy_percent",
    "onnx_vs_pytorch_diff_percent_points", "gpu_csv_vs_pytorch_cpu_diff_percent_points",
    "pytorch_macro_precision_percent", "pytorch_macro_recall_percent", "pytorch_macro_f1_percent",
    "onnx_macro_precision_percent", "onnx_macro_recall_percent", "onnx_macro_f1_percent",
    "num_test_images", "status", "notes",
]

try:
    import timm
    HAS_TIMM = True
except Exception:
    timm = None
    HAS_TIMM = False


def print_banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def normalize_dataset_name(value: str) -> str:
    lower = str(value).strip().lower()
    if lower == "eurosat":
        return "EuroSAT"
    if lower == "aid":
        return "AID"
    if lower in ["resisc45", "nwpu-resisc45", "nwpu_resisc45"]:
        return "RESISC45"
    return str(value).strip()


def dataset_key(value: str) -> str:
    return normalize_dataset_name(value).lower()


def onnx_filename(model_name: str, dataset_name: str) -> str:
    return f"{model_name.lower()}_{dataset_key(dataset_name)}_converged.onnx"


def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower().strip()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "shufflenet_v2_x1_0":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if name == "repvit_m0_9":
        if not HAS_TIMM:
            raise ImportError("repvit_m0_9 requires timm.")
        return timm.create_model("repvit_m0_9.dist_450e_in1k", pretrained=False, num_classes=num_classes)
    if name == "starnet_s4":
        if not HAS_TIMM:
            raise ImportError("starnet_s4 requires timm.")
        return timm.create_model("starnet_s4", pretrained=False, num_classes=num_classes)
    if name == "shvit_s4":
        if not HAS_TIMM:
            raise ImportError("shvit_s4 requires timm.")
        return timm.create_model("shvit_s4", pretrained=False, num_classes=num_classes)
    if name == "mambaout_kobe":
        if not HAS_TIMM:
            raise ImportError("mambaout_kobe requires timm.")
        return timm.create_model("mambaout_kobe", pretrained=False, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def load_checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def clean_state_dict_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = str(key)
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        cleaned[new_key] = value
    return cleaned


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_resisc45_root(data_root: Path) -> Path:
    candidates = [
        data_root / "RESISC45", data_root / "NWPU-RESISC45", data_root / "NWPU_RESISC45",
        data_root / "resisc45", data_root / "nwpu-resisc45",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and len([p for p in candidate.iterdir() if p.is_dir()]) >= 40:
            return candidate
    raise FileNotFoundError("Could not find RESISC45 folder under data root.")


def make_dataset(dataset_name: str, data_root: Path):
    key = dataset_key(dataset_name)
    transform = get_eval_transform()
    if key == "eurosat":
        return datasets.EuroSAT(root=str(data_root), download=False, transform=transform)
    if key == "aid":
        root = data_root / "AID"
        if not root.exists():
            raise FileNotFoundError(f"AID folder not found: {root}")
        return datasets.ImageFolder(root=str(root), transform=transform)
    if key == "resisc45":
        return datasets.ImageFolder(root=str(find_resisc45_root(data_root)), transform=transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def deterministic_test_loader(dataset_name: str, data_root: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, int, int]:
    dataset = make_dataset(dataset_name, data_root)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size
    _, _, test_set = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return loader, len(dataset.classes), len(test_set)


def compute_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float, float]:
    precisions, recalls, f1s = [], [], []
    for cls in range(num_classes):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precisions.append(precision); recalls.append(recall); f1s.append(f1)
    return 100.0 * float(np.mean(precisions)), 100.0 * float(np.mean(recalls)), 100.0 * float(np.mean(f1s))


@torch.no_grad()
def evaluate_pytorch_cpu(model: nn.Module, loader: DataLoader, num_classes: int):
    model.eval()
    all_true, all_pred = [], []
    for images, labels in loader:
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        all_true.append(labels.numpy()); all_pred.append(preds.numpy())
    y_true = np.concatenate(all_true); y_pred = np.concatenate(all_pred)
    acc = 100.0 * float((y_true == y_pred).sum()) / max(1, len(y_true))
    precision, recall, f1 = compute_macro_metrics(y_true, y_pred, num_classes)
    return acc, precision, recall, f1, y_true, y_pred


def evaluate_onnx_cpu(onnx_path: Path, loader: DataLoader, num_classes: int):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    all_true, all_pred = [], []
    for images, labels in loader:
        outputs = session.run(None, {input_name: images.numpy().astype(np.float32, copy=False)})[0]
        preds = np.argmax(outputs, axis=1)
        all_true.append(labels.numpy()); all_pred.append(preds)
    y_true = np.concatenate(all_true); y_pred = np.concatenate(all_pred)
    acc = 100.0 * float((y_true == y_pred).sum()) / max(1, len(y_true))
    precision, recall, f1 = compute_macro_metrics(y_true, y_pred, num_classes)
    return acc, precision, recall, f1, y_true, y_pred


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Input CSV is empty: {csv_path}")
    return rows


def write_rows(csv_path: Path, rows: List[Dict[str, object]], overwrite: bool) -> None:
    if csv_path.exists() and not overwrite:
        raise FileExistsError(f"Output CSV already exists: {csv_path}. Use --overwrite to replace it.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX CPU sanity check for 3-dataset converged models.")
    parser.add_argument("--gpu-results-csv", type=Path, default=DEFAULT_GPU_RESULTS_CSV)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--threshold-pp", type=float, default=DEFAULT_THRESHOLD_PP)
    parser.add_argument("--datasets", nargs="*", default=None, choices=["EuroSAT", "AID", "RESISC45", "eurosat", "aid", "resisc45"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print_banner("ONNX CPU SANITY CHECK: CONVERGED 3-DATASET MODELS")
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] GPU results CSV:    {args.gpu_results_csv}")
    print(f"[INFO] ONNX dir:           {args.onnx_dir}")
    print(f"[INFO] Output CSV:         {args.output_csv}")
    print(f"[INFO] ONNX Runtime:       {ort.__version__}")
    print(f"[INFO] Providers:          {ort.get_available_providers()}")
    print("[IMPORTANT] Turbo/performance mode is NOT required. This is correctness-only.")

    rows = read_rows(args.gpu_results_csv)
    dataset_filter = {normalize_dataset_name(d) for d in args.datasets} if args.datasets else None
    model_filter = {m.lower().strip() for m in args.models} if args.models else None
    selected = []
    for row in rows:
        dataset = normalize_dataset_name(row["dataset"])
        model_name = row["model"].lower().strip()
        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model_name not in model_filter:
            continue
        selected.append(row)
    if not selected:
        raise RuntimeError("No rows selected. Check filters.")
    print(f"[INFO] Rows selected: {len(selected)}")

    loader_cache: Dict[str, Tuple[DataLoader, int, int]] = {}
    output_rows: List[Dict[str, object]] = []
    passed = checked = failed = 0

    for row in selected:
        dataset = normalize_dataset_name(row["dataset"])
        key = dataset_key(dataset)
        model_name = row["model"].lower().strip()
        num_classes = int(row["num_classes"])
        checkpoint_path = PROJECT_ROOT / row["checkpoint_path"]
        onnx_path = args.onnx_dir / onnx_filename(model_name, dataset)
        gpu_csv_acc = float(row["test_accuracy_percent"])
        print_banner(f"CHECKING {dataset} | {model_name}")
        print(f"[INFO] Checkpoint: {checkpoint_path}")
        print(f"[INFO] ONNX:       {onnx_path}")
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            if not onnx_path.exists():
                raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
            if key not in loader_cache:
                loader_cache[key] = deterministic_test_loader(dataset, args.data_root, args.batch_size, args.num_workers)
                print(f"[OK] Prepared {dataset} test loader.")
            loader, detected_classes, num_test_images = loader_cache[key]
            if detected_classes != num_classes:
                raise RuntimeError(f"Class count mismatch: CSV={num_classes}, dataset={detected_classes}")
            model = build_model(model_name, num_classes=num_classes)
            state_dict = clean_state_dict_keys(load_checkpoint_state(checkpoint_path))
            model.load_state_dict(state_dict, strict=True)
            model.cpu(); model.eval()
            print("[OK] Strict PyTorch checkpoint loading passed.")
            pt_acc, pt_prec, pt_rec, pt_f1, y_true_pt, _ = evaluate_pytorch_cpu(model, loader, num_classes)
            onnx_acc, onnx_prec, onnx_rec, onnx_f1, y_true_onnx, _ = evaluate_onnx_cpu(onnx_path, loader, num_classes)
            if not np.array_equal(y_true_pt, y_true_onnx):
                raise RuntimeError("PyTorch and ONNX label arrays differ. Loader mismatch.")
            onnx_vs_pt = onnx_acc - pt_acc
            gpu_vs_pt = gpu_csv_acc - pt_acc
            status = "PASS" if abs(onnx_vs_pt) <= args.threshold_pp else "CHECK"
            passed += int(status == "PASS"); checked += int(status == "CHECK")
            print(f"[RESULT] GPU CSV test acc:   {gpu_csv_acc:.6f}%")
            print(f"[RESULT] PyTorch CPU acc:    {pt_acc:.6f}%")
            print(f"[RESULT] ONNX CPU acc:       {onnx_acc:.6f}%")
            print(f"[RESULT] ONNX-PyTorch diff:  {onnx_vs_pt:+.6f} pp")
            print(f"[{status}] Threshold <= {args.threshold_pp} pp")
            output_rows.append({
                "dataset": dataset, "model": model_name, "num_classes": num_classes,
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "onnx_path": str(onnx_path.relative_to(PROJECT_ROOT)),
                "gpu_csv_test_accuracy_percent": f"{gpu_csv_acc:.6f}",
                "pytorch_cpu_accuracy_percent": f"{pt_acc:.6f}",
                "onnx_cpu_accuracy_percent": f"{onnx_acc:.6f}",
                "onnx_vs_pytorch_diff_percent_points": f"{onnx_vs_pt:.6f}",
                "gpu_csv_vs_pytorch_cpu_diff_percent_points": f"{gpu_vs_pt:.6f}",
                "pytorch_macro_precision_percent": f"{pt_prec:.6f}",
                "pytorch_macro_recall_percent": f"{pt_rec:.6f}",
                "pytorch_macro_f1_percent": f"{pt_f1:.6f}",
                "onnx_macro_precision_percent": f"{onnx_prec:.6f}",
                "onnx_macro_recall_percent": f"{onnx_rec:.6f}",
                "onnx_macro_f1_percent": f"{onnx_f1:.6f}",
                "num_test_images": num_test_images,
                "status": status,
                "notes": "Compared ONNX Runtime CPU vs PyTorch CPU on the exact same deterministic test loader.",
            })
        except Exception as exc:
            failed += 1
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {dataset} | {model_name}: {msg}")
            output_rows.append({
                "dataset": dataset, "model": model_name, "num_classes": num_classes,
                "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.exists() else str(checkpoint_path),
                "onnx_path": str(onnx_path.relative_to(PROJECT_ROOT)) if onnx_path.exists() else str(onnx_path),
                "gpu_csv_test_accuracy_percent": f"{gpu_csv_acc:.6f}",
                "pytorch_cpu_accuracy_percent": "", "onnx_cpu_accuracy_percent": "",
                "onnx_vs_pytorch_diff_percent_points": "", "gpu_csv_vs_pytorch_cpu_diff_percent_points": "",
                "pytorch_macro_precision_percent": "", "pytorch_macro_recall_percent": "", "pytorch_macro_f1_percent": "",
                "onnx_macro_precision_percent": "", "onnx_macro_recall_percent": "", "onnx_macro_f1_percent": "",
                "num_test_images": "", "status": "FAIL", "notes": msg,
            })
    write_rows(args.output_csv, output_rows, overwrite=args.overwrite)
    print_banner("ONNX CPU SANITY SUMMARY")
    print(f"[INFO] Attempted: {len(selected)}")
    print(f"[INFO] Passed:    {passed}")
    print(f"[INFO] Check:     {checked}")
    print(f"[INFO] Failed:    {failed}")
    print(f"[INFO] Output CSV: {args.output_csv}")
    if failed == 0 and checked == 0:
        print("[OK] All ONNX models match PyTorch checkpoints within threshold.")
    else:
        print("[WARN] Some rows require checking. Inspect the output CSV.")


if __name__ == "__main__":
    main()

    
