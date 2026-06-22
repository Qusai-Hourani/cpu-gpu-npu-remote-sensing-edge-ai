from __future__ import annotations

r"""

Purpose:
    Benchmark the 36 converged ONNX models on the Intel NPU using ONNX Runtime
    with the OpenVINO Execution Provider at batch size 1 for single-image edge latency.

Inputs:
    results\gpu_benchmark_results_batch1_3datasets.csv
    onnx_models_converged_3datasets\*_converged.onnx
    data\EuroSAT torchvision data
    data\AID
    data\NWPU-RESISC45 or data\RESISC45

Output:
    results\npu_benchmark_results_batch1_3datasets.csv


Environment:
    npu_env

Recommended smoke test:
    python main_scripts\benchmark_npu_batch1_3datasets.py --datasets EuroSAT --models resnet18 --output-csv results\npu_benchmark_smoke_v3_strict.csv --overwrite

Recommended full run in chunks:
    python main_scripts\benchmark_npu_batch1_3datasets.py --datasets EuroSAT --overwrite
    python main_scripts\benchmark_npu_batch1_3datasets.py --datasets AID
    python main_scripts\benchmark_npu_batch1_3datasets.py --datasets RESISC45


Important:
    By default this script does NOT allow CPUExecutionProvider fallback and requests ORT CPU EP fallback disabling.
    It requests OpenVINOExecutionProvider with device_type=NPU.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np



# Windows OpenVINO DLL setup before importing onnxruntime

def setup_openvino_dlls_windows() -> None:
    if sys.platform != "win32":
        return

    try:
        from openvino.utils import add_openvino_libs_to_path
        add_openvino_libs_to_path()
        print("[INFO] openvino.utils.add_openvino_libs_to_path() succeeded.")
    except Exception as exc:
        print(f"[WARN] add_openvino_libs_to_path() not available or failed: {exc}")

    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "openvino" / "libs",
        Path(sys.prefix) / "Lib" / "site-packages" / "openvino" / "runtime" / "libs",
        Path(sys.prefix) / "Lib" / "site-packages" / "openvino" / "runtime" / "bin",
        Path(sys.prefix) / "Lib" / "site-packages" / "openvino" / "bin",
    ]

    for p in candidates:
        if p.exists():
            try:
                os.add_dll_directory(str(p))
                print(f"[INFO] Added OpenVINO DLL directory: {p}")
            except Exception as exc:
                print(f"[WARN] Could not add DLL directory {p}: {exc}")


setup_openvino_dlls_windows()

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, random_split  # noqa: E402
from torchvision import datasets, transforms  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GPU_RESULTS_CSV = PROJECT_ROOT / "results" / "gpu_benchmark_results_batch1_3datasets.csv"
DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_models_converged_3datasets"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "results" / "npu_benchmark_results_batch1_3datasets.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

SEED = 42
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 1
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_LATENCY_REPEATS = 3
DEFAULT_NUM_WORKERS = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_COLUMNS = [
    "run_version",
    "dataset",
    "model",
    "model_family",
    "accelerator",
    "runtime",
    "provider_requested",
    "providers_active",
    "provider_options",
    "num_classes",
    "epochs_trained",
    "best_epoch",
    "early_stopped",
    "batch_size",
    "learning_rate",
    "params_million",
    "pth_size_mb",
    "onnx_path",
    "onnx_size_mb",
    "gpu_accuracy_percent",
    "npu_accuracy_percent",
    "accuracy_diff_vs_gpu_percent_points",
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
    "num_test_images",
    "status",
    "notes",
]


def print_banner(title: str, char: str = "=") -> None:
    print("\n" + char * 100)
    print(title)
    print(char * 100)


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


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_resisc45_root(data_root: Path) -> Path:
    candidates = [
        data_root / "RESISC45",
        data_root / "NWPU-RESISC45",
        data_root / "NWPU_RESISC45",
        data_root / "resisc45",
        data_root / "nwpu-resisc45",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            class_dirs = [p for p in candidate.iterdir() if p.is_dir()]
            if len(class_dirs) >= 40:
                return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find RESISC45 folder. Searched:\n{searched}")


def make_base_dataset(dataset_name: str, data_root: Path):
    key = dataset_key(dataset_name)
    transform = get_transform()

    if key == "eurosat":
        return datasets.EuroSAT(root=str(data_root), transform=transform, download=False)

    if key == "aid":
        aid_root = data_root / "AID"
        if not aid_root.exists():
            raise FileNotFoundError(f"AID folder not found: {aid_root}")
        return datasets.ImageFolder(root=str(aid_root), transform=transform)

    if key == "resisc45":
        resisc_root = find_resisc45_root(data_root)
        return datasets.ImageFolder(root=str(resisc_root), transform=transform)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def make_test_loader(dataset_name: str, data_root: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, int, int]:
    base = make_base_dataset(dataset_name, data_root)

    total = len(base)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    _, _, test_set = random_split(
        base,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return loader, len(base.classes), len(test_set)


def read_gpu_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"GPU results CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"GPU results CSV is empty: {csv_path}")

    return rows


def read_existing_completed_keys(output_csv: Path) -> set[tuple[str, str]]:
    if not output_csv.exists():
        return set()

    completed: set[tuple[str, str]] = set()
    try:
        with output_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status", "").upper() == "PASS":
                    completed.add((normalize_dataset_name(row["dataset"]), row["model"].lower().strip()))
    except Exception:
        return set()

    return completed


def append_output_row(output_csv: Path, row: Dict[str, object], write_header: bool) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if write_header else "a"
    with output_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def filter_rows(rows: List[Dict[str, str]], datasets_arg: List[str] | None, models_arg: List[str] | None) -> List[Dict[str, str]]:
    dataset_filter = {normalize_dataset_name(d) for d in datasets_arg} if datasets_arg else None
    model_filter = {m.lower().strip() for m in models_arg} if models_arg else None

    selected = []
    for row in rows:
        dataset = normalize_dataset_name(row["dataset"])
        model_name = row["model"].lower().strip()

        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model_name not in model_filter:
            continue

        selected.append(row)

    return selected


def create_npu_session(onnx_path: Path, allow_cpu_fallback: bool, enable_profiling: bool = False) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if "OpenVINOExecutionProvider" not in available:
        raise RuntimeError(f"OpenVINOExecutionProvider is not available. Available providers: {available}")

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = enable_profiling

    # Stronger safety check:
    # ONNX Runtime 1.16+ supports disabling automatic CPU EP fallback.
    # If OpenVINO/NPU cannot execute a required node, session creation or inference
    # should fail instead of silently falling back.
    try:
        sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        print("[INFO] ONNX Runtime CPU EP fallback disabled via session.disable_cpu_ep_fallback=1")
    except Exception as exc:
        print(f"[WARN] Could not set session.disable_cpu_ep_fallback=1: {exc}")

    openvino_options = {"device_type": "NPU"}

    if allow_cpu_fallback:
        providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        provider_options = [openvino_options, {}]
    else:
        providers = ["OpenVINOExecutionProvider"]
        provider_options = [openvino_options]

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )

    active = session.get_providers()

    if "OpenVINOExecutionProvider" not in active:
        raise RuntimeError(f"OpenVINOExecutionProvider is not active. Active providers: {active}")

    if active[0] != "OpenVINOExecutionProvider":
        raise RuntimeError(f"OpenVINOExecutionProvider is active but not first. Active providers: {active}")

    # ONNX Runtime may still list CPUExecutionProvider even when only OpenVINO EP was
    # requested. For official runs, we require OpenVINOExecutionProvider to be active
    # and first, and we record the full active provider list in the output CSV. We do
    # not automatically treat a listed CPUExecutionProvider as failure because ORT can
    # append it internally on Windows.
    if (not allow_cpu_fallback) and "CPUExecutionProvider" in active:
        print(
            "[NOTE] CPUExecutionProvider appears in session.get_providers(), "
            "but OpenVINOExecutionProvider is active and first. Continuing and recording this in notes."
        )

    return session


def compute_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float, float]:
    precisions = []
    recalls = []
    f1s = []

    for cls in range(num_classes):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return 100.0 * float(np.mean(precisions)), 100.0 * float(np.mean(recalls)), 100.0 * float(np.mean(f1s))


def get_input_name(session: ort.InferenceSession) -> str:
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError("ONNX model has no inputs.")
    return inputs[0].name


def evaluate_accuracy(session: ort.InferenceSession, loader: DataLoader, num_classes: int) -> Tuple[float, float, float, float]:
    input_name = get_input_name(session)

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for images, labels in loader:
        images_np = images.numpy().astype(np.float32, copy=False)
        outputs = session.run(None, {input_name: images_np})[0]
        preds = np.argmax(outputs, axis=1)

        all_true.append(labels.numpy())
        all_pred.append(preds)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    acc = 100.0 * float((y_true == y_pred).sum()) / max(1, len(y_true))
    precision, recall, f1 = compute_macro_metrics(y_true, y_pred, num_classes)
    return acc, precision, recall, f1


def run_warmup(session: ort.InferenceSession, loader: DataLoader, warmup_batches: int) -> None:
    input_name = get_input_name(session)
    completed = 0

    for images, _ in loader:
        images_np = images.numpy().astype(np.float32, copy=False)
        session.run(None, {input_name: images_np})
        completed += 1
        if completed >= warmup_batches:
            break

    print(f"[INFO] Completed warmup batches: {completed}")


def measure_latency(session: ort.InferenceSession, loader: DataLoader, warmup_batches: int, repeats: int) -> Tuple[float, float, float, float, List[float]]:
    input_name = get_input_name(session)

    print(f"[INFO] Measuring NPU latency: warmup_batches={warmup_batches}, repeats={repeats}, one sample per timed batch.")
    run_warmup(session, loader, warmup_batches)

    all_batch_latencies_ms_per_image: List[float] = []
    repeat_means: List[float] = []
    total_images_all = 0
    total_seconds_all = 0.0

    for repeat_idx in range(1, repeats + 1):
        print(f"[INFO] Latency repeat {repeat_idx}/{repeats} ...")
        repeat_images = 0
        repeat_seconds = 0.0

        for images, _ in loader:
            images_np = images.numpy().astype(np.float32, copy=False)
            batch_size = images_np.shape[0]

            start = time.perf_counter()
            session.run(None, {input_name: images_np})
            end = time.perf_counter()

            elapsed = end - start
            batch_ms_per_image = (elapsed / batch_size) * 1000.0

            all_batch_latencies_ms_per_image.append(batch_ms_per_image)
            repeat_images += batch_size
            repeat_seconds += elapsed

        repeat_mean_ms = (repeat_seconds / repeat_images) * 1000.0 if repeat_images else 0.0
        repeat_means.append(float(repeat_mean_ms))
        total_images_all += repeat_images
        total_seconds_all += repeat_seconds

        print(f"[INFO] Repeat {repeat_idx}/{repeats}: mean={repeat_mean_ms:.6f} ms/image over {repeat_images} images")

    mean_latency_ms = (total_seconds_all / total_images_all) * 1000.0 if total_images_all else 0.0
    median_latency_ms = float(np.median(all_batch_latencies_ms_per_image)) if all_batch_latencies_ms_per_image else 0.0
    p95_latency_ms = float(np.percentile(all_batch_latencies_ms_per_image, 95)) if all_batch_latencies_ms_per_image else 0.0
    fps = float(total_images_all / total_seconds_all) if total_seconds_all > 0 else 0.0

    return mean_latency_ms, median_latency_ms, p95_latency_ms, fps, repeat_means


def output_row_from_failure(gpu_row: Dict[str, str], onnx_path: Path, status: str, notes: str, warmup_batches: int, repeats: int) -> Dict[str, object]:
    dataset = normalize_dataset_name(gpu_row.get("dataset", ""))
    model_name = gpu_row.get("model", "").lower().strip()

    return {
        "run_version": "batch1_3datasets_npu",
        "dataset": dataset,
        "model": model_name,
        "model_family": gpu_row.get("model_family", ""),
        "accelerator": "NPU",
        "runtime": "ONNX Runtime + OpenVINO EP",
        "provider_requested": "OpenVINOExecutionProvider(device_type=NPU)",
        "providers_active": "",
        "provider_options": "",
        "num_classes": gpu_row.get("num_classes", ""),
        "epochs_trained": gpu_row.get("epochs_trained", gpu_row.get("epochs", "")),
        "best_epoch": gpu_row.get("best_epoch", ""),
        "early_stopped": gpu_row.get("early_stopped", ""),
        "batch_size": gpu_row.get("batch_size", ""),
        "learning_rate": gpu_row.get("learning_rate", ""),
        "params_million": gpu_row.get("params_million", ""),
        "pth_size_mb": gpu_row.get("pth_size_mb", ""),
        "onnx_path": str(onnx_path),
        "onnx_size_mb": f"{file_size_mb(onnx_path):.6f}" if onnx_path.exists() else "0.000000",
        "gpu_accuracy_percent": gpu_row.get("test_accuracy_percent", ""),
        "npu_accuracy_percent": "",
        "accuracy_diff_vs_gpu_percent_points": "",
        "precision_macro": "",
        "recall_macro": "",
        "f1_macro": "",
        "mean_latency_ms": "",
        "median_latency_ms": "",
        "p95_latency_ms": "",
        "fps": "",
        "latency_repeats": repeats,
        "warmup_batches": warmup_batches,
        "repeat_mean_latencies_ms": "",
        "num_test_images": "",
        "status": status,
        "notes": notes,
    }


def benchmark_one(
    gpu_row: Dict[str, str],
    onnx_dir: Path,
    loader_cache: Dict[str, Tuple[DataLoader, int, int]],
    data_root: Path,
    batch_size: int,
    num_workers: int,
    warmup_batches: int,
    repeats: int,
    allow_cpu_fallback: bool,
) -> Dict[str, object]:
    dataset = normalize_dataset_name(gpu_row["dataset"])
    key = dataset_key(dataset)
    model_name = gpu_row["model"].lower().strip()
    num_classes = int(gpu_row["num_classes"])
    gpu_acc = float(gpu_row["test_accuracy_percent"])

    onnx_path = onnx_dir / onnx_filename(model_name, dataset)

    print_banner(f"NPU BENCHMARK | {dataset} | {model_name}")
    print(f"[INFO] ONNX: {onnx_path}")

    if not onnx_path.exists():
        return output_row_from_failure(gpu_row, onnx_path, "FAIL", f"ONNX file not found: {onnx_path}", warmup_batches, repeats)

    if key not in loader_cache:
        loader_cache[key] = make_test_loader(
            dataset_name=dataset,
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        print(f"[OK] Prepared {dataset} test loader.")

    loader, detected_classes, num_test_images = loader_cache[key]
    if detected_classes != num_classes:
        return output_row_from_failure(
            gpu_row,
            onnx_path,
            "FAIL",
            f"Class count mismatch: CSV={num_classes}, dataset={detected_classes}",
            warmup_batches,
            repeats,
        )

    session = create_npu_session(
        onnx_path=onnx_path,
        allow_cpu_fallback=allow_cpu_fallback,
    )

    active_providers = session.get_providers()
    provider_options = session.get_provider_options()

    print(f"[INFO] Active providers: {active_providers}")
    print(f"[INFO] Provider options: {provider_options}")

    print("[INFO] Evaluating NPU accuracy...")
    npu_acc, precision, recall, f1 = evaluate_accuracy(session, loader, num_classes=num_classes)

    print("[INFO] Measuring official NPU single-image latency at batch_size=1 ...")
    mean_latency, median_latency, p95_latency, fps, repeat_means = measure_latency(
        session=session,
        loader=loader,
        warmup_batches=warmup_batches,
        repeats=repeats,
    )

    diff = npu_acc - gpu_acc

    print_banner(f"FINAL NPU RESULT | {dataset} | {model_name}")
    print(f"GPU CSV accuracy: {gpu_acc:.6f}%")
    print(f"NPU accuracy:     {npu_acc:.6f}%")
    print(f"Accuracy diff:    {diff:+.6f} pp")
    print(f"Macro precision:  {precision:.6f}%")
    print(f"Macro recall:     {recall:.6f}%")
    print(f"Macro F1:         {f1:.6f}%")
    print(f"Mean latency:     {mean_latency:.6f} ms/image")
    print(f"Median latency:   {median_latency:.6f} ms/image")
    print(f"P95 latency:      {p95_latency:.6f} ms/image")
    print(f"FPS:              {fps:.2f}")
    print(f"Repeat means:     {repeat_means}")

    notes = (
        "NPU benchmark using ONNX Runtime + OpenVINO EP with device_type=NPU; "
        "OpenVINOExecutionProvider active and first. CPUExecutionProvider may be listed by ONNX Runtime, "
        "but CPU fallback was not explicitly allowed by this script."
    )
    if "CPUExecutionProvider" in active_providers and not allow_cpu_fallback:
        notes += (
            " CPUExecutionProvider appeared in the provider list; interpreted as ORT provider listing behavior, "
            "not as an automatic failure, because OpenVINOExecutionProvider is first."
        )
    if allow_cpu_fallback:
        notes = (
            "NPU benchmark requested OpenVINO EP with device_type=NPU, but CPU fallback was explicitly allowed; "
            "do not use as official NPU-only latency without checking provider behavior"
        )

    return {
        "run_version": "batch1_3datasets_npu",
        "dataset": dataset,
        "model": model_name,
        "model_family": gpu_row.get("model_family", ""),
        "accelerator": "NPU",
        "runtime": "ONNX Runtime + OpenVINO EP",
        "provider_requested": "OpenVINOExecutionProvider(device_type=NPU)",
        "providers_active": ";".join(active_providers),
        "provider_options": str(provider_options),
        "num_classes": num_classes,
        "epochs_trained": gpu_row.get("epochs_trained", gpu_row.get("epochs", "")),
        "best_epoch": gpu_row.get("best_epoch", ""),
        "early_stopped": gpu_row.get("early_stopped", ""),
        "batch_size": batch_size,
        "learning_rate": gpu_row.get("learning_rate", ""),
        "params_million": gpu_row.get("params_million", ""),
        "pth_size_mb": gpu_row.get("pth_size_mb", ""),
        "onnx_path": str(onnx_path),
        "onnx_size_mb": f"{file_size_mb(onnx_path):.6f}",
        "gpu_accuracy_percent": f"{gpu_acc:.6f}",
        "npu_accuracy_percent": f"{npu_acc:.6f}",
        "accuracy_diff_vs_gpu_percent_points": f"{diff:.6f}",
        "precision_macro": f"{precision:.6f}",
        "recall_macro": f"{recall:.6f}",
        "f1_macro": f"{f1:.6f}",
        "mean_latency_ms": f"{mean_latency:.6f}",
        "median_latency_ms": f"{median_latency:.6f}",
        "p95_latency_ms": f"{p95_latency:.6f}",
        "fps": f"{fps:.6f}",
        "latency_repeats": repeats,
        "warmup_batches": warmup_batches,
        "repeat_mean_latencies_ms": ";".join(f"{x:.6f}" for x in repeat_means),
        "num_test_images": num_test_images,
        "status": "PASS",
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark converged 3-dataset ONNX models on NPU at batch size 1.")
    parser.add_argument("--gpu-results-csv", type=Path, default=DEFAULT_GPU_RESULTS_CSV)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None, choices=["EuroSAT", "AID", "RESISC45", "eurosat", "aid", "resisc45"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size != 1:
        print(f"[WARN] You set --batch-size {args.batch_size}. Official single-image edge latency should use --batch-size 1.")

    print_banner("OFFICIAL NPU BATCH-1 SINGLE-IMAGE LATENCY BENCHMARK")
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] GPU results CSV:    {args.gpu_results_csv}")
    print(f"[INFO] ONNX dir:           {args.onnx_dir}")
    print(f"[INFO] Output CSV:         {args.output_csv}")
    print(f"[INFO] Data root:          {args.data_root}")
    print(f"[INFO] ONNX Runtime:       {ort.__version__}")
    print(f"[INFO] Available providers:{ort.get_available_providers()}")
    print(f"[INFO] Batch size:         {args.batch_size}")
    print(f"[INFO] Warmup batches:     {args.warmup_batches}")
    print(f"[INFO] Latency repeats:    {args.latency_repeats}")
    print("[IMPORTANT] Official latency requires Turbo/performance mode ON, Battery Saver OFF, plugged in.")
    print("[IMPORTANT] By default, CPUExecutionProvider fallback is not allowed, and ORT CPU EP fallback disabling is requested.")

    if args.overwrite and args.output_csv.exists():
        args.output_csv.unlink()
        print(f"[INFO] Deleted old output CSV: {args.output_csv}")

    gpu_rows = read_gpu_rows(args.gpu_results_csv)
    selected_rows = filter_rows(gpu_rows, args.datasets, args.models)

    if not selected_rows:
        raise RuntimeError("No rows selected. Check --datasets / --models filters.")

    completed = read_existing_completed_keys(args.output_csv) if not args.overwrite else set()
    rows_to_run = []
    for row in selected_rows:
        key = (normalize_dataset_name(row["dataset"]), row["model"].lower().strip())
        if key in completed:
            print(f"[SKIP] Already completed: {key[0]} | {key[1]}")
            continue
        rows_to_run.append(row)

    print(f"[INFO] Selected rows: {len(selected_rows)}")
    print(f"[INFO] Rows already completed/skipped: {len(selected_rows) - len(rows_to_run)}")
    print(f"[INFO] Rows to run now: {len(rows_to_run)}")

    if not rows_to_run:
        print("[OK] Nothing to run.")
        return

    loader_cache: Dict[str, Tuple[DataLoader, int, int]] = {}

    passed = 0
    failed = 0
    write_header = not args.output_csv.exists()

    for row in rows_to_run:
        dataset = normalize_dataset_name(row["dataset"])
        model_name = row["model"].lower().strip()
        try:
            output_row = benchmark_one(
                gpu_row=row,
                onnx_dir=args.onnx_dir,
                loader_cache=loader_cache,
                data_root=args.data_root,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                warmup_batches=args.warmup_batches,
                repeats=args.latency_repeats,
                allow_cpu_fallback=args.allow_cpu_fallback,
            )
            if output_row["status"] == "PASS":
                passed += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            onnx_path = args.onnx_dir / onnx_filename(model_name, dataset)
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {dataset} | {model_name}: {msg}")
            output_row = output_row_from_failure(
                gpu_row=row,
                onnx_path=onnx_path,
                status="FAIL",
                notes=msg,
                warmup_batches=args.warmup_batches,
                repeats=args.latency_repeats,
            )

        append_output_row(args.output_csv, output_row, write_header=write_header)
        write_header = False
        print(f"[INFO] Wrote result row to: {args.output_csv}")

    print_banner("NPU BENCHMARK SUMMARY")
    print(f"[INFO] Attempted this run: {len(rows_to_run)}")
    print(f"[INFO] Passed:             {passed}")
    print(f"[INFO] Failed:             {failed}")
    print(f"[INFO] Output CSV:         {args.output_csv}")

    if failed == 0:
        print("[OK] NPU benchmark completed for selected rows.")
    else:
        print("[WARN] Some rows failed. Inspect the status and notes columns in the output CSV.")


if __name__ == "__main__":
    main()
