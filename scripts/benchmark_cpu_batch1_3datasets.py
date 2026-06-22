from __future__ import annotations

r"""

Purpose:
    Official CPU batch-size-1 inference benchmark for the strengthened
    CPU/GPU/NPU remote-sensing paper.

    This script DOES NOT train. It loads the already converged .pth
    checkpoints, evaluates test accuracy as a sanity check, and measures
    single-image CPU inference latency with batch_size=1.

Environment:
    ai_env

Typical commands:
    python main_scripts\benchmark_cpu_batch1_3datasets.py --dataset eurosat --models mobilenet_v2 --smoke-test
    python main_scripts\benchmark_cpu_batch1_3datasets.py --dataset eurosat --models all
    python main_scripts\benchmark_cpu_batch1_3datasets.py --dataset aid --models all
    python main_scripts\benchmark_cpu_batch1_3datasets.py --dataset resisc45 --models all

Output:
    results\cpu_benchmark_results_batch1_3datasets.csv
"""

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------
def infer_project_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "main_scripts":
        return here.parent.parent
    return here.parent


PROJECT_ROOT = infer_project_root()
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "cpu_benchmark_results_batch1_3datasets.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models_converged"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

TRAINING_MODULE_CANDIDATES = [
    PROJECT_ROOT / "main_scripts" / "benchmark_train_gpu_v4_convergence_resisc45.py",
    PROJECT_ROOT / "main_scripts" / "benchmark_train_gpu_v4_convergence.py",
    PROJECT_ROOT / "benchmark_train_gpu_v4_convergence_resisc45.py",
    PROJECT_ROOT / "benchmark_train_gpu_v4_convergence.py",
]

DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 0
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_LATENCY_REPEATS = 3
SEED = 42

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


# Helpers

def print_banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def load_training_module():
    for module_path in TRAINING_MODULE_CANDIDATES:
        if module_path.exists():
            print(f"[INFO] Importing helper functions from: {module_path}")
            module_name = "benchmark_train_gpu_v4_convergence_imported_for_batch1"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not import helper module: {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            required = [
                "ALL_MODELS",
                "create_dataset_bundle",
                "create_model",
                "evaluate_model",
                "measure_latency",
                "set_seed",
                "configure_torch",
                "model_family",
                "count_parameters",
            ]
            missing = [name for name in required if not hasattr(module, name)]
            if missing:
                raise AttributeError(f"Helper module is missing required functions: {missing}")
            return module, module_path

    searched = "\n".join(str(p) for p in TRAINING_MODULE_CANDIDATES)
    raise FileNotFoundError(
        "Could not find the convergence training helper script. Searched:\n" + searched
    )


def parse_models(raw_models: List[str], all_models: List[str]) -> List[str]:
    if len(raw_models) == 1 and raw_models[0].lower() == "all":
        return list(all_models)
    selected: List[str] = []
    for m in raw_models:
        m_lower = m.lower()
        if m_lower not in all_models:
            raise ValueError(f"Unknown model '{m}'. Allowed: {all_models} or all")
        selected.append(m_lower)
    return selected


def checkpoint_candidates(models_dir: Path, model_name: str, dataset_name: str) -> List[Path]:
    return [
        models_dir / f"{model_name}_{dataset_name}_converged.pth",
        models_dir / f"{model_name}_{dataset_name}.pth",
        PROJECT_ROOT / "models" / f"{model_name}_{dataset_name}_converged.pth",
        PROJECT_ROOT / "models" / f"{model_name}_{dataset_name}.pth",
    ]


def find_checkpoint(models_dir: Path, model_name: str, dataset_name: str) -> Path:
    candidates = checkpoint_candidates(models_dir, model_name, dataset_name)
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Checkpoint not found for model={model_name}, dataset={dataset_name}.\nSearched:\n{searched}"
    )


def load_checkpoint_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        return checkpoint
    raise RuntimeError(f"Unsupported checkpoint format: {path}")


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def append_result_csv(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_device() -> torch.device:
    device = torch.device("cpu")
    print("[INFO] Using CPU for official PyTorch CPU benchmark.")
    return device


def maybe_clear_existing_results(results_path: Path, overwrite: bool) -> None:
    if results_path.exists() and overwrite:
        results_path.unlink()
        print(f"[INFO] Deleted old results CSV: {results_path}")
    elif results_path.exists() and not overwrite:
        print(f"[INFO] Appending to existing results CSV: {results_path}")


# Benchmark one model

def run_one_model(
    module,
    model_name: str,
    dataset_bundle,
    dataset_name: str,
    device: torch.device,
    models_dir: Path,
    results_path: Path,
    warmup_batches: int,
    latency_repeats: int,
    smoke_test: bool,
) -> None:
    display_name = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    print_banner(f"CPU BATCH-1 BENCHMARK | {display_name} | {model_name}")

    checkpoint_path = find_checkpoint(models_dir, model_name, dataset_name)
    print(f"[INFO] Checkpoint: {checkpoint_path}")

    model = module.create_model(model_name, num_classes=dataset_bundle.num_classes)
    state_dict = load_checkpoint_state(checkpoint_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print("[OK] Strict checkpoint loading passed.")

    params_million = module.count_parameters(model) / 1_000_000.0
    pth_mb = file_size_mb(checkpoint_path)

    print("[INFO] Evaluating test accuracy at batch_size=1 ...")
    test_metrics = module.evaluate_model(
        model=model,
        dataloader=dataset_bundle.test_loader,
        device=device,
        num_classes=dataset_bundle.num_classes,
    )

    print("[INFO] Measuring official CPU single-image latency at batch_size=1 ...")
    latency = module.measure_latency(
        model=model,
        dataloader=dataset_bundle.test_loader,
        device=device,
        warmup_batches=warmup_batches,
        repeats=latency_repeats,
    )

    print("\n[RESULT]")
    print(f"Dataset:        {display_name}")
    print(f"Model:          {model_name}")
    print(f"Accuracy:       {test_metrics.accuracy_percent:.6f}%")
    print(f"Macro F1:       {test_metrics.f1_macro:.6f}%")
    print(f"Mean latency:   {latency.mean_latency_ms:.6f} ms/image")
    print(f"Median latency: {latency.median_latency_ms:.6f} ms/image")
    print(f"P95 latency:    {latency.p95_latency_ms:.6f} ms/image")
    print(f"FPS:            {latency.fps:.6f}")
    print("Peak CPU memory: N/A / not reported in this script")
    print(f"Repeat means:   {latency.repeat_mean_latencies_ms}")

    row = {
        "run_version": "batch1_single_image_latency",
        "dataset": display_name,
        "model": model_name,
        "model_family": module.model_family(model_name),
        "accelerator": "CPU",
        "runtime": "PyTorch CPU FP32",
        "num_classes": dataset_bundle.num_classes,
        "max_epochs": DATASET_DEFAULT_MAX_EPOCHS.get(dataset_name, ""),
        "epochs_completed": "",
        "best_epoch": "",
        "early_stopped": "",
        "patience": "",
        "min_delta": "",
        "batch_size": 1,
        "learning_rate": "",
        "params_million": f"{params_million:.6f}",
        "pth_size_mb": f"{pth_mb:.6f}",
        "best_val_accuracy_percent": "",
        "best_val_f1_macro": "",
        "test_accuracy_percent": f"{test_metrics.accuracy_percent:.6f}",
        "precision_macro": f"{test_metrics.precision_macro:.6f}",
        "recall_macro": f"{test_metrics.recall_macro:.6f}",
        "f1_macro": f"{test_metrics.f1_macro:.6f}",
        "mean_latency_ms": f"{latency.mean_latency_ms:.6f}",
        "median_latency_ms": f"{latency.median_latency_ms:.6f}",
        "p95_latency_ms": f"{latency.p95_latency_ms:.6f}",
        "fps": f"{latency.fps:.6f}",
        "latency_repeats": latency_repeats,
        "warmup_batches": warmup_batches,
        "repeat_mean_latencies_ms": ";".join(f"{x:.6f}" for x in latency.repeat_mean_latencies_ms),
        "gpu_peak_vram_mb": "",
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT) if checkpoint_path.is_relative_to(PROJECT_ROOT) else checkpoint_path),
        "history_csv": "",
        "notes": (
            "Official CPU batch-size-1 single-image latency rerun; no retraining; "
            "PyTorch CPU FP32 checkpoint; run under AC power and Turbo/performance mode"
            if not smoke_test else
            "SMOKE TEST ONLY - not for paper results"
        ),
    }

    append_result_csv(results_path, row)
    print(f"[INFO] Appended row to: {results_path}")

    del model



# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official CPU batch-size-1 inference benchmark for EuroSAT/AID/RESISC45."
    )
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
        default=["all"],
        help="Model names or 'all'.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_LATENCY_REPEATS)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small subsets only. Results are not valid for the paper.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Delete the output CSV before starting this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size != 1:
        raise ValueError(
            "This official script is intentionally restricted to batch_size=1. "
            "Do not change it for the batch-1 latency rerun."
        )

    module, module_path = load_training_module()
    module.set_seed(SEED)
    module.configure_torch()
    selected_models = parse_models(args.models, list(module.ALL_MODELS))

    maybe_clear_existing_results(args.results_path, overwrite=args.overwrite_results)

    print_banner("OFFICIAL CPU BATCH-1 SINGLE-IMAGE LATENCY BENCHMARK")
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Helper module: {module_path}")
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Models: {selected_models}")
    print(f"[INFO] Data root: {args.data_root}")
    print(f"[INFO] Models dir: {args.models_dir}")
    print(f"[INFO] Results CSV: {args.results_path}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Num workers: {args.num_workers}")
    print(f"[INFO] Warmup batches: {args.warmup_batches}")
    print(f"[INFO] Latency repeats: {args.latency_repeats}")
    print("[IMPORTANT] Official run must be on AC power with Turbo/performance mode ON and Battery Saver OFF.")

    if args.smoke_test:
        print("[WARN] Smoke-test mode enabled. Results are NOT valid for the paper.")

    device = get_device()

    dataset_bundle = module.create_dataset_bundle(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=1,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
    )

    for model_name in selected_models:
        run_one_model(
            module=module,
            model_name=model_name,
            dataset_bundle=dataset_bundle,
            dataset_name=args.dataset,
            device=device,
            models_dir=args.models_dir,
            results_path=args.results_path,
            warmup_batches=args.warmup_batches,
            latency_repeats=args.latency_repeats,
            smoke_test=args.smoke_test,
        )

    print_banner("DONE")
    print(f"[OK] CPU batch-1 benchmark finished: {args.results_path}")
    print("[NEXT] Run the same script for the remaining datasets, then send me the CSV.")


if __name__ == "__main__":
    main()
