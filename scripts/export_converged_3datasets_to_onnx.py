from __future__ import annotations

r"""

Purpose:
    Export the 36 converged checkpoints from the 3-dataset GPU experiment to ONNX.

Inputs:
    results\gpu_benchmark_results_converged_3datasets.csv

Outputs:
    onnx_models_converged_3datasets\*_converged.onnx
    results\onnx_export_results_converged_3datasets.csv


Environment:
    ai_env

"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GPU_RESULTS_CSV = PROJECT_ROOT / "results" / "gpu_benchmark_results_converged_3datasets.csv"
DEFAULT_ONNX_DIR = PROJECT_ROOT / "onnx_models_converged_3datasets"
DEFAULT_EXPORT_RESULTS_CSV = PROJECT_ROOT / "results" / "onnx_export_results_converged_3datasets.csv"

IMAGE_SIZE = 224
OPSET_VERSION = 18

EXPECTED_MODELS = [
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

EXPECTED_DATASETS = ["EuroSAT", "AID", "RESISC45"]

EXPORT_COLUMNS = [
    "dataset",
    "model",
    "num_classes",
    "checkpoint_path",
    "onnx_path",
    "onnx_size_mb",
    "opset_version",
    "status",
    "message",
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
    value = str(value).strip()
    lower = value.lower()

    if lower == "eurosat":
        return "EuroSAT"
    if lower == "aid":
        return "AID"
    if lower in ["resisc45", "nwpu-resisc45", "nwpu_resisc45"]:
        return "RESISC45"

    return value


def dataset_slug(value: str) -> str:
    return normalize_dataset_name(value).lower()


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
            raise ImportError("repvit_m0_9 requires timm in ai_env.")
        return timm.create_model(
            "repvit_m0_9.dist_450e_in1k",
            pretrained=False,
            num_classes=num_classes,
        )

    if name == "starnet_s4":
        if not HAS_TIMM:
            raise ImportError("starnet_s4 requires timm in ai_env.")
        return timm.create_model(
            "starnet_s4",
            pretrained=False,
            num_classes=num_classes,
        )

    if name == "shvit_s4":
        if not HAS_TIMM:
            raise ImportError("shvit_s4 requires timm in ai_env.")
        return timm.create_model(
            "shvit_s4",
            pretrained=False,
            num_classes=num_classes,
        )

    if name == "mambaout_kobe":
        if not HAS_TIMM:
            raise ImportError("mambaout_kobe requires timm in ai_env.")
        return timm.create_model(
            "mambaout_kobe",
            pretrained=False,
            num_classes=num_classes,
        )

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
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value

    return cleaned


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing GPU results CSV: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    return rows


def verify_input_rows(rows: List[Dict[str, str]]) -> None:
    required = ["dataset", "model", "num_classes", "checkpoint_path"]
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise RuntimeError(f"Input CSV missing required column(s): {missing}")

    seen = set()
    problems = []

    for row in rows:
        dataset = normalize_dataset_name(row["dataset"])
        model = row["model"].lower().strip()
        key = (dataset, model)

        if key in seen:
            problems.append(f"Duplicate row: {dataset} / {model}")
        seen.add(key)

    for dataset in EXPECTED_DATASETS:
        for model in EXPECTED_MODELS:
            if (dataset, model) not in seen:
                problems.append(f"Missing row: {dataset} / {model}")

    if len(rows) != len(EXPECTED_DATASETS) * len(EXPECTED_MODELS):
        problems.append(f"Expected 36 rows, found {len(rows)}")

    if problems:
        raise RuntimeError("Input verification failed:\n" + "\n".join(problems))

    print("[OK] Input CSV verification passed: 36 rows = 3 datasets x 12 models.")


def onnx_filename(model_name: str, dataset_name: str) -> str:
    return f"{model_name.lower()}_{dataset_slug(dataset_name)}_converged.onnx"


def file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def try_onnx_check(onnx_path: Path) -> str:
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        return "ONNX checker passed"
    except ImportError:
        return "ONNX package not available; checker skipped"
    except Exception as exc:
        raise RuntimeError(f"ONNX checker failed: {type(exc).__name__}: {exc}") from exc


def export_one(row: Dict[str, str], onnx_dir: Path, force: bool) -> Dict[str, object]:
    dataset = normalize_dataset_name(row["dataset"])
    model_name = row["model"].lower().strip()
    num_classes = int(row["num_classes"])

    checkpoint_path = PROJECT_ROOT / row["checkpoint_path"]
    onnx_path = onnx_dir / onnx_filename(model_name, dataset)

    result = {
        "dataset": dataset,
        "model": model_name,
        "num_classes": num_classes,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)) if checkpoint_path.exists() else str(checkpoint_path),
        "onnx_path": str(onnx_path.relative_to(PROJECT_ROOT)),
        "onnx_size_mb": "",
        "opset_version": OPSET_VERSION,
        "status": "FAIL",
        "message": "",
    }

    print_banner(f"EXPORTING {dataset} | {model_name}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] ONNX path:  {onnx_path}")

    if not checkpoint_path.exists():
        msg = f"Checkpoint not found: {checkpoint_path}"
        print(f"[FAIL] {msg}")
        result["message"] = msg
        return result

    if onnx_path.exists() and not force:
        msg = "ONNX already exists; skipped. Use --force to overwrite."
        print(f"[SKIP] {msg}")
        result["status"] = "SKIP"
        result["onnx_size_mb"] = f"{file_size_mb(onnx_path):.6f}"
        result["message"] = msg
        return result

    try:
        model = build_model(model_name, num_classes=num_classes)
        model.eval()
        model.cpu()

        state_dict = clean_state_dict_keys(load_checkpoint_state(checkpoint_path))
        model.load_state_dict(state_dict, strict=True)
        print("[OK] Strict checkpoint loading passed.")

        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

        with torch.no_grad():
            out = model(dummy)

        expected_shape = (1, num_classes)
        if tuple(out.shape) != expected_shape:
            raise RuntimeError(f"Unexpected PyTorch output shape: got {tuple(out.shape)}, expected {expected_shape}")

        print(f"[OK] PyTorch forward check passed: output={tuple(out.shape)}")

        onnx_dir.mkdir(parents=True, exist_ok=True)

        export_kwargs = dict(
            model=model,
            args=dummy,
            f=str(onnx_path),
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        # dynamo=False avoids some dynamic-shape problems seen in newer PyTorch ONNX export paths.
        try:
            torch.onnx.export(**export_kwargs, dynamo=False)
        except TypeError:
            torch.onnx.export(**export_kwargs)

        size_mb = file_size_mb(onnx_path)
        if size_mb <= 0:
            raise RuntimeError("ONNX file was not written or has zero size.")

        checker_message = try_onnx_check(onnx_path)

        result["onnx_size_mb"] = f"{size_mb:.6f}"
        result["status"] = "PASS"
        result["message"] = f"Export successful. {checker_message}"

        print(f"[OK] Exported: {onnx_path}")
        print(f"[INFO] ONNX size: {size_mb:.2f} MB")
        print(f"[OK] {checker_message}")

    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"[FAIL] {dataset} | {model_name}: {msg}")
        result["onnx_size_mb"] = f"{file_size_mb(onnx_path):.6f}" if onnx_path.exists() else "0.000000"
        result["message"] = msg

    return result


def write_results(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def filter_rows(rows: List[Dict[str, str]], datasets: List[str] | None, models_filter: List[str] | None) -> List[Dict[str, str]]:
    dataset_filter = {normalize_dataset_name(d) for d in datasets} if datasets else None
    model_filter = {m.lower().strip() for m in models_filter} if models_filter else None

    selected = []

    for row in rows:
        dataset = normalize_dataset_name(row["dataset"])
        model = row["model"].lower().strip()

        if dataset_filter and dataset not in dataset_filter:
            continue
        if model_filter and model not in model_filter:
            continue

        selected.append(row)

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3-dataset converged checkpoints to ONNX.")
    parser.add_argument("--gpu-results-csv", type=Path, default=DEFAULT_GPU_RESULTS_CSV)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--export-results-csv", type=Path, default=DEFAULT_EXPORT_RESULTS_CSV)
    parser.add_argument("--datasets", nargs="*", default=None, choices=["EuroSAT", "AID", "RESISC45", "eurosat", "aid", "resisc45"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing ONNX files.")
    args = parser.parse_args()

    print_banner("EXPORT CONVERGED 3-DATASET CHECKPOINTS TO ONNX")
    print(f"[INFO] Project root:       {PROJECT_ROOT}")
    print(f"[INFO] GPU results CSV:    {args.gpu_results_csv}")
    print(f"[INFO] Output ONNX dir:    {args.onnx_dir}")
    print(f"[INFO] Export results CSV: {args.export_results_csv}")
    print(f"[INFO] Torch version:      {torch.__version__}")
    print(f"[INFO] ONNX opset:         {OPSET_VERSION}")
    print(f"[INFO] timm available:     {HAS_TIMM}")
    print("[IMPORTANT] Turbo/performance mode is NOT required for ONNX export.")
    print("[IMPORTANT] This script does not train or modify checkpoints/results.")

    rows = read_rows(args.gpu_results_csv)
    verify_input_rows(rows)

    selected_rows = filter_rows(rows, args.datasets, args.models)
    if not selected_rows:
        raise RuntimeError("No rows selected. Check --datasets / --models filters.")

    print(f"[INFO] Rows selected for export: {len(selected_rows)}")

    results = []
    passed = 0
    skipped = 0
    failed = 0

    for row in selected_rows:
        result = export_one(row, args.onnx_dir, args.force)
        results.append(result)
        write_results(args.export_results_csv, results)

        if result["status"] == "PASS":
            passed += 1
        elif result["status"] == "SKIP":
            skipped += 1
        else:
            failed += 1

    print_banner("ONNX EXPORT SUMMARY")
    print(f"[INFO] Attempted/selected: {len(selected_rows)}")
    print(f"[INFO] Passed:             {passed}")
    print(f"[INFO] Skipped:            {skipped}")
    print(f"[INFO] Failed:             {failed}")
    print(f"[INFO] Export results CSV: {args.export_results_csv}")
    print(f"[INFO] ONNX directory:     {args.onnx_dir}")

    if failed == 0:
        print("[OK] ONNX export completed without failures.")
    else:
        print("[WARN] Some exports failed. Open the export results CSV and inspect the message column.")


if __name__ == "__main__":
    main()



