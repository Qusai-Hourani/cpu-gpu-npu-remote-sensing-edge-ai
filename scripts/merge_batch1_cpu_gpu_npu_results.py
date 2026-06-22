r"""

Purpose:

This script builds the official single-image edge-inference result tables for
Qusai's CPU/GPU/NPU remote-sensing benchmark. It is designed to be strict:
it validates row counts, dataset/model coverage, batch size, accelerators,
NPU PASS status, duplicate rows, and accuracy preservation before producing
paper-ready summary tables.


Inputs expected by default:
    results\cpu_benchmark_results_batch1_3datasets.csv
    results\gpu_benchmark_results_batch1_3datasets.csv
    results\npu_benchmark_results_batch1_3datasets.csv

Outputs:
    results\final_cpu_gpu_npu_batch1_comparison_3datasets.csv
    results\table2_dataset_summary_batch1.csv
    results\table2_dataset_summary_batch1.md
    results\table3_best_fastest_batch1.csv
    results\table3_best_fastest_batch1.md
    results\batch1_merge_validation_report.txt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

DATASET_ORDER = ["EuroSAT", "AID", "RESISC45"]
MODEL_ORDER = [
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
MODEL_PRETTY = {
    "resnet18": "ResNet18",
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "shufflenet_v2_x1_0": "ShuffleNetV2",
    "convnext_tiny": "ConvNeXt-Tiny",
    "repvit_m0_9": "RepViT-M0.9",
    "starnet_s4": "StarNet-S4",
    "shvit_s4": "SHViT-S4",
    "mambaout_kobe": "MambaOut-Kobe",
}

EXPECTED_ROWS = len(DATASET_ORDER) * len(MODEL_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge official batch-1 CPU/GPU/NPU benchmark CSVs.")
    parser.add_argument("--cpu-csv", default="results/cpu_benchmark_results_batch1_3datasets.csv")
    parser.add_argument("--gpu-csv", default="results/gpu_benchmark_results_batch1_3datasets.csv")
    parser.add_argument("--npu-csv", default="results/npu_benchmark_results_batch1_3datasets.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--accuracy-tolerance-pp", type=float, default=0.15,
                        help="Maximum allowed absolute NPU/GPU accuracy difference in percentage points before warning.")
    parser.add_argument("--strict", action="store_true",
                        help="Turn selected warnings into hard errors.")
    return parser.parse_args()


def normalize_dataset_name(x: object) -> str:
    s = str(x).strip()
    low = s.lower()
    if low in {"eurosat", "euro_sat"}:
        return "EuroSAT"
    if low == "aid":
        return "AID"
    if low in {"resisc45", "nwpu-resisc45", "nwpu_resisc45", "resisc"}:
        return "RESISC45"
    return s


def read_csv(path: Path, expected_accelerator: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "model", "accelerator", "batch_size", "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "fps"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["dataset"] = df["dataset"].map(normalize_dataset_name)
    df["model"] = df["model"].astype(str).str.strip()
    df["accelerator"] = df["accelerator"].astype(str).str.strip().str.upper()
    exp = expected_accelerator.upper()
    bad_acc = sorted(set(df["accelerator"]) - {exp})
    if bad_acc:
        raise ValueError(f"{path} contains unexpected accelerator values {bad_acc}; expected only {exp}.")
    return df


def validate_coverage(df: pd.DataFrame, name: str, report: list[str], strict: bool) -> None:
    report.append(f"[{name}] rows: {len(df)}")
    if len(df) != EXPECTED_ROWS:
        msg = f"[{name}] expected {EXPECTED_ROWS} rows but found {len(df)}."
        if strict:
            raise ValueError(msg)
        report.append("WARNING: " + msg)

    dup = df.duplicated(subset=["dataset", "model"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["dataset", "model"]].sort_values(["dataset", "model"])
        raise ValueError(f"[{name}] duplicate dataset/model rows found:\n{bad.to_string(index=False)}")

    expected_pairs = {(d, m) for d in DATASET_ORDER for m in MODEL_ORDER}
    actual_pairs = {(r.dataset, r.model) for r in df[["dataset", "model"]].itertuples(index=False)}
    missing = sorted(expected_pairs - actual_pairs)
    extra = sorted(actual_pairs - expected_pairs)
    if missing:
        raise ValueError(f"[{name}] missing dataset/model pairs: {missing}")
    if extra:
        msg = f"[{name}] extra dataset/model pairs not in official set: {extra}"
        if strict:
            raise ValueError(msg)
        report.append("WARNING: " + msg)

    bad_batch = df.loc[pd.to_numeric(df["batch_size"], errors="coerce") != 1, ["dataset", "model", "batch_size"]]
    if not bad_batch.empty:
        raise ValueError(f"[{name}] all rows must have batch_size=1. Bad rows:\n{bad_batch.to_string(index=False)}")

    for col in ["mean_latency_ms", "median_latency_ms", "p95_latency_ms", "fps"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any() or (vals <= 0).any():
            bad = df.loc[vals.isna() | (vals <= 0), ["dataset", "model", col]]
            raise ValueError(f"[{name}] invalid {col} values:\n{bad.to_string(index=False)}")


def build_accel_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # CPU/GPU use test_accuracy_percent; NPU uses npu_accuracy_percent.
    if prefix == "npu":
        acc_col = "npu_accuracy_percent" if "npu_accuracy_percent" in df.columns else "test_accuracy_percent"
    else:
        acc_col = "test_accuracy_percent"

    cols = [
        "dataset", "model", "model_family", "runtime", "batch_size",
        acc_col, "precision_macro", "recall_macro", "f1_macro",
        "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "fps",
        "params_million", "pth_size_mb", "notes",
    ]
    available = [c for c in cols if c in df.columns]
    out = df[available].copy()
    rename = {
        "runtime": f"{prefix}_runtime",
        "batch_size": f"{prefix}_batch_size",
        acc_col: f"{prefix}_accuracy_percent",
        "precision_macro": f"{prefix}_precision_macro",
        "recall_macro": f"{prefix}_recall_macro",
        "f1_macro": f"{prefix}_f1_macro",
        "mean_latency_ms": f"{prefix}_mean_latency_ms",
        "median_latency_ms": f"{prefix}_median_latency_ms",
        "p95_latency_ms": f"{prefix}_p95_latency_ms",
        "fps": f"{prefix}_fps",
        "notes": f"{prefix}_notes",
    }
    out = out.rename(columns=rename)
    return out


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_order"] = df["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)})
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df = df.sort_values(["dataset_order", "model_order"]).drop(columns=["dataset_order", "model_order"])
    return df


def fmt_pct(x: float) -> str:
    return f"{x:.2f}"


def fmt_ms(x: float) -> str:
    return f"{x:.2f}"


def fmt_ratio(x: float) -> str:
    return f"{x:.2f}x"


def format_model_value(model: str, value: float, suffix: str, decimals: int = 2) -> str:
    pretty = MODEL_PRETTY.get(model, model)
    return f"{pretty} ({value:.{decimals}f} {suffix})"


def make_markdown_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_path = Path(args.cpu_csv)
    gpu_path = Path(args.gpu_csv)
    npu_path = Path(args.npu_csv)

    report: list[str] = []
    report.append("BATCH-1 CPU/GPU/NPU MERGE VALIDATION REPORT")
    report.append("=" * 70)
    report.append(f"Project root: {project_root}")
    report.append(f"CPU CSV: {cpu_path}")
    report.append(f"GPU CSV: {gpu_path}")
    report.append(f"NPU CSV: {npu_path}")

    cpu = read_csv(cpu_path, "CPU")
    gpu = read_csv(gpu_path, "GPU")
    npu = read_csv(npu_path, "NPU")

    validate_coverage(cpu, "CPU", report, args.strict)
    validate_coverage(gpu, "GPU", report, args.strict)
    validate_coverage(npu, "NPU", report, args.strict)

    if "status" in npu.columns:
        bad_status = npu.loc[npu["status"].astype(str).str.upper() != "PASS", ["dataset", "model", "status"]]
        if not bad_status.empty:
            raise ValueError(f"[NPU] Non-PASS rows found:\n{bad_status.to_string(index=False)}")
        report.append("[NPU] status check: all PASS")

    cpu_f = build_accel_frame(cpu, "cpu")
    gpu_f = build_accel_frame(gpu, "gpu")
    npu_f = build_accel_frame(npu, "npu")

    merged = cpu_f.merge(gpu_f.drop(columns=[c for c in ["model_family", "params_million", "pth_size_mb"] if c in gpu_f.columns]),
                         on=["dataset", "model"], how="inner", validate="one_to_one")
    merged = merged.merge(npu_f.drop(columns=[c for c in ["model_family", "params_million", "pth_size_mb"] if c in npu_f.columns]),
                          on=["dataset", "model"], how="inner", validate="one_to_one")

    # Ensure numeric columns.
    numeric_cols = [c for c in merged.columns if c.endswith("_ms") or c.endswith("_fps") or c.endswith("_percent") or c in {"params_million", "pth_size_mb"}]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Derived model-level ratios. These are official ratios for summary averaging.
    merged["gpu_cpu_speedup"] = merged["cpu_mean_latency_ms"] / merged["gpu_mean_latency_ms"]
    merged["npu_cpu_speedup"] = merged["cpu_mean_latency_ms"] / merged["npu_mean_latency_ms"]
    merged["npu_gpu_latency_ratio"] = merged["npu_mean_latency_ms"] / merged["gpu_mean_latency_ms"]
    merged["gpu_npu_latency_ratio"] = merged["gpu_mean_latency_ms"] / merged["npu_mean_latency_ms"]
    merged["cpu_gpu_accuracy_diff_pp"] = merged["cpu_accuracy_percent"] - merged["gpu_accuracy_percent"]
    merged["npu_gpu_accuracy_diff_pp"] = merged["npu_accuracy_percent"] - merged["gpu_accuracy_percent"]
    merged["model_display"] = merged["model"].map(MODEL_PRETTY).fillna(merged["model"])

    # Validation checks on merged data.
    if len(merged) != EXPECTED_ROWS:
        raise ValueError(f"Merged row count is {len(merged)}, expected {EXPECTED_ROWS}.")

    max_abs_cpu_gpu = float(merged["cpu_gpu_accuracy_diff_pp"].abs().max())
    max_abs_npu_gpu = float(merged["npu_gpu_accuracy_diff_pp"].abs().max())
    report.append(f"Max |CPU-GPU accuracy diff|: {max_abs_cpu_gpu:.6f} percentage points")
    report.append(f"Max |NPU-GPU accuracy diff|: {max_abs_npu_gpu:.6f} percentage points")
    if max_abs_npu_gpu > args.accuracy_tolerance_pp:
        msg = f"Max NPU/GPU accuracy difference exceeds tolerance {args.accuracy_tolerance_pp} pp."
        if args.strict:
            raise ValueError(msg)
        report.append("WARNING: " + msg)

    # Count winners per dataset and overall.
    winner_cols = ["cpu_mean_latency_ms", "gpu_mean_latency_ms", "npu_mean_latency_ms"]
    winner_map = {"cpu_mean_latency_ms": "CPU", "gpu_mean_latency_ms": "GPU", "npu_mean_latency_ms": "NPU"}
    merged["fastest_accelerator"] = merged[winner_cols].idxmin(axis=1).map(winner_map)
    report.append("Fastest accelerator counts overall:")
    for accel, cnt in merged["fastest_accelerator"].value_counts().sort_index().items():
        report.append(f"  {accel}: {cnt}")
    report.append("Fastest accelerator counts by dataset:")
    for dataset in DATASET_ORDER:
        sub = merged[merged["dataset"] == dataset]
        counts = sub["fastest_accelerator"].value_counts().to_dict()
        report.append(f"  {dataset}: " + ", ".join(f"{k}={counts.get(k,0)}" for k in ["CPU", "GPU", "NPU"]))

    # Output detailed merged table
    detailed_cols = [
        "dataset", "model", "model_display", "model_family", "params_million", "pth_size_mb",
        "cpu_accuracy_percent", "gpu_accuracy_percent", "npu_accuracy_percent",
        "cpu_gpu_accuracy_diff_pp", "npu_gpu_accuracy_diff_pp",
        "cpu_mean_latency_ms", "gpu_mean_latency_ms", "npu_mean_latency_ms",
        "cpu_median_latency_ms", "gpu_median_latency_ms", "npu_median_latency_ms",
        "cpu_p95_latency_ms", "gpu_p95_latency_ms", "npu_p95_latency_ms",
        "cpu_fps", "gpu_fps", "npu_fps",
        "gpu_cpu_speedup", "npu_cpu_speedup", "npu_gpu_latency_ratio", "gpu_npu_latency_ratio",
        "fastest_accelerator", "cpu_runtime", "gpu_runtime", "npu_runtime",
    ]
    detailed_cols = [c for c in detailed_cols if c in merged.columns]
    detailed = ordered(merged[detailed_cols])
    final_path = out_dir / "final_cpu_gpu_npu_batch1_comparison_3datasets.csv"
    detailed.to_csv(final_path, index=False)

    # Table II dataset-level summary
    summary_rows = []
    for dataset in DATASET_ORDER:
        sub = merged[merged["dataset"] == dataset].copy()
        row = {
            "Dataset": dataset,
            "CPU Acc. (%)": sub["cpu_accuracy_percent"].mean(),
            "GPU Acc. (%)": sub["gpu_accuracy_percent"].mean(),
            "NPU Acc. (%)": sub["npu_accuracy_percent"].mean(),
            "CPU Lat. (ms/img)": sub["cpu_mean_latency_ms"].mean(),
            "GPU Lat. (ms/img)": sub["gpu_mean_latency_ms"].mean(),
            "NPU Lat. (ms/img)": sub["npu_mean_latency_ms"].mean(),
            "GPU/CPU Speedup": sub["gpu_cpu_speedup"].mean(),
            "NPU/CPU Speedup": sub["npu_cpu_speedup"].mean(),
            "NPU/GPU Latency Ratio": sub["npu_gpu_latency_ratio"].mean(),
            "NPU Faster Than GPU (count)": int((sub["npu_mean_latency_ms"] < sub["gpu_mean_latency_ms"]).sum()),
        }
        # Diagnostic ratios derived from displayed mean latency columns. Not official speedup columns.
        row["GPU/CPU Ratio From Mean Latencies"] = row["CPU Lat. (ms/img)"] / row["GPU Lat. (ms/img)"]
        row["NPU/CPU Ratio From Mean Latencies"] = row["CPU Lat. (ms/img)"] / row["NPU Lat. (ms/img)"]
        row["NPU/GPU Ratio From Mean Latencies"] = row["NPU Lat. (ms/img)"] / row["GPU Lat. (ms/img)"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    table2_csv = out_dir / "table2_dataset_summary_batch1.csv"
    summary.to_csv(table2_csv, index=False)

    table2_display = summary[[
        "Dataset", "CPU Acc. (%)", "GPU Acc. (%)", "NPU Acc. (%)",
        "CPU Lat. (ms/img)", "GPU Lat. (ms/img)", "NPU Lat. (ms/img)",
        "GPU/CPU Speedup", "NPU/CPU Speedup", "NPU/GPU Latency Ratio", "NPU Faster Than GPU (count)"
    ]].copy()
    for c in ["CPU Acc. (%)", "GPU Acc. (%)", "NPU Acc. (%)"]:
        table2_display[c] = table2_display[c].map(lambda x: f"{x:.2f}")
    for c in ["CPU Lat. (ms/img)", "GPU Lat. (ms/img)", "NPU Lat. (ms/img)"]:
        table2_display[c] = table2_display[c].map(lambda x: f"{x:.2f}")
    for c in ["GPU/CPU Speedup", "NPU/CPU Speedup", "NPU/GPU Latency Ratio"]:
        table2_display[c] = table2_display[c].map(lambda x: f"{x:.2f}x")
    table2_display["NPU Faster Than GPU (count)"] = table2_display["NPU Faster Than GPU (count)"].map(lambda x: f"{x}/12")
    make_markdown_table(table2_display, out_dir / "table2_dataset_summary_batch1.md")

    # Table III: best accuracy and fastest latency models per dataset.
    best_rows = []
    for dataset in DATASET_ORDER:
        sub = merged[merged["dataset"] == dataset].copy()
        # For best accuracy, use the maximum across the three deployment paths for each row.
        sub["best_any_accuracy_percent"] = sub[["cpu_accuracy_percent", "gpu_accuracy_percent", "npu_accuracy_percent"]].max(axis=1)
        best_acc = sub.sort_values(["best_any_accuracy_percent", "model"], ascending=[False, True]).iloc[0]
        fastest_cpu = sub.sort_values(["cpu_mean_latency_ms", "model"], ascending=[True, True]).iloc[0]
        fastest_gpu = sub.sort_values(["gpu_mean_latency_ms", "model"], ascending=[True, True]).iloc[0]
        fastest_npu = sub.sort_values(["npu_mean_latency_ms", "model"], ascending=[True, True]).iloc[0]
        best_rows.append({
            "Dataset": dataset,
            "Best Accuracy Model": format_model_value(best_acc["model"], float(best_acc["best_any_accuracy_percent"]), "%", 2),
            "Fastest CPU Model": format_model_value(fastest_cpu["model"], float(fastest_cpu["cpu_mean_latency_ms"]), "ms/img", 2),
            "Fastest GPU Model": format_model_value(fastest_gpu["model"], float(fastest_gpu["gpu_mean_latency_ms"]), "ms/img", 2),
            "Fastest NPU Model": format_model_value(fastest_npu["model"], float(fastest_npu["npu_mean_latency_ms"]), "ms/img", 2),
        })
    table3 = pd.DataFrame(best_rows)
    table3_csv = out_dir / "table3_best_fastest_batch1.csv"
    table3.to_csv(table3_csv, index=False)
    make_markdown_table(table3, out_dir / "table3_best_fastest_batch1.md")

    # Write validation report with exact summary values.
    report.append("")
    report.append("Dataset-level official Table II values:")
    for _, r in summary.iterrows():
        report.append(
            f"  {r['Dataset']}: CPU Acc={r['CPU Acc. (%)']:.4f}, GPU Acc={r['GPU Acc. (%)']:.4f}, NPU Acc={r['NPU Acc. (%)']:.4f}; "
            f"CPU Lat={r['CPU Lat. (ms/img)']:.6f}, GPU Lat={r['GPU Lat. (ms/img)']:.6f}, NPU Lat={r['NPU Lat. (ms/img)']:.6f}; "
            f"mean model-ratio GPU/CPU={r['GPU/CPU Speedup']:.6f}, NPU/CPU={r['NPU/CPU Speedup']:.6f}, NPU/GPU={r['NPU/GPU Latency Ratio']:.6f}; "
            f"ratios from mean latencies GPU/CPU={r['GPU/CPU Ratio From Mean Latencies']:.6f}, NPU/CPU={r['NPU/CPU Ratio From Mean Latencies']:.6f}, NPU/GPU={r['NPU/GPU Ratio From Mean Latencies']:.6f}; "
            f"NPU faster than GPU={int(r['NPU Faster Than GPU (count)'])}/12"
        )
    report.append("")
    report.append("Output files:")
    for p in [final_path, table2_csv, out_dir / "table2_dataset_summary_batch1.md", table3_csv, out_dir / "table3_best_fastest_batch1.md"]:
        report.append(f"  {p}")
    report_path = out_dir / "batch1_merge_validation_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print("\n[OK] Batch-1 CPU/GPU/NPU merge completed successfully.")


if __name__ == "__main__":
    main()


