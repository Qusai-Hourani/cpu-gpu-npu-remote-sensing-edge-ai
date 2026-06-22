"""
plot_batch1_cpu_gpu_npu_results_publication.py

Default inputs:
    results/final_cpu_gpu_npu_batch1_comparison_3datasets.csv
    results/table2_dataset_summary_batch1.csv
    results/table3_best_fastest_batch1.csv

Default output:
    results/batch1_figures_publication
"""

from __future__ import annotations

import argparse
import math
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
import matplotlib.patheffects as pe

DATASET_ORDER = ["EuroSAT", "AID", "RESISC45"]
ACCELS = ["CPU", "GPU", "NPU"]
ACCEL_COLORS = {"CPU": "#6B7280", "GPU": "#2563EB", "NPU": "#059669"}
ACCEL_HATCHES = {"CPU": "", "GPU": "//", "NPU": "\\\\"}

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
MODEL_DISPLAY = {
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

MAIN_FIGURES = set()


def setup_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "main_paper").mkdir(parents=True, exist_ok=True)
    (out_dir / "supplementary").mkdir(parents=True, exist_ok=True)
    (out_dir / "data_exports").mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_dir: Path, base: str, main: bool, index_rows: List[Dict[str, str]], note: str) -> None:
    folder = out_dir / ("main_paper" if main else "supplementary")
    png = folder / f"{base}.png"
    pdf = folder / f"{base}.pdf"
    # bbox_inches='tight' is intentionally used to prevent cut-off labels.
    fig.savefig(png, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    
    pdf_value = ""
    if main:
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.18, facecolor="white")
        pdf_value = str(pdf.relative_to(out_dir))
    plt.close(fig)
    index_rows.append({
        "file": str(png.relative_to(out_dir)),
        "pdf": pdf_value,
        "recommended_for_main_paper": "YES" if main else "NO",
        "note": note,
    })


def accel_legend_handles() -> List[Patch]:
    return [Patch(facecolor=ACCEL_COLORS[a], edgecolor="black", hatch=ACCEL_HATCHES[a], label=a, linewidth=0.7) for a in ACCELS]


def model_labels(df: pd.DataFrame) -> List[str]:
    mapping = df.drop_duplicates("model").set_index("model")["model_display"].to_dict()
    return [mapping.get(m, MODEL_DISPLAY.get(m, m)) for m in MODEL_ORDER if m in mapping or m in df["model"].values]


def ordered_models(df: pd.DataFrame) -> List[str]:
    present = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in present] + [m for m in sorted(present) if m not in MODEL_ORDER]


def relative_luminance(rgb) -> float:
    """Return perceived luminance for an RGB or RGBA color tuple."""
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_text_style(bg_color=None, prefer_dark: bool | None = None) -> Dict[str, object]:
    """
    Choose a high-contrast text color plus an outline stroke.

    This is deliberately used for all heatmap annotations and bar-value labels so
    numeric values remain visible on both dark and light backgrounds.
    """
    if prefer_dark is None:
        if bg_color is None:
            prefer_dark = True
        else:
            prefer_dark = relative_luminance(bg_color) > 0.55
    fg = "black" if prefer_dark else "white"
    stroke = "white" if fg == "black" else "black"
    return {
        "color": fg,
        "path_effects": [pe.withStroke(linewidth=2.3, foreground=stroke)],
    }


def add_readable_text(ax: plt.Axes, x, y, text: str, *, bg_color=None, fontsize: int = 8,
                      ha: str = "center", va: str = "center", fontweight: str = "semibold",
                      bbox: bool = False, **kwargs) -> None:
    style = readable_text_style(bg_color)
    box = None
    if bbox:
        # A small nearly-white box makes numbers unambiguous in dense heatmaps.
        box = dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.78)
        style = {"color": "black", "path_effects": []}
    ax.text(
        x, y, text,
        ha=ha, va=va, fontsize=fontsize, fontweight=fontweight,
        bbox=box, clip_on=False, **style, **kwargs
    )


def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}", rotation: int = 0, fontsize: int = 8) -> None:
    ymax = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        if not np.isfinite(h):
            continue
        
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + ymax * 0.012,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
            clip_on=False,
            color="black",
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.82),
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
        )


def read_inputs(final_csv: Path, summary_csv: Path, best_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(final_csv)
    summary = pd.read_csv(summary_csv)
    best = pd.read_csv(best_csv)
    final["dataset"] = pd.Categorical(final["dataset"], DATASET_ORDER, ordered=True)
    final["model"] = pd.Categorical(final["model"], ordered_models(final), ordered=True)
    final = final.sort_values(["dataset", "model"]).reset_index(drop=True)
    summary["Dataset"] = pd.Categorical(summary["Dataset"], DATASET_ORDER, ordered=True)
    summary = summary.sort_values("Dataset").reset_index(drop=True)
    return final, summary, best


def grouped_summary_bar(summary: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]], metric_cols: List[str], title: str, ylabel: str, base: str, ylim_pad: float, main: bool, note: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.1), layout="constrained")
    x = np.arange(len(summary))
    width = 0.24
    for i, acc in enumerate(ACCELS):
        col = metric_cols[i]
        bars = ax.bar(
            x + (i - 1) * width,
            summary[col].astype(float),
            width=width,
            color=ACCEL_COLORS[acc],
            edgecolor="black",
            linewidth=0.7,
            hatch=ACCEL_HATCHES[acc],
            label=acc,
        )
        add_bar_labels(ax, bars, "{:.2f}", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Dataset"].astype(str))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    vals = summary[metric_cols].to_numpy(dtype=float)
    ymin = 0 if vals.min() > 10 and "Acc" not in ylabel else max(0, vals.min() - ylim_pad)
    ymax = vals.max() + ylim_pad
    ax.set_ylim(ymin, ymax)
    ax.legend(handles=accel_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True)
    save_fig(fig, out_dir, base, main, index_rows, note)


def plot_summary_speedups(summary: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.3), layout="constrained")
    x = np.arange(len(summary))
    width = 0.26
    series = [
        ("GPU/CPU speedup", "GPU/CPU Speedup", "#2563EB", "//"),
        ("NPU/CPU speedup", "NPU/CPU Speedup", "#059669", "\\\\"),
        ("NPU/GPU latency ratio", "NPU/GPU Latency Ratio", "#D97706", "xx"),
    ]
    for i, (label, col, color, hatch) in enumerate(series):
        bars = ax.bar(x + (i - 1) * width, summary[col].astype(float), width, label=label, color=color, edgecolor="black", linewidth=0.7, hatch=hatch)
        add_bar_labels(ax, bars, "{:.2f}", fontsize=8)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(len(summary) - 0.55, 1.05, "1.0 baseline", fontsize=8, ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Dataset"].astype(str))
    ax.set_ylabel("Ratio (model-level mean)")
    ax.set_title("Batch-1 Speedup and Relative-Latency Ratios by Dataset", pad=10)
    ax.set_ylim(0, max(summary["NPU/CPU Speedup"].max(), summary["GPU/CPU Speedup"].max()) * 1.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    save_fig(fig, out_dir, "batch1_summary_speedup_and_latency_ratios", True, index_rows, "Main summary ratio figure; values are arithmetic means of model-level ratios.")


def plot_fastest_counts(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    counts = final.groupby(["dataset", "fastest_accelerator"], observed=False).size().unstack(fill_value=0).reindex(index=DATASET_ORDER, columns=ACCELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))
    for acc in ACCELS:
        vals = counts[acc].values
        bars = ax.bar(x, vals, bottom=bottom, color=ACCEL_COLORS[acc], edgecolor="black", linewidth=0.7, hatch=ACCEL_HATCHES[acc], label=acc)
        for b, v, bot in zip(bars, vals, bottom):
            if v > 0:
                add_readable_text(ax, b.get_x() + b.get_width()/2, bot + v/2, f"{int(v)}", bg_color=to_rgb(ACCEL_COLORS[acc]), fontsize=10, fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(str))
    ax.set_ylim(0, 12.8)
    ax.set_ylabel("Number of models (out of 12)")
    ax.set_title("Fastest Accelerator Counts Under Batch-1 Inference", pad=10)
    ax.legend(handles=accel_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=True)
    save_fig(fig, out_dir, "batch1_fastest_accelerator_counts", True, index_rows, "Shows how often each accelerator is fastest within each dataset.")


def heatmap_matrix(final: pd.DataFrame, col: str) -> pd.DataFrame:
    models = ordered_models(final)
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    mat = final.pivot(index="model", columns="dataset", values=col).reindex(index=models, columns=DATASET_ORDER)
    mat.index = labels
    return mat.astype(float)


def plot_heatmap(final: pd.DataFrame, col: str, title: str, cbar_label: str, base: str, main: bool, index_rows: List[Dict[str, str]], out_dir: Path, cmap: str = "viridis", center_zero: bool = False, fmt: str = ".2f", note: str = "") -> None:
    mat = heatmap_matrix(final, col)
    h = max(6.4, 0.48 * len(mat.index) + 2.1)
    fig, ax = plt.subplots(figsize=(8.6, h), layout="constrained")
    arr = mat.to_numpy(dtype=float)
    if center_zero:
        vmax = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)), 0.01)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Model")
    ax.grid(False)

    # Readable annotation rule:
    # - compute the actual cell background color from the colormap/norm
    # - choose black text on light cells and white text on dark cells
    # - add an opposite-color outline stroke
    # This fixes the hidden-number problem in dark blue and pale yellow cells.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not np.isfinite(val):
                continue
            bg = im.cmap(im.norm(val))
            text = format(val, fmt)
            add_readable_text(ax, j, i, text, bg_color=bg, fontsize=8, fontweight="bold", bbox=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.035)
    cbar.set_label(cbar_label)
    save_fig(fig, out_dir, base, main, index_rows, note)


def plot_combined_accuracy_diff(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    mats = [heatmap_matrix(final, "cpu_gpu_accuracy_diff_pp"), heatmap_matrix(final, "npu_gpu_accuracy_diff_pp")]
    titles = ["CPU - GPU accuracy difference", "NPU - GPU accuracy difference"]
    vmax = max(max(abs(m.to_numpy(dtype=float)).max() for m in mats), 0.01)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 7.5), layout="constrained", sharey=True)
    cmap = "coolwarm"
    im = None
    for ax, mat, title in zip(axes, mats, titles):
        arr = mat.to_numpy(dtype=float)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(title, pad=9)
        ax.set_xlabel("Dataset")
        ax.grid(False)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                bg = im.cmap(norm(val))
                add_readable_text(ax, j, i, f"{val:+.2f}", bg_color=bg, fontsize=7, fontweight="bold", bbox=False)
    axes[0].set_ylabel("Model")
    cbar = fig.colorbar(im, ax=axes, shrink=0.80, pad=0.025)
    cbar.set_label("Accuracy difference (percentage points)")
    fig.suptitle("Batch-1 Accuracy Differences Relative to GPU", y=1.02)
    save_fig(fig, out_dir, "batch1_combined_accuracy_difference_heatmaps", True, index_rows, "Main accuracy preservation heatmap. Differences near zero indicate stable deployment behavior.")


def plot_latency_by_model(final: pd.DataFrame, dataset: str, out_dir: Path, index_rows: List[Dict[str, str]], p95: bool = False) -> None:
    sub = final[final["dataset"].astype(str) == dataset].copy()
    models = ordered_models(sub)
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13.4, 6.2), layout="constrained")
    suffix = "p95_latency_ms" if p95 else "mean_latency_ms"
    for i, acc in enumerate(ACCELS):
        vals = sub.set_index("model").reindex(models)[f"{acc.lower()}_{suffix}"].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 1) * width, vals, width, color=ACCEL_COLORS[acc], edgecolor="black", linewidth=0.6, hatch=ACCEL_HATCHES[acc], label=acc)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel(("P95" if p95 else "Mean") + " latency (ms/image)")
    ax.set_title(f"{dataset}: Batch-1 {'P95' if p95 else 'Mean'} Latency by Model and Accelerator", pad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    ax.legend(handles=accel_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, frameon=True)
    base = f"batch1_{'p95' if p95 else 'mean'}_latency_by_model_{dataset.lower()}"
    save_fig(fig, out_dir, base, False, index_rows, f"Supplementary detailed {dataset} {'P95' if p95 else 'mean'} latency figure.")


def plot_speedup_by_model(final: pd.DataFrame, dataset: str, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    sub = final[final["dataset"].astype(str) == dataset].copy()
    models = ordered_models(sub)
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=(13.4, 6.1), layout="constrained")
    series = [("GPU/CPU", "gpu_cpu_speedup", "#2563EB", "//"), ("NPU/CPU", "npu_cpu_speedup", "#059669", "\\\\")]
    for i, (label, col, color, hatch) in enumerate(series):
        vals = sub.set_index("model").reindex(models)[col].to_numpy(dtype=float)
        ax.bar(x + (i - 0.5) * width, vals, width, color=color, edgecolor="black", linewidth=0.6, hatch=hatch, label=label)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Speedup over CPU (higher is better)")
    ax.set_title(f"{dataset}: Batch-1 Accelerator Speedup over CPU", pad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=True)
    save_fig(fig, out_dir, f"batch1_speedup_over_cpu_by_model_{dataset.lower()}", False, index_rows, f"Supplementary detailed {dataset} model-level CPU-normalized speedup figure.")


def plot_npu_gpu_ratio_by_model(final: pd.DataFrame, dataset: str, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    sub = final[final["dataset"].astype(str) == dataset].copy()
    models = ordered_models(sub)
    labels = [MODEL_DISPLAY.get(m, m) for m in models]
    vals = sub.set_index("model").reindex(models)["npu_gpu_latency_ratio"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13.2, 5.8), layout="constrained")
    colors = ["#059669" if v < 1 else "#D97706" for v in vals]
    bars = ax.bar(np.arange(len(models)), vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.1)
    ax.text(len(models)-0.2, 1.03, "NPU = GPU", ha="right", va="bottom", fontsize=8, bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8))
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("NPU/GPU latency ratio (lower is better)")
    ax.set_title(f"{dataset}: NPU Relative Latency Compared with GPU", pad=10)
    ax.set_ylim(0, max(1.1, vals.max() * 1.18))
    legend = [Patch(facecolor="#059669", edgecolor="black", label="NPU faster than GPU"), Patch(facecolor="#D97706", edgecolor="black", label="NPU slower than GPU")]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2, frameon=True)
    save_fig(fig, out_dir, f"batch1_npu_gpu_relative_latency_by_model_{dataset.lower()}", False, index_rows, f"Supplementary detailed {dataset} NPU/GPU relative latency figure.")


def plot_accuracy_latency_tradeoff(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]], dataset: str | None = None) -> None:
    sub = final.copy() if dataset is None else final[final["dataset"].astype(str) == dataset].copy()
    fig, ax = plt.subplots(figsize=(9.4, 6.4), layout="constrained")
    markers = {"CPU": "o", "GPU": "s", "NPU": "^"}
    for acc in ACCELS:
        x = sub[f"{acc.lower()}_mean_latency_ms"].to_numpy(dtype=float)
        y = sub[f"{acc.lower()}_accuracy_percent"].to_numpy(dtype=float)
        ax.scatter(x, y, s=62, marker=markers[acc], color=ACCEL_COLORS[acc], edgecolor="black", linewidth=0.5, alpha=0.88, label=acc)
    ax.set_xlabel("Mean latency (ms/image, lower is better)")
    ax.set_ylabel("Accuracy (%)")
    name = "All datasets" if dataset is None else dataset
    ax.set_title(f"Batch-1 Accuracy-Latency Tradeoff ({name})", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    if dataset is None:
        base = "batch1_accuracy_latency_tradeoff_all_datasets"
        main = True
        note = "Main aggregate scatter showing deployment tradeoff across all models and datasets."
    else:
        base = f"batch1_accuracy_latency_tradeoff_{dataset.lower()}"
        main = False
        note = f"Supplementary {dataset} tradeoff scatter."
    save_fig(fig, out_dir, base, main, index_rows, note)


def compute_balanced_scores(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, sub in final.groupby("dataset", observed=False):
        sub = sub.copy()
        acc_values = []
        lat_values = []
        for acc in ACCELS:
            for _, r in sub.iterrows():
                acc_values.append(r[f"{acc.lower()}_accuracy_percent"])
                lat_values.append(r[f"{acc.lower()}_mean_latency_ms"])
        acc_min, acc_max = min(acc_values), max(acc_values)
        lat_min, lat_max = min(lat_values), max(lat_values)
        for _, r in sub.iterrows():
            for acc in ACCELS:
                raw_acc = r[f"{acc.lower()}_accuracy_percent"]
                raw_lat = r[f"{acc.lower()}_mean_latency_ms"]
                norm_acc = 1.0 if acc_max == acc_min else (raw_acc - acc_min) / (acc_max - acc_min)
                norm_lat = 1.0 if lat_max == lat_min else (lat_max - raw_lat) / (lat_max - lat_min)
                score = 0.6 * norm_acc + 0.4 * norm_lat
                rows.append({"dataset": str(ds), "model": str(r["model"]), "model_display": r["model_display"], "accelerator": acc, "accuracy_percent": raw_acc, "latency_ms": raw_lat, "balanced_score": score})
    return pd.DataFrame(rows)


def plot_balanced_score(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]], dataset: str | None = None) -> None:
    scores = compute_balanced_scores(final)
    if dataset is not None:
        scores = scores[scores["dataset"] == dataset]
    pivot = scores.groupby("accelerator")["balanced_score"].mean().reindex(ACCELS)
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    bars = ax.bar(pivot.index, pivot.values, color=[ACCEL_COLORS[a] for a in pivot.index], edgecolor="black", linewidth=0.8)
    add_bar_labels(ax, bars, "{:.3f}", fontsize=9)
    ax.set_ylim(0, max(1.0, pivot.max() * 1.18))
    ax.set_ylabel("Balanced score (0.6 accuracy + 0.4 latency)")
    title = "Average Balanced Deployment Score" if dataset is None else f"{dataset}: Balanced Deployment Score"
    ax.set_title(title, pad=10)
    base = "batch1_balanced_score_all_datasets" if dataset is None else f"batch1_balanced_score_{dataset.lower()}"
    main = dataset is None
    save_fig(fig, out_dir, base, main, index_rows, "Balanced score is optional; useful if the paper discusses deployment decisions.")


def plot_model_robustness(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    rows = []
    for model, sub in final.groupby("model", observed=False):
        if len(sub) == 0:
            continue
        display = sub["model_display"].iloc[0]
        mean_acc = np.mean([sub[f"{a.lower()}_accuracy_percent"].mean() for a in ACCELS])
        mean_lat = np.mean([sub[f"{a.lower()}_mean_latency_ms"].mean() for a in ACCELS])
        # rank accuracy descending and latency ascending across models
        rows.append({"model": str(model), "model_display": display, "mean_accuracy": mean_acc, "mean_latency": mean_lat})
    ranking = pd.DataFrame(rows)
    ranking["accuracy_rank"] = ranking["mean_accuracy"].rank(ascending=False, method="average")
    ranking["latency_rank"] = ranking["mean_latency"].rank(ascending=True, method="average")
    ranking["average_rank"] = (ranking["accuracy_rank"] + ranking["latency_rank"]) / 2
    ranking = ranking.sort_values("average_rank")
    ranking.to_csv(out_dir / "data_exports" / "batch1_model_robustness_ranking.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.0, 6.2), layout="constrained")
    y = np.arange(len(ranking))
    bars = ax.barh(y, ranking["average_rank"], color="#4B5563", edgecolor="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ranking["model_display"])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank across accuracy and latency (lower is better)")
    ax.set_title("Model Robustness Ranking Across Batch-1 Deployments", pad=10)
    for b, val in zip(bars, ranking["average_rank"]):
        ax.text(val + 0.08, b.get_y()+b.get_height()/2, f"{val:.2f}", va="center", fontsize=8, color="black", bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82))
    ax.set_xlim(0, ranking["average_rank"].max() * 1.18)
    save_fig(fig, out_dir, "batch1_model_robustness_ranking", False, index_rows, "Supplementary model robustness ranking.")


def plot_family_latency(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    fam = final.groupby("model_family", observed=False)[["cpu_mean_latency_ms", "gpu_mean_latency_ms", "npu_mean_latency_ms"]].mean().sort_values("npu_mean_latency_ms")
    labels = fam.index.astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10.8, 5.8), layout="constrained")
    for i, acc in enumerate(ACCELS):
        ax.bar(x + (i - 1) * width, fam[f"{acc.lower()}_mean_latency_ms"], width, color=ACCEL_COLORS[acc], edgecolor="black", linewidth=0.7, hatch=ACCEL_HATCHES[acc], label=acc)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Mean latency (ms/image)")
    ax.set_title("Average Batch-1 Latency by Model Family", pad=10)
    ax.legend(handles=accel_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True)
    save_fig(fig, out_dir, "batch1_family_average_latency", False, index_rows, "Supplementary model-family latency analysis.")


def plot_pareto(final: pd.DataFrame, out_dir: Path, index_rows: List[Dict[str, str]], dataset: str) -> None:
    sub = final[final["dataset"].astype(str) == dataset].copy()
    fig, ax = plt.subplots(figsize=(9.2, 6.5), layout="constrained")
    markers = {"CPU": "o", "GPU": "s", "NPU": "^"}
    for acc in ACCELS:
        x = sub[f"{acc.lower()}_mean_latency_ms"].to_numpy(dtype=float)
        y = sub[f"{acc.lower()}_accuracy_percent"].to_numpy(dtype=float)
        ax.scatter(x, y, marker=markers[acc], s=75, color=ACCEL_COLORS[acc], edgecolor="black", linewidth=0.5, label=acc, alpha=0.88)
    # Annotate only top accuracy and lowest latency points to avoid clutter.
    points = []
    for acc in ACCELS:
        lat_col = f"{acc.lower()}_mean_latency_ms"
        acc_col = f"{acc.lower()}_accuracy_percent"
        points.append(sub.loc[sub[lat_col].idxmin(), ["model_display", lat_col, acc_col]].tolist() + [acc])
        points.append(sub.loc[sub[acc_col].idxmax(), ["model_display", lat_col, acc_col]].tolist() + [acc])
    seen = set()
    for name, lat, accv, a in points:
        key = (name, a)
        if key in seen:
            continue
        seen.add(key)
        ax.annotate(str(name), (lat, accv), xytext=(5, 4), textcoords="offset points", fontsize=7, bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.78))
    ax.set_xlabel("Mean latency (ms/image, lower is better)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{dataset}: Accuracy-Latency Deployment Frontier", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    save_fig(fig, out_dir, f"batch1_pareto_frontier_{dataset.lower()}", False, index_rows, f"Supplementary {dataset} Pareto-style tradeoff figure.")


def write_support_files(out_dir: Path, index_rows: List[Dict[str, str]]) -> None:
    idx = pd.DataFrame(index_rows)
    idx.to_csv(out_dir / "batch1_publication_figure_index.csv", index=False)
    with open(out_dir / "batch1_publication_figure_index.md", "w", encoding="utf-8") as f:
        f.write("# Batch-1 Clean Figure Index\n\n")
        for _, r in idx.iterrows():
            f.write(f"- **{r['file']}** | Main paper: {r['recommended_for_main_paper']} | {r['note']}\n")
    with open(out_dir / "batch1_publication_latex_snippets.tex", "w", encoding="utf-8") as f:
        f.write("% Publication-safe batch-1 LaTeX snippets. Copy only the selected main-paper figures.\n\n")
        for _, r in idx[idx["recommended_for_main_paper"] == "YES"].iterrows():
            base = Path(r["file"]).stem
            f.write(textwrap.dedent(f"""
            \\begin{{figure}}[!t]
                \\centering
                \\includegraphics[width=\\linewidth]{{batch1_figures_publication/{r['file']}}}
                \\caption{{TODO: replace with final caption.}}
                \\label{{fig:{base}}}
            \\end{{figure}}

            """))
    with open(out_dir / "batch1_publication_caption_suggestions.md", "w", encoding="utf-8") as f:
        f.write("# Caption suggestions for selected clean batch-1 figures\n\n")
        f.write("Use only the strongest 6-8 figures in the paper. The rest are supplementary/supporting figures.\n\n")
        for _, r in idx[idx["recommended_for_main_paper"] == "YES"].iterrows():
            f.write(f"## {r['file']}\n{r['note']}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-safe IEEE-style batch-1 CPU/GPU/NPU figures with readable annotations.")
    parser.add_argument("--final-csv", default="results/final_cpu_gpu_npu_batch1_comparison_3datasets.csv")
    parser.add_argument("--summary-csv", default="results/table2_dataset_summary_batch1.csv")
    parser.add_argument("--best-csv", default="results/table3_best_fastest_batch1.csv")
    parser.add_argument("--output-dir", default="results/batch1_figures_publication")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete the output folder before generating figures.")
    args = parser.parse_args()

    setup_matplotlib()
    final_csv = Path(args.final_csv)
    summary_csv = Path(args.summary_csv)
    best_csv = Path(args.best_csv)
    out_dir = Path(args.output_dir)

    if not final_csv.exists():
        raise FileNotFoundError(f"Missing final comparison CSV: {final_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")
    if not best_csv.exists():
        raise FileNotFoundError(f"Missing best/fastest CSV: {best_csv}")

    if args.no_clean:
        (out_dir / "main_paper").mkdir(parents=True, exist_ok=True)
        (out_dir / "supplementary").mkdir(parents=True, exist_ok=True)
        (out_dir / "data_exports").mkdir(parents=True, exist_ok=True)
    else:
        clean_output_dir(out_dir)

    final, summary, best = read_inputs(final_csv, summary_csv, best_csv)
    final.to_csv(out_dir / "data_exports" / "batch1_plot_input_final_comparison_copy.csv", index=False)
    summary.to_csv(out_dir / "data_exports" / "batch1_plot_input_table2_summary_copy.csv", index=False)
    best.to_csv(out_dir / "data_exports" / "batch1_plot_input_table3_best_fastest_copy.csv", index=False)

    index_rows: List[Dict[str, str]] = []

    # Main summary figures.
    grouped_summary_bar(summary, out_dir, index_rows, ["CPU Acc. (%)", "GPU Acc. (%)", "NPU Acc. (%)"], "Batch-1 Mean Accuracy by Dataset", "Accuracy (%)", "batch1_summary_accuracy_by_dataset", 0.25, True, "Main accuracy summary. Y-axis is zoomed only enough to show small deployment differences.")
    grouped_summary_bar(summary, out_dir, index_rows, ["CPU Lat. (ms/img)", "GPU Lat. (ms/img)", "NPU Lat. (ms/img)"], "Batch-1 Mean Latency by Dataset", "Mean latency (ms/image)", "batch1_summary_latency_by_dataset", 2.0, True, "Main latency summary showing NPU average advantage under batch-1 inference.")
    plot_summary_speedups(summary, out_dir, index_rows)
    plot_fastest_counts(final, out_dir, index_rows)
    plot_heatmap(final, "npu_gpu_latency_ratio", "NPU/GPU Batch-1 Latency Ratio by Model and Dataset", "NPU/GPU latency ratio (lower is better)", "batch1_heatmap_npu_gpu_latency_ratio", True, index_rows, out_dir, cmap="YlGnBu_r", fmt=".2f", note="Main heatmap showing where NPU is faster/slower than GPU. Values below 1 favor NPU.")
    plot_heatmap(final, "npu_cpu_speedup", "NPU Speedup over CPU by Model and Dataset", "NPU/CPU speedup (higher is better)", "batch1_heatmap_npu_cpu_speedup", True, index_rows, out_dir, cmap="YlGnBu", fmt=".2f", note="Main heatmap showing CPU-normalized NPU speedup.")
    plot_heatmap(final, "gpu_cpu_speedup", "GPU Speedup over CPU by Model and Dataset", "GPU/CPU speedup (higher is better)", "batch1_heatmap_gpu_cpu_speedup", True, index_rows, out_dir, cmap="Blues", fmt=".2f", note="Main or optional heatmap showing CPU-normalized GPU speedup.")
    plot_combined_accuracy_diff(final, out_dir, index_rows)
    plot_accuracy_latency_tradeoff(final, out_dir, index_rows, dataset=None)
    plot_balanced_score(final, out_dir, index_rows, dataset=None)

    # Supplementary detailed figures.
    plot_heatmap(final, "cpu_mean_latency_ms", "CPU Batch-1 Mean Latency", "ms/image", "batch1_heatmap_cpu_mean_latency", False, index_rows, out_dir, cmap="Oranges", fmt=".1f", note="Supplementary CPU latency heatmap.")
    plot_heatmap(final, "gpu_mean_latency_ms", "GPU Batch-1 Mean Latency", "ms/image", "batch1_heatmap_gpu_mean_latency", False, index_rows, out_dir, cmap="Blues", fmt=".1f", note="Supplementary GPU latency heatmap.")
    plot_heatmap(final, "npu_mean_latency_ms", "NPU Batch-1 Mean Latency", "ms/image", "batch1_heatmap_npu_mean_latency", False, index_rows, out_dir, cmap="Greens", fmt=".1f", note="Supplementary NPU latency heatmap.")
    plot_heatmap(final, "cpu_gpu_accuracy_diff_pp", "CPU - GPU Accuracy Difference", "percentage points", "batch1_heatmap_cpu_gpu_accuracy_diff_pp", False, index_rows, out_dir, cmap="coolwarm", center_zero=True, fmt="+.2f", note="Supplementary CPU/GPU accuracy difference heatmap.")
    plot_heatmap(final, "npu_gpu_accuracy_diff_pp", "NPU - GPU Accuracy Difference", "percentage points", "batch1_heatmap_npu_gpu_accuracy_diff_pp", False, index_rows, out_dir, cmap="coolwarm", center_zero=True, fmt="+.2f", note="Supplementary NPU/GPU accuracy difference heatmap.")

    for ds in DATASET_ORDER:
        plot_latency_by_model(final, ds, out_dir, index_rows, p95=False)
        plot_latency_by_model(final, ds, out_dir, index_rows, p95=True)
        plot_speedup_by_model(final, ds, out_dir, index_rows)
        plot_npu_gpu_ratio_by_model(final, ds, out_dir, index_rows)
        

    plot_model_robustness(final, out_dir, index_rows)
    if "model_family" in final.columns:
        plot_family_latency(final, out_dir, index_rows)

    scores = compute_balanced_scores(final)
    scores.to_csv(out_dir / "data_exports" / "batch1_balanced_scores_long.csv", index=False)
    write_support_files(out_dir, index_rows)

    print("PUBLICATION-SAFE BATCH-1 IEEE FIGURE GENERATOR")
    print("=" * 72)
    print(f"Input final CSV:   {final_csv}")
    print(f"Input summary CSV: {summary_csv}")
    print(f"Input best CSV:    {best_csv}")
    print(f"Output directory:  {out_dir}")
    print(f"Main-paper figures: {sum(r['recommended_for_main_paper']=='YES' for r in index_rows)}")
    print(f"Supplementary figures: {sum(r['recommended_for_main_paper']=='NO' for r in index_rows)}")
    print("Annotation style: adaptive high-contrast text with outline strokes")
    print("\nMain-paper PNG files:")
    for r in index_rows:
        if r["recommended_for_main_paper"] == "YES":
            print("  " + str(out_dir / r["file"]))
    print("\nIndex:", out_dir / "batch1_publication_figure_index.md")
    print("Captions:", out_dir / "batch1_publication_caption_suggestions.md")
    print("LaTeX:", out_dir / "batch1_publication_latex_snippets.tex")
    print("\n[OK] Publication-safe batch-1 figure generation completed successfully.")


if __name__ == "__main__":
    main()


