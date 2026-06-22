# CPU, GPU, and NPU Deployment for Remote-Sensing Image Classification on an Edge-Class Laptop

This repository contains the scripts, result tables, and figures supporting the paper:

**CPU, GPU, and NPU Deployment for Remote-Sensing Image Classification on an Edge-Class Laptop: A Three-Dataset, Twelve-Model Study**

## Overview

This project evaluates CPU, GPU, and NPU deployment performance for remote-sensing image classification on an edge-class laptop. The study compares twelve lightweight and modern vision models across three remote-sensing datasets using a single-image inference setting.

The goal is to study practical deployment behavior rather than only theoretical model complexity. The reported latency values should be interpreted as single-image inference-path latency, including framework and runtime execution overhead, not isolated accelerator-kernel time.

## Datasets

The experiments use three remote-sensing image classification datasets:

* **EuroSAT**: 10 land-use and land-cover classes
* **AID**: 30 aerial scene classes
* **RESISC45**: 45 remote-sensing scene classes

Each dataset uses a deterministic 80/10/10 train/validation/test split with seed 42. Images are resized to 224 x 224 and normalized using ImageNet mean and standard deviation. Training uses random horizontal flipping, while validation and testing use deterministic preprocessing.

## Models

The benchmark includes twelve models:

* ResNet18
* MobileNetV2
* MobileNetV3-Small
* MobileNetV3-Large
* EfficientNet-B0
* EfficientNetV2-S
* ShuffleNetV2
* ConvNeXt-Tiny
* RepViT-M0.9
* StarNet-S4
* SHViT-S4
* MambaOut-Kobe

The models are trained using ImageNet-pretrained weights where available, with the final classification layer adapted to the number of classes in each dataset.

## Hardware

Experiments were conducted on an edge-class laptop with:

* Intel Core Ultra 9 275HX CPU
* NVIDIA GeForce RTX 5070 Ti Laptop GPU
* Integrated Intel NPU
* Windows 11

## Software Environments

Two environments were used:

* `ai_env`: PyTorch CPU/GPU training, CPU/GPU inference, ONNX export, result merging, and plotting
* `npu_env`: ONNX Runtime with OpenVINO Execution Provider for NPU inference

The repository includes:

```text
requirements_ai_env.txt
requirements_npu_env.txt
```

These files were exported from the actual environments used during the project.

## Deployment Paths

Three deployment paths are evaluated:

1. **CPU**: PyTorch CPU inference
2. **GPU**: PyTorch CUDA inference
3. **NPU**: ONNX Runtime with OpenVINO Execution Provider using `device_type=NPU`

The NPU path uses ONNX-exported models. ONNX CPU sanity checks were performed before NPU benchmarking to verify that exported models preserved the expected behavior.

## Benchmark Protocol

The official inference benchmark uses batch size 1 to represent single-image edge responsiveness.

For each model-dataset-accelerator combination, the benchmark records:

* Test accuracy
* Macro precision
* Macro recall
* Macro F1-score
* Mean latency
* Median latency
* P95 latency
* FPS
* Accelerator speedup or latency ratio metrics

The final benchmark contains:

```text
3 datasets x 12 models x 3 accelerators = 108 deployment results
```

## Main Findings

The final batch-size-1 benchmark shows:

* Accuracy is preserved across CPU, GPU, and NPU deployment paths.
* The maximum CPU-GPU and NPU-GPU accuracy differences are within 0.10 percentage points.
* The NPU achieves the lowest latency in 29 out of 36 model-dataset cases.
* The GPU is fastest in the remaining 7 cases.
* The CPU is not the fastest deployment path for any model-dataset pair.
* Accelerator performance is model-dependent and affected by runtime stack, operator composition, and batch size.

## Repository Structure

```text
scripts/
    Training, ONNX export, sanity checking, CPU/GPU/NPU benchmarking, merging, and plotting scripts.

results/
    Final CSV and Markdown result tables used for the paper.

figures/
    Final publication figures generated from the result CSV files.

paper/
    Optional paper PDF, if included.
```

## Main Scripts

The main pipeline scripts are:

```text
benchmark_train_gpu_v4_convergence_resisc45.py
export_converged_3datasets_to_onnx.py
onnx_cpu_sanity_converged_3datasets.py
benchmark_gpu_batch1_3datasets.py
benchmark_cpu_batch1_3datasets.py
benchmark_npu_batch1_3datasets.py
merge_batch1_cpu_gpu_npu_results.py
plot_batch1_cpu_gpu_npu_results_publication.py
```

The intended workflow is:

```text
train models
export trained checkpoints to ONNX
run ONNX CPU sanity checks
benchmark CPU, GPU, and NPU inference
merge final results
generate publication figures
```

## Final Result Files

The final result files are:

```text
final_cpu_gpu_npu_batch1_comparison_3datasets.csv
cpu_benchmark_results_batch1_3datasets.csv
gpu_benchmark_results_batch1_3datasets.csv
npu_benchmark_results_batch1_3datasets.csv
table2_dataset_summary_batch1.csv
table2_dataset_summary_batch1.md
table3_best_fastest_batch1.csv
table3_best_fastest_batch1.md
batch1_merge_validation_report.txt
```

## Notes on Large Files

Large datasets, trained `.pth` checkpoints, ONNX model files, and virtual environment folders are not included in this repository due to size constraints.

The repository is intended to provide the scripts, final result tables, and figures needed to support the paper and reproduce the analysis workflow.
