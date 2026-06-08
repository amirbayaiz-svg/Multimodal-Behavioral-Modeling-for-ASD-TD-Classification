# HMB-TSC: Hybrid Multibranch Time-Series Classification for ASD/TD Screening

This repository contains a fully executable reconstruction of the HMB-TSC experimental workflow for binary ASD/TD classification from 2D skeletal time-series data. The project implements the complete machine-learning pipeline: dataset discovery, subject-level data preparation, feature transformation, exploratory analysis, model training, holdout evaluation, cross-validation, ablation analysis, nested validation, statistical testing, robustness checks, and publication-ready visualizations.

The main goal of the repository is to provide a reproducible implementation of the proposed HMB-TSC model and all supporting experiments using the available `Dataset-2` data directory. All performance values reported in this README are taken only from the released notebook outputs stored in `results/tables/`.

## Reproducible Result Scope

The repository contains only the 100 original 2D trajectories and regenerates synthetic variants inside each training split. Therefore, the reproducible values for this archive are the generated values listed in `results/tables/` and repeated below. No additional performance values are used in the README tables.

For this released run, HMB-TSC reached holdout `Accuracy = 0.850`, `AUROC = 0.948`, and `F1 = 0.880`; 5-fold and nested 5x3 validation both reached `Accuracy = 0.850 ± 0.035`, `AUROC = 0.940 ± 0.029`, and `F1 = 0.845 ± 0.036`.

## Repository Structure

```text
.
├── Dataset-2/
├── HMB_TSC_main-code.ipynb
├── IEEE_color_palette_before_after_dark2.csv
├── IEEE_color_palette_before_after_dark2.md
├── README.md
├── requirements_hmb_tsc.txt
└── results/
    ├── figures/
    ├── histories/
    ├── predictions/
    └── tables/
```

## Main Notebook

The complete executable workflow is implemented in:

```text
HMB_TSC_main-code.ipynb
```

The notebook is self-contained and includes the experimental code for:

- dataset loading and file auditing;
- original/augmented file handling;
- trajectory cleaning and outlier filtering;
- demographic and exploratory data analysis;
- mutual information and correlation analysis;
- Euclidean magnitude transformation;
- logarithmic feature transformation;
- standardization using training statistics only;
- interpolation to fixed-length sequences;
- subject-level train/test splitting;
- baseline model training;
- full HMB-TSC training;
- 5-fold cross-validation;
- ablation studies;
- nested 5x3 cross-validation;
- statistical comparisons;
- Kinect coordinate-noise sensitivity analysis;
- leave-10%-subjects validation;
- final result tables and publication-ready figures.

## Dataset Summary

The experiment uses the original 2D skeletal trajectories stored under `Dataset-2/`.
The GitHub archive contains only the 100 original 2D trajectory files that are required to rerun the released workflow. Stored augmentation files and raw video files are not included in the archive because they are not used as validation/test samples.

| Item | Value |
|---|---:|
| Original 2D files used | 100 |
| Stored augmentation files included in archive | 0 |
| Train-only synthetic variants generated per training trajectory | 7 |
| Rows before filtering | 3,183 |
| Rows after filtering | 3,128 |
| Removed rows | 55 |
| Removed rows, % | 1.73 |
| Trajectory tensor before feature reduction | `(100, 256, 25, 2)` |
| Model input after log-magnitude transformation | `(100, 256, 25)` |
| TD class samples | 50 |
| ASD class samples | 50 |
| Subjects | 100 |

The validation workflow is subject-aware. Synthetic augmentation variants are generated only from the training subjects inside each split and are never used as validation/test samples, which prevents inflated performance from near-duplicate augmented trajectories.

## Preprocessing Pipeline

The notebook reconstructs the preprocessing pipeline used for the experiment:

1. Load original 2D skeletal files from `Dataset-2`.
2. Audit original files and optional stored augmentation paths if present.
3. Parse trajectory-level metadata and labels.
4. Filter invalid or outlier trajectory frames.
5. Convert joint coordinates into Euclidean magnitude-based features.
6. Apply logarithmic transformation to stabilize feature scale.
7. Interpolate each sequence to a fixed temporal length of 256.
8. Standardize features using training-set statistics only.
9. Build subject-level splits with zero subject overlap between train and test partitions.

## Train-Only Augmentation Protocol

The notebook implements seven train-only transformations in `augment_training_coords`. The transformations are applied only after a subject-level split has been created and only to the training subset of the current holdout, 5-fold, nested inner/outer, or leave-10%-subjects split.

| Variant | Implementation in the notebook |
|---|---|
| Magnitude scaling (MS) | Multiplies all coordinates by `alpha`, where `alpha` is sampled from `[0.9, 1.1]` |
| Time warping (TW) | Uses cubic B-spline speed interpolation with `sigma = 0.2` and `3` internal control points |
| Jittering (JT) | Adds Gaussian noise with `mu = 0` and `sigma = 0.01` |
| Horizontal mirroring (HM) | Reflects the horizontal coordinate by multiplying the X-axis by `-1` |
| Random rotation (RR) | Applies a 2D rotation sampled from `[-5, 5]` degrees |
| Scaling + JT | Applies MS followed by JT |
| TW + JT | Applies TW followed by JT |

The original trajectory is retained in the training set together with these seven generated variants, giving eight training instances per original training trajectory. Test and validation subjects remain unaugmented.

## Model Overview

HMB-TSC is a hybrid multibranch neural architecture for skeletal time-series classification. The released implementation uses the full scaled architecture with `6,818,113` trainable parameters.

Input shape:

```text
batch_size x 256 x 25
```

The full HMB-TSC model contains three complementary branches:

- Transformer branch for global temporal dependencies;
- CNN-BiLSTM branch for local spatiotemporal structure and long-range recurrent dynamics;
- CNN-BiGRU branch for complementary sequential dynamics.

The branch outputs are concatenated and passed through a dense classification head that produces a single binary logit.

Full HMB-TSC trainable parameters:

```text
6,818,113
```

## Compared Models

The notebook evaluates the proposed model against three baseline architectures:

- LSTM;
- CNN;
- Transformer;
- HMB-TSC.

The baseline models are kept separate from the HMB-TSC ablation variants.

## Validation Protocols

The project includes several validation settings:

| Protocol | Purpose |
|---|---|
| Holdout validation | Independent subject-level test split |
| 5-fold cross-validation | Outer-fold performance estimation |
| Nested 5x3 cross-validation | Outer validation with inner hyperparameter selection |
| Ablation 5-fold CV | Contribution analysis of HMB-TSC branches |
| Kinect noise sensitivity | Robustness to joint-coordinate measurement error |
| Leave-10%-subjects validation | Generalization to unseen participants |

The leakage audit confirms zero subject overlap between training and testing partitions for the holdout, 5-fold, nested outer folds, and leave-10%-subjects validation.

## Holdout Results

| Model | Parameters | Accuracy | AUROC | Precision | Recall | F1 | MacroF1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HMB-TSC | 6,818,113 | 0.850 | 0.948 | 0.846 | 0.917 | 0.880 | 0.840 | 0.750 |
| LSTM | 8,097 | 0.400 | 0.438 | 0.000 | 0.000 | 0.000 | 0.286 | 1.000 |
| CNN | 9,697 | 0.650 | 0.740 | 1.000 | 0.417 | 0.588 | 0.642 | 1.000 |
| Transformer | 36,193 | 0.650 | 0.906 | 1.000 | 0.417 | 0.588 | 0.642 | 1.000 |

HMB-TSC achieved the strongest holdout performance, with AUROC `0.948` and F1 `0.880`.

## 5-Fold Cross-Validation Results

| Model | Accuracy mean | Accuracy worst | AUROC mean | AUROC worst | F1 mean | F1 worst |
|---|---:|---:|---:|---:|---:|---:|
| HMB-TSC | 0.850 | 0.800 | 0.940 | 0.890 | 0.845 | 0.800 |
| CNN | 0.700 | 0.600 | 0.774 | 0.630 | 0.597 | 0.429 |
| LSTM | 0.550 | 0.450 | 0.646 | 0.490 | 0.383 | 0.000 |
| Transformer | 0.650 | 0.550 | 0.734 | 0.580 | 0.561 | 0.429 |

HMB-TSC achieved the highest mean accuracy, AUROC, and F1 across the 5-fold validation protocol.

## Nested 5x3 Cross-Validation Results

| Model | Accuracy mean | Accuracy worst | AUROC mean | AUROC worst | F1 mean | F1 worst |
|---|---:|---:|---:|---:|---:|---:|
| HMB-TSC | 0.850 | 0.800 | 0.940 | 0.890 | 0.845 | 0.800 |
| CNN | 0.700 | 0.600 | 0.774 | 0.630 | 0.597 | 0.429 |
| LSTM | 0.550 | 0.450 | 0.646 | 0.490 | 0.383 | 0.000 |
| Transformer | 0.650 | 0.550 | 0.734 | 0.580 | 0.561 | 0.429 |

The nested protocol uses the same outer folds while executing the inner model-selection structure. In the released notebook, the recovered hyperparameter grid contains a single selected configuration, so the generated nested values match the 5-fold outer metrics.

## HMB-TSC Ablation Study

The ablation study evaluates the contribution of each branch and branch-only configuration using 5-fold cross-validation.

| Variant | Parameters | Accuracy, % | MacroF1, % | AUROC | p Accuracy vs full | p MacroF1 vs full | p AUROC vs full |
|---|---:|---:|---:|---:|---:|---:|---:|
| HMB-TSC | 6,818,113 | 85.00 | 84.98 | 0.940 | - | - | - |
| without-transformer | 5,509,441 | 81.00 | 80.90 | 0.934 | 0.5122 | 0.5094 | 0.7753 |
| without-cnn-bilstm | 3,968,577 | 89.00 | 88.99 | 0.940 | 0.2420 | 0.2408 | 1.0000 |
| without-cnn-bigru | 4,166,209 | 83.00 | 82.86 | 0.940 | 0.4766 | 0.4689 | 1.0000 |
| transformer-only | 1,316,673 | 77.00 | 76.86 | 0.900 | 0.1202 | 0.1148 | 0.0247 |
| cnn-bilstm-only | 2,857,537 | 85.00 | 84.84 | 0.932 | 1.0000 | 0.9756 | 0.7075 |
| cnn-bigru-only | 2,659,905 | 88.00 | 87.99 | 0.946 | 0.4263 | 0.4239 | 0.7160 |

The table reports paired t-test p-values across the same external folds. This makes the ablation comparison fold-aligned rather than based only on independent mean values.

## Statistical Testing Against Baselines

The repository includes paired statistical comparisons between HMB-TSC and the baseline models.

| Comparison | p Accuracy | p AUROC | McNemar p-value | Bonferroni alpha |
|---|---:|---:|---:|---:|
| HMB-TSC vs LSTM | 0.0020 | 0.0109 | 5.30e-06 | 0.0167 |
| HMB-TSC vs CNN | 0.0231 | 0.0234 | 0.0107 | 0.0167 |
| HMB-TSC vs Transformer | 0.0135 | 0.0023 | 0.0003 | 0.0167 |

These tests are stored in:

```text
results/tables/table12_statistical_tests_5fold.csv
```

## Kinect Noise Sensitivity

To evaluate robustness to Kinect joint-localization error, uniform noise was injected into joint coordinates at two levels: `±13 mm` and `±25 mm`.

| Noise level | AUROC mean | AUROC std | Accuracy mean | F1 mean | Mean AUROC drop | Worst AUROC drop |
|---:|---:|---:|---:|---:|---:|---:|
| 0 mm | 0.940 | 0.029 | 0.850 | 0.845 | - | - |
| 13 mm | 0.940 | 0.029 | 0.810 | 0.835 | 0.000 | 0.000 |
| 25 mm | 0.934 | 0.055 | 0.740 | 0.801 | 0.006 | 0.050 |

The AUROC remained stable under `±13 mm` coordinate noise and decreased only slightly under `±25 mm` noise.

## Leave-10%-Subjects Validation

The leave-10%-subjects protocol evaluates generalization to previously unseen participants. Each fold excludes 10 subjects from training and uses them only for testing.

| Metric | Mean | Standard deviation | Worst fold |
|---|---:|---:|---:|
| Accuracy | 0.880 | 0.140 | 0.600 |
| AUROC | 0.920 | 0.118 | 0.680 |
| F1 | 0.875 | 0.145 | 0.600 |
| Specificity | 0.900 | 0.141 | 0.600 |

The subject-leakage audit confirms zero overlap between training and testing subjects.

## Visualization Standards

All regenerated publication figures use:

- Matplotlib;
- Times New Roman;
- high-resolution export at 350 DPI;
- consistent font sizing;
- annotated confusion matrices;
- readable plot margins for labels and bar annotations;
- ColorBrewer Dark2 color palette.

The applied ColorBrewer Dark2 palette is:

```text
#1b9e77  green
#d95f02  orange
#7570b3  purple
#e7298a  pink
#66a61e  olive green
#e6ab02  mustard
#a6761d  brown
#666666  gray
```


## Reproducibility

Recommended environment:

```text
Python 3.11
PyTorch
Jupyter/IPython kernel
```

The notebook was executed with the following local kernel:

```text
ai-pytorch-mps (Python 3.11)
```

Install the required Python packages:

```bash
pip install -r requirements_hmb_tsc.txt
```

Run the full experiment by opening:

```text
HMB_TSC_main-code.ipynb
```

Then execute all cells from top to bottom. The generated outputs are written to:

```text
results/
```

## Notes on Reproducible Interpretation

- The reported repository-run values in this README are taken from existing generated result files under `results/tables/`.
- The validation procedures are subject-aware and include leakage audits.
- Synthetic augmentation variants are generated only in training folds and are excluded from validation/test.
- The HMB-TSC model in the notebook uses the full scaled architecture with approximately 6.8M trainable parameters.
- The repository includes both baseline comparisons and branch-level ablation variants.
- Robustness and generalization are evaluated using Kinect noise sensitivity and leave-10%-subjects validation.

## Scope

This repository is intended for research reproducibility. It is not a clinical diagnostic product. The results should be interpreted as experimental evidence for the proposed skeletal time-series classification approach on the available dataset.
