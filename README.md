# Multimodal Behavioral Modeling for ASD/TD Classification

This repository contains two research notebooks for multimodal behavioral analysis and ASD/TD classification:

```text
01_dataset_exploration.ipynb
02_model_training_and_multitask_inference.ipynb
```

The project studies behavioral differences between children with autism spectrum disorder (`ASD`) and typically developing children (`TD`) using session-level, temporal and modality-specific features. The workflow includes exploratory data analysis, unsupervised structure analysis, supervised model training, cross-validation, holdout evaluation and reproducible multitask inference.

## Repository Contents

```text
.
├── 01_dataset_exploration.ipynb
├── 02_model_training_and_multitask_inference.ipynb
└── README.md
```

## Notebook 1: Dataset Exploration and Unsupervised Analysis

`01_dataset_exploration.ipynb` performs the exploratory and statistical analysis of the dataset.

The notebook includes:

- dataset audit;
- class distribution analysis;
- missing-value analysis;
- temporal and segment-level feature analysis;
- effect-size ranking;
- feature correlation analysis;
- PCA projection;
- t-SNE projection;
- KMeans clustering;
- hierarchical clustering;
- TD versus ASD single-session temporal comparison.

## Dataset Summary

| Metric | Value |
|---|---:|
| Sessions | 99 |
| Children | 99 |
| Total extracted features | 224 |
| Modeling features | 149 |
| ASD sessions | 52 |
| TD sessions | 47 |
| ASD ratio | 0.525 |
| Sequence dataset shape | `(99, 160, 14)` |

The dataset is close to balanced: 52 ASD sessions and 47 TD sessions. This is suitable for binary classification because the reported performance is less likely to be dominated by class imbalance.

## Temporal Feature Representation

The temporal sequence representation contains 160 normalized time bins and 14 sequence features:

| Feature group | Features |
|---|---|
| Eye and object coordinates | `x`, `y`, `center_x`, `center_y`, `gaze_object_distance` |
| Signal validity | `validity`, `face_ok`, `object_visible`, `object_exploded` |
| Head movement | `head_yaw`, `head_pitch` |
| Facial behavior | `smile_prob`, `au6`, `au12` |

This representation allows the analysis to capture session dynamics rather than relying only on static summary values.

## Main Exploratory Findings

The strongest ASD/TD differences were found in gaze validity, gaze-object distance, gaze dispersion, area-of-interest behavior and gaze-following measures.

| Rank | Feature | Absolute Cohen's d |
|---:|---|---:|
| 1 | `F3_gaze_valid_rate` | 1.786 |
| 2 | `gaze_object_distance_p75` | 1.741 |
| 3 | `gaze_object_distance_mean` | 1.657 |
| 4 | `gaze_x_std` | 1.482 |
| 5 | `F1_gaze_valid_rate` | 1.367 |
| 6 | `F3_Gfollow` | 1.321 |
| 7 | `all_mean_Gfollow` | 1.321 |
| 8 | `gaze_object_distance_p25` | 1.302 |
| 9 | `F3_aoi_hits` | 1.276 |
| 10 | `gaze_y_std` | 1.267 |
| 11 | `F3_aoi_dwell_ms` | 1.248 |
| 12 | `gaze_validity_rate` | 1.169 |
| 13 | `F1_aoi_hits` | 1.155 |
| 14 | `F5_gaze_valid_rate` | 1.121 |
| 15 | `all_mean_aoi_dwell_ms` | 1.106 |

These results indicate that visual attention and object-oriented gaze behavior are the dominant discriminative signals in the dataset.

## Temporal ASD/TD Differences

| Feature | ASD mean | TD mean | ASD - TD |
|---|---:|---:|---:|
| `gaze_object_distance` | 0.387 | 0.255 | +0.132 |
| `smile_prob` | 0.039 | 0.026 | +0.013 |
| `head_yaw` | 0.394 | -1.107 | +1.500 |
| `validity` | 0.652 | 0.853 | -0.200 |
| `object_visible` | 0.979 | 0.967 | +0.012 |

ASD sessions showed a larger gaze-object distance and lower gaze validity. This supports the interpretation that the main class differences are related to visual attention stability and interaction with task-relevant objects.

## Unsupervised Analysis

The first notebook applies PCA, t-SNE, KMeans and hierarchical clustering to evaluate whether ASD and TD samples form separable structures without supervised labels.

| Metric | Value |
|---|---:|
| Adjusted Rand Index | 0.425 |
| Normalized Mutual Information | 0.379 |
| Silhouette | 0.122 |

The unsupervised separation is moderate. The low silhouette score indicates that ASD and TD do not form two cleanly separated natural clusters in the raw feature space. This explains why supervised learning is necessary: the class signal is distributed across multiple behavioral variables rather than concentrated in one simple cluster boundary.

## Notebook 2: Supervised Modeling and Multitask Inference

`02_model_training_and_multitask_inference.ipynb` trains and evaluates classical machine-learning models and deep temporal models.

The notebook includes:

- train/test split;
- cross-validation;
- classical model training;
- deep time-series model training;
- holdout evaluation;
- combined model comparison;
- modality-level interpretation;
- model persistence;
- reproducible multitask inference.

## Models Evaluated

| Model family | Models |
|---|---|
| Classical machine learning | Logistic Regression, SVC RBF, Random Forest |
| Deep time-series learning | BiLSTM_TS, CNN1D_TS, Transformer_TS |

## Modeling Summary

| Metric | Value |
|---|---:|
| Samples | 99 |
| Modeling features | 149 |
| Best classical model | Logistic Regression |
| Best deep model | BiLSTM_TS |
| Best classical CV ROC-AUC | 0.976 |
| Best deep CV ROC-AUC | 0.921 |
| Best classical holdout ROC-AUC | 0.985 |
| Best deep holdout ROC-AUC | 0.980 |

## Classical Model Results

### Cross-Validation

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.899 | 0.898 | 0.889 | 0.923 | 0.906 | 0.978 | 0.976 |
| SVC RBF | 0.889 | 0.887 | 0.873 | 0.923 | 0.897 | 0.966 | 0.958 |
| Random Forest | 0.879 | 0.880 | 0.900 | 0.865 | 0.882 | 0.953 | 0.927 |

### Holdout

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.879 | 0.875 | 0.810 | 1.000 | 0.895 | 0.987 | 0.985 |
| SVC RBF | 0.848 | 0.844 | 0.773 | 1.000 | 0.872 | 0.988 | 0.985 |
| Random Forest | 0.879 | 0.877 | 0.842 | 0.941 | 0.889 | 0.982 | 0.978 |

The strongest classical model is Logistic Regression. This result is scientifically reasonable because the tabular representation already contains informative engineered behavioral features. With 99 samples, a regularized linear model can generalize better than more flexible methods.

## Deep Time-Series Model Results

### Cross-Validation

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM_TS | 0.879 | 0.880 | 0.900 | 0.865 | 0.882 | 0.917 | 0.921 |
| CNN1D_TS | 0.828 | 0.829 | 0.857 | 0.808 | 0.832 | 0.918 | 0.901 |
| Transformer_TS | 0.818 | 0.817 | 0.815 | 0.846 | 0.830 | 0.868 | 0.887 |

### Holdout

| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| BiLSTM_TS | 0.900 | 0.900 | 0.833 | 1.000 | 0.909 | 0.983 | 0.980 |
| CNN1D_TS | 0.900 | 0.900 | 0.833 | 1.000 | 0.909 | 0.981 | 0.980 |
| Transformer_TS | 0.850 | 0.850 | 0.818 | 0.900 | 0.857 | 0.948 | 0.930 |

The strongest deep temporal model is BiLSTM_TS in cross-validation. This suggests that bidirectional temporal modeling is useful for session-level behavioral trajectories. CNN1D_TS also performs strongly on the holdout split, while Transformer_TS is weaker in this small-sample setting.

## Overall Model Comparison

| Rank | Model | Family | ROC-AUC | PR-AUC | Balanced accuracy | F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Logistic Regression | Classical | 0.976 | 0.978 | 0.898 | 0.906 |
| 2 | SVC RBF | Classical | 0.958 | 0.966 | 0.887 | 0.897 |
| 3 | Random Forest | Classical | 0.927 | 0.953 | 0.880 | 0.882 |
| 4 | BiLSTM_TS | Deep | 0.921 | 0.917 | 0.880 | 0.882 |
| 5 | CNN1D_TS | Deep | 0.901 | 0.918 | 0.829 | 0.832 |
| 6 | Transformer_TS | Deep | 0.887 | 0.868 | 0.817 | 0.830 |

The classical models outperform deep temporal models in cross-validation. The most likely explanation is sample efficiency: the classical models use 149 engineered behavioral features, while deep models must learn temporal representations from only 99 sequences.

## Modality-Level Interpretation

The second notebook evaluates predictive performance by modality.

| Modality | Number of features | CV ROC-AUC |
|---|---:|---:|
| Tobii / gaze | 62 | 0.945 |
| Mimic | 44 | 0.916 |
| Face points | 13 | 0.855 |
| Emotion | 3 | 0.682 |

The strongest standalone modality is Tobii/gaze. This agrees with the exploratory analysis, where gaze validity, gaze-object distance and AOI-related features dominate the effect-size ranking. Mimic features also provide strong complementary information. Face-point features are useful but weaker, while emotion features alone are the least informative modality.

## Multitask Inference

The second notebook generates reproducible inference outputs with overall and modality-specific probabilities.

| Output group | Meaning |
|---|---|
| `prob_asd_overall_classical_pct` | ASD probability from the selected classical model |
| `prob_asd_overall_deep_pct` | ASD probability from the selected deep model |
| `prob_asd_overall_ensemble_pct` | Combined classical/deep ASD probability |
| `prob_asd_tobii_pct` | ASD probability from Tobii/gaze features |
| `prob_asd_emotion_pct` | ASD probability from emotion features |
| `prob_asd_mimic_pct` | ASD probability from mimic features |
| `prob_asd_face_points_pct` | ASD probability from face-point features |

| Probability column | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| Overall classical ASD, % | 52.44 | 48.54 | 0.00 | 91.51 | 100.00 |
| Overall deep ASD, % | 57.22 | 17.18 | 12.71 | 59.05 | 87.72 |
| Overall ensemble ASD, % | 54.83 | 30.37 | 6.36 | 74.01 | 93.86 |
| Tobii ASD, % | 52.24 | 45.65 | 0.00 | 79.17 | 100.00 |
| Emotion ASD, % | 50.29 | 16.73 | 24.33 | 49.76 | 87.73 |
| Mimic ASD, % | 51.99 | 41.98 | 0.01 | 55.93 | 100.00 |
| Face-points ASD, % | 50.93 | 28.17 | 1.70 | 41.75 | 100.00 |

This inference design is useful because it provides both a final ASD/TD probability and separate modality-level probabilities.

## Interpretation of Results

The experiments show that ASD/TD classification is mainly supported by gaze-related and object-interaction behavior. The strongest EDA features and the highest modality-level ROC-AUC both point to Tobii/gaze features as the dominant signal source.

Classical models outperform deep temporal models in cross-validation because the engineered tabular features are highly informative and the dataset is small. Deep models still achieve strong holdout results, especially BiLSTM_TS and CNN1D_TS, which suggests that temporal dynamics contain useful information but require careful validation on larger datasets.

The unsupervised analysis shows only moderate cluster separation. This means the ASD/TD distinction is not represented as two simple natural clusters; instead, the classification signal is distributed across multiple behavioral features and temporal patterns.

## Reproducibility

Run the notebooks in this order:

```text
1. 01_dataset_exploration.ipynb
2. 02_model_training_and_multitask_inference.ipynb
```

The notebooks assume that the required data directory and the `analysis_tools` package are available in the execution environment.

## Scope and Limitations

The reported results are experimental and should be interpreted as research evidence only. The dataset contains 99 sessions, so independent external validation is required before making broader generalization claims.

During model training, one warning appears because an internal split contains only one class. This should be addressed in future validation protocols by enforcing class-balanced folds.
