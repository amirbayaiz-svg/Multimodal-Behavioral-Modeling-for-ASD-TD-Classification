# Multimodal Behavioral Modeling for ASD/TD Classification

Профессиональный исследовательский README, подготовленный только на основе двух ноутбуков:

- `01_dataset_exploration.ipynb`
- `02_model_training_and_multitask_inference.ipynb`

Проект описывает полный аналитический и экспериментальный workflow для классификации `ASD` и `TD` на основе мультимодальных поведенческих данных: gaze/Tobii-показателей, координат взаимодействия с объектом, положения головы, facial action units, smile probability, face visibility и других temporal/session-level признаков.

> Назначение: исследовательская воспроизводимость и экспертный анализ моделей. Это не клинический диагностический продукт.

## Scientific Summary

В двух ноутбуках выполнены две взаимосвязанные части исследования:

| Notebook | Scientific role | Main output |
|---|---|---|
| `01_dataset_exploration.ipynb` | Dataset audit, exploratory data analysis, temporal/segment-level analysis, unsupervised structure analysis | Dataset profile, class balance, feature effect sizes, temporal ASD/TD differences, PCA, t-SNE, KMeans, hierarchical clustering |
| `02_model_training_and_multitask_inference.ipynb` | Supervised modeling and reproducible inference | Classical ML, deep time-series models, cross-validation, holdout testing, modality-level interpretation, saved inference output |

Главный научный вывод: различие между ASD и TD в этих данных сильнее всего проявляется в gaze/object-interaction признаках и temporal behavioral dynamics. Это подтверждается как EDA-результатами, так и качеством modality-level моделей.

## Dataset Profile

Первый ноутбук сформировал исследовательский датасет на уровне сессий и временных последовательностей.

| Metric | Value |
|---|---:|
| Sessions | 99 |
| Children | 99 |
| Total extracted features | 224 |
| Modeling features | 149 |
| ASD sessions | 52 |
| TD sessions | 47 |
| ASD ratio | 0.525 |
| Sequence shape | `(99, 160, 14)` |

Class distribution is close to balanced, which is important for supervised binary classification.

![Class distribution](README_assets/01_EDA/class_distribution.png)

## Sequence Representation

The temporal dataset contains 160 normalized time bins and 14 sequence features.

| Temporal feature group | Features used in sequence modeling |
|---|---|
| Eye/object coordinates | `x`, `y`, `center_x`, `center_y`, `gaze_object_distance` |
| Data validity | `validity`, `face_ok`, `object_visible`, `object_exploded` |
| Head movement | `head_yaw`, `head_pitch` |
| Facial behavior | `smile_prob`, `au6`, `au12` |

This representation allows the models to use not only static summary statistics, but also session dynamics.

## Exploratory Data Analysis

EDA shows that the strongest discriminative signals are concentrated in gaze validity, gaze-object distance, gaze dispersion, AOI interaction and gaze-following metrics.

### Top Effect Sizes

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

![Effect size ranking](README_assets/01_EDA/effect_size_top20.png)

Interpretation: the top-ranked variables indicate that ASD/TD separation is driven primarily by differences in visual attention stability, object-oriented gaze behavior and interaction with task-relevant areas of interest.

### Missingness and Feature Distributions

![Missing ratio top 20](README_assets/01_EDA/missing_ratio_top20.png)

![Top feature histograms](README_assets/01_EDA/top_feature_histograms.png)

The missingness plot is important because behavioral data often contain sensor dropout, tracking loss and event-level sparsity. The histogram panel shows that many discriminative variables are not normally distributed, which explains why both linear models with scaling and nonlinear models are scientifically relevant.

### Correlation Structure

![Correlation heatmap](README_assets/01_EDA/correlation_heatmap_top25.png)

The correlation structure shows that many gaze and AOI variables are related but not identical. This supports the use of multivariate models: the classification signal is distributed across several correlated behavioral indicators rather than isolated in a single feature.

## Temporal ASD/TD Differences

The notebooks compare ASD and TD temporal profiles across normalized session time.

| Feature | ASD mean | TD mean | ASD - TD |
|---|---:|---:|---:|
| `gaze_object_distance` | 0.387 | 0.255 | +0.132 |
| `smile_prob` | 0.039 | 0.026 | +0.013 |
| `head_yaw` | 0.394 | -1.107 | +1.500 |
| `validity` | 0.652 | 0.853 | -0.200 |
| `object_visible` | 0.979 | 0.967 | +0.012 |

![Temporal class mean profiles](README_assets/01_EDA/temporal_profiles_class_mean.png)

![Segment metric trends](README_assets/01_EDA/segment_metric_trends.png)

Interpretation: ASD sessions show larger gaze-object distance and lower validity. This means that the ASD/TD difference is not only a class-level difference, but also a temporal behavioral pattern across the task.

## Single-Session Behavioral Panels

The first notebook also compares one TD and one ASD session using normalized temporal profiles.

![Eye object dynamics](README_assets/01_EDA/single_session_eye_object_xy_td_vs_asd.png)

![Head rotation dynamics](README_assets/01_EDA/single_session_head_rotation_td_vs_asd.png)

![Facial mimic dynamics](README_assets/01_EDA/single_session_facial_mimic_td_vs_asd.png)

![Emotion dynamics](README_assets/01_EDA/single_session_emotion_td_vs_asd.png)

These panels are useful for qualitative interpretation. They show how coordinate dynamics, head motion, facial mimic and emotion-related signals change over time in a session, instead of reducing the participant to one static score.

## Unsupervised Structure

The EDA notebook applies PCA, t-SNE, KMeans and hierarchical clustering.

| Metric | Value |
|---|---:|
| Adjusted Rand Index | 0.425 |
| Normalized Mutual Information | 0.379 |
| Silhouette | 0.122 |

![PCA projection](README_assets/01_EDA/pca_projection.png)

![t-SNE projection](README_assets/01_EDA/tsne_projection.png)

![KMeans clusters in PCA space](README_assets/01_EDA/kmeans_clusters_pca.png)

![Hierarchical dendrogram](README_assets/01_EDA/hierarchical_dendrogram.png)

Interpretation: unsupervised structure is present but moderate. The low silhouette value indicates that ASD and TD do not form two perfectly separated natural clusters in the raw feature space. This explains why supervised learning improves performance: labels help combine distributed weak and medium behavioral signals into a stronger decision boundary.

## Supervised Modeling

The second notebook trains and evaluates two groups of models:

| Model family | Models |
|---|---|
| Classical ML | Logistic Regression, SVC with RBF kernel, Random Forest |
| Deep time-series | BiLSTM, CNN1D, Transformer |

The notebook reports:

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

## Classical ML Results

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

![Classical ROC PR curves](README_assets/02_Modeling/classical_roc_pr_curves.png)

Interpretation: Logistic Regression performs best among classical models. This is scientifically plausible because the EDA found strong engineered gaze and AOI predictors. With only 99 samples, a regularized linear model can generalize better than more flexible models when the feature representation is already informative.

## Deep Time-Series Results

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

![Deep ROC curves](README_assets/02_Modeling/deep_roc_curve_models.png)

![Deep training history](README_assets/02_Modeling/deep_training_history_best_model.png)

Interpretation: the BiLSTM model is the strongest deep temporal model in cross-validation. This is expected because BiLSTM can model sequential dependencies in both temporal directions, which is useful for session-level behavioral trajectories. The Transformer is competitive but weaker here, likely because the dataset is small and transformer-style models usually require more data to fully benefit from attention-based representations.

## Overall Model Comparison

![Model comparison](README_assets/02_Modeling/model_comparison_bar.png)

| Rank | Model | Family | ROC-AUC | PR-AUC | Balanced accuracy | F1 |
|---:|---|---|---:|---:|---:|---:|
| 1 | Logistic Regression | Classical | 0.976 | 0.978 | 0.898 | 0.906 |
| 2 | SVC RBF | Classical | 0.958 | 0.966 | 0.887 | 0.897 |
| 3 | Random Forest | Classical | 0.927 | 0.953 | 0.880 | 0.882 |
| 4 | BiLSTM_TS | Deep | 0.921 | 0.917 | 0.880 | 0.882 |
| 5 | CNN1D_TS | Deep | 0.901 | 0.918 | 0.829 | 0.832 |
| 6 | Transformer_TS | Deep | 0.887 | 0.868 | 0.817 | 0.830 |

The classical models outperform deep temporal models in cross-validation, while deep models remain strong on the holdout split. The most defensible explanation is sample efficiency: the tabular representation contains 149 engineered behavioral features, whereas deep sequence models learn from 99 sequences with 160 time bins and 14 temporal channels. For this sample size, engineered features plus regularized classification provide a strong bias-variance tradeoff.

## Modality-Level Interpretation

The second notebook evaluates modality-specific predictive quality.

| Modality | Number of features | CV ROC-AUC |
|---|---:|---:|
| Tobii / gaze | 62 | 0.945 |
| Mimic | 44 | 0.916 |
| Face points | 13 | 0.855 |
| Emotion | 3 | 0.682 |

Interpretation:

- Tobii/gaze is the strongest modality, consistent with the EDA where gaze validity, gaze-object distance and AOI metrics dominate the effect-size ranking.
- Mimic features are also highly informative, suggesting that facial and head-orientation behavior contributes complementary signal.
- Face-point features are useful but weaker than gaze and mimic.
- Emotion features alone are limited, likely because this modality has only three summary features and captures a narrower behavioral channel.

## Multitask Inference Output

The notebook produces reproducible inference with overall and modality-specific probabilities:

| Prediction block | Meaning |
|---|---|
| `prob_asd_overall_classical_pct` | ASD probability from the selected classical model |
| `prob_asd_overall_deep_pct` | ASD probability from the selected deep model |
| `prob_asd_overall_ensemble_pct` | Combined classical/deep probability |
| `prob_asd_tobii_pct` | Modality-level ASD probability from Tobii/gaze features |
| `prob_asd_emotion_pct` | Modality-level ASD probability from emotion features |
| `prob_asd_mimic_pct` | Modality-level ASD probability from mimic features |
| `prob_asd_face_points_pct` | Modality-level ASD probability from face-point features |

Probability distribution across all 99 samples:

| Probability column | Mean | Std | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| Overall classical ASD, % | 52.44 | 48.54 | 0.00 | 91.51 | 100.00 |
| Overall deep ASD, % | 57.22 | 17.18 | 12.71 | 59.05 | 87.72 |
| Overall ensemble ASD, % | 54.83 | 30.37 | 6.36 | 74.01 | 93.86 |
| Tobii ASD, % | 52.24 | 45.65 | 0.00 | 79.17 | 100.00 |
| Emotion ASD, % | 50.29 | 16.73 | 24.33 | 49.76 | 87.73 |
| Mimic ASD, % | 51.99 | 41.98 | 0.01 | 55.93 | 100.00 |
| Face-points ASD, % | 50.93 | 28.17 | 1.70 | 41.75 | 100.00 |

The inference design is scientifically useful because it gives both a final ASD/TD decision and interpretable modality-level probabilities.

## Why The Results Look This Way

1. Gaze-related features dominate because the task is behaviorally centered around visual attention and object interaction. This is visible in the highest Cohen's d values and in the Tobii/gaze ROC-AUC.
2. Classical models perform very strongly because the engineered tabular features already encode high-level behavioral summaries. With 99 samples, this representation is statistically efficient.
3. Deep time-series models still perform well because they capture dynamic patterns, but the dataset size limits the advantage of high-capacity sequence models.
4. Unsupervised clustering is moderate because ASD/TD differences are distributed, overlapping and multidimensional. The data are not separated by one simple geometric boundary.
5. Modality-level inference shows that no single behavioral channel explains everything. Gaze is strongest, mimic is complementary, face points add spatial facial behavior, and emotion alone is weaker but still informative.

## Reproducible Workflow

Recommended execution order:

```text
1. Run 01_dataset_exploration.ipynb
2. Run 02_model_training_and_multitask_inference.ipynb
```

Expected generated outputs from the notebooks:

```text
Analysis_Result/
├── 01_dataset_exploration.ipynb
├── 02_model_training_and_multitask_inference.ipynb
├── 01_EDA/
│   ├── tables/
│   └── figures/
└── 02_Modeling/
    ├── tables/
    ├── figures/
    └── models/
```

This README also includes extracted notebook-output figures under:

```text
README_assets/
├── 01_EDA/
└── 02_Modeling/
```

## Graphical Artifact Index

| Area | Figure |
|---|---|
| EDA | `README_assets/01_EDA/class_distribution.png` |
| EDA | `README_assets/01_EDA/missing_ratio_top20.png` |
| EDA | `README_assets/01_EDA/effect_size_top20.png` |
| EDA | `README_assets/01_EDA/top_feature_histograms.png` |
| EDA | `README_assets/01_EDA/correlation_heatmap_top25.png` |
| EDA | `README_assets/01_EDA/pca_projection.png` |
| EDA | `README_assets/01_EDA/tsne_projection.png` |
| EDA | `README_assets/01_EDA/kmeans_clusters_pca.png` |
| EDA | `README_assets/01_EDA/hierarchical_dendrogram.png` |
| EDA | `README_assets/01_EDA/segment_metric_trends.png` |
| EDA | `README_assets/01_EDA/temporal_profiles_class_mean.png` |
| EDA | `README_assets/01_EDA/single_session_eye_object_xy_td_vs_asd.png` |
| EDA | `README_assets/01_EDA/single_session_head_rotation_td_vs_asd.png` |
| EDA | `README_assets/01_EDA/single_session_facial_mimic_td_vs_asd.png` |
| EDA | `README_assets/01_EDA/single_session_emotion_td_vs_asd.png` |
| Modeling | `README_assets/02_Modeling/classical_roc_pr_curves.png` |
| Modeling | `README_assets/02_Modeling/deep_roc_curve_models.png` |
| Modeling | `README_assets/02_Modeling/deep_training_history_best_model.png` |
| Modeling | `README_assets/02_Modeling/model_comparison_bar.png` |

## Limitations

- The dataset contains 99 sessions, so all model results should be interpreted as experimental evidence, not as clinical validation.
- The strongest cross-validation performance comes from engineered tabular features; external validation on independent cohorts would be required for stronger generalization claims.
- A warning appears during classical model fitting for some folds because one internal split contained only one class. The final reported model tables were still produced, but this should be considered when designing future validation protocols.

## Recommended GitHub Repository Name

Short repository name:

```text
asd-multimodal-behavior-modeling
```

Scientific repository title:

```text
Multimodal Behavioral Modeling for ASD/TD Classification
```
