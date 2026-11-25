# Results

Generated from actual evaluation outputs in `artifacts/metrics/`. Dataset mode: FULL MODE (complete dataset), n_customers=42285.

## Model Comparison (test set)

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | P@5% | P@10% |
|---|---|---|---|---|---|---|---|---|---|
| logistic_regression | 0.8660 | 0.6207 | 0.2644 | 0.3253 | 0.2917 | 0.7536 | 0.2502 | 0.3691 | 0.2697 |
| random_forest | 0.8751 | 0.6670 | 0.3191 | 0.4164 | 0.3613 | 0.7780 | 0.3109 | 0.4259 | 0.3328 |
| xgboost | 0.8868 | 0.6616 | 0.3500 | 0.3903 | 0.3691 | 0.7794 | 0.3464 | 0.4732 | 0.3407 |

## Ranking by PR-AUC (primary metric for imbalanced theft detection)

1. **xgboost**: 0.3464
2. **random_forest**: 0.3109
3. **logistic_regression**: 0.2502

## Summary

**xgboost** achieves the best PR-AUC (0.3464) of the three models evaluated on this imbalanced theft-detection task. Accuracy alone is not a reliable signal here given the ~8.5% theft prevalence — a trivial all-negative classifier already scores above 91% accuracy — so PR-AUC and Precision@K are reported as the primary metrics.
