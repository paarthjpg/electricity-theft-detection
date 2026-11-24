# Results

Generated from actual evaluation outputs in `artifacts/metrics/`. Dataset mode: FULL MODE (complete dataset), n_customers=42285.

## Model Comparison (test set)

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | P@5% | P@10% |
|---|---|---|---|---|---|---|---|---|---|
| gcn_k5 | 0.8841 | 0.6129 | 0.3050 | 0.2862 | 0.2953 | 0.7432 | 0.2562 | 0.3817 | 0.2776 |
| graphsage_k10 | 0.8885 | 0.6398 | 0.3421 | 0.3401 | 0.3411 | 0.7637 | 0.2821 | 0.4101 | 0.3091 |
| graphsage_k3 | 0.8870 | 0.6397 | 0.3364 | 0.3420 | 0.3392 | 0.7599 | 0.2827 | 0.4006 | 0.3107 |
| graphsage_k5_randomized | 0.8663 | 0.6285 | 0.2714 | 0.3420 | 0.3026 | 0.7533 | 0.2601 | 0.3502 | 0.2760 |
| graphsage_k5 | 0.8841 | 0.6458 | 0.3310 | 0.3587 | 0.3443 | 0.7628 | 0.2822 | 0.4069 | 0.3202 |
| logistic_regression | 0.8660 | 0.6207 | 0.2644 | 0.3253 | 0.2917 | 0.7536 | 0.2502 | 0.3691 | 0.2697 |
| random_forest | 0.8751 | 0.6670 | 0.3191 | 0.4164 | 0.3613 | 0.7780 | 0.3109 | 0.4259 | 0.3328 |
| xgboost | 0.8868 | 0.6616 | 0.3500 | 0.3903 | 0.3691 | 0.7794 | 0.3464 | 0.4732 | 0.3407 |

## Ranking by PR-AUC (primary metric for imbalanced theft detection)

1. **xgboost**: 0.3464
2. **random_forest**: 0.3109
3. **graphsage_k3**: 0.2827
4. **graphsage_k5**: 0.2822
5. **graphsage_k10**: 0.2821
6. **graphsage_k5_randomized**: 0.2601
7. **gcn_k5**: 0.2562
8. **logistic_regression**: 0.2502

## Honest Comparison

The best conventional baseline (**xgboost**, PR-AUC=0.3464) matched or outperformed the best GNN (**graphsage_k3**, PR-AUC=0.2827) on this dataset and split. This is reported as-is rather than adjusted: possible causes include weak homophily in the k-NN similarity graph relative to the engineered summary-statistic features, class imbalance limiting the GNN's effective supervision signal, oversmoothing across only two message-passing layers being insufficient to exploit relational structure, or the engineered features already capturing most of the discriminative signal present in the raw consumption series.

Randomized-edge control: GraphSAGE with the real similarity graph scored PR-AUC=0.2822 vs. PR-AUC=0.2601 with edges randomly shuffled (same degree sequence, same edge count, no label information used). The real graph's specific structure provides a measurable benefit over an arbitrary graph of the same size.
