# Model comparison

Test set: 1409 customers, churn rate 0.265. Selected model (by CV ROC-AUC on train): **XGBoost**.

## Test metrics at threshold 0.5

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.800 | 0.655 | 0.519 | 0.579 | 0.842 |
| Decision Tree | 0.777 | 0.577 | 0.604 | 0.590 | 0.812 |
| Random Forest | 0.799 | 0.667 | 0.487 | 0.563 | 0.844 |
| Gradient Boosting | 0.802 | 0.663 | 0.516 | 0.580 | 0.846 |
| XGBoost | 0.799 | 0.659 | 0.505 | 0.572 | 0.848 |

## Test metrics at F1-optimal threshold (chosen on out-of-fold train predictions)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.363 | 0.776 | 0.562 | 0.701 | 0.624 | 0.842 |
| Decision Tree | 0.485 | 0.758 | 0.536 | 0.655 | 0.590 | 0.812 |
| Random Forest | 0.333 | 0.774 | 0.554 | 0.751 | 0.638 | 0.844 |
| Gradient Boosting | 0.320 | 0.776 | 0.558 | 0.741 | 0.637 | 0.846 |
| XGBoost | 0.352 | 0.781 | 0.568 | 0.725 | 0.637 | 0.848 |

## Cross-validation (train, 5-fold, tuned)

| model | imbalance_strategy | cv_roc_auc | cv_roc_auc_std | oof_f1_threshold |
|---|---|---|---|---|
| Logistic Regression | none | 0.8461 | 0.0115 | 0.3628 |
| Decision Tree | smote | 0.8197 | 0.0068 | 0.4851 |
| Random Forest | none | 0.8479 | 0.0104 | 0.3326 |
| Gradient Boosting | none | 0.8502 | 0.0110 | 0.3198 |
| XGBoost | none | 0.8508 | 0.0114 | 0.3515 |

## Imbalance experiment (5-fold CV on train, untuned models, threshold 0.5)

| model | strategy | accuracy | precision | recall | f1 | roc_auc | roc_auc_std |
|---|---|---|---|---|---|---|---|
| Logistic Regression | none | 0.805 | 0.665 | 0.534 | 0.592 | 0.846 | 0.011 |
| Logistic Regression | class_weight | 0.750 | 0.519 | 0.796 | 0.628 | 0.846 | 0.011 |
| Logistic Regression | smote | 0.754 | 0.525 | 0.787 | 0.629 | 0.844 | 0.012 |
| Decision Tree | none | 0.725 | 0.483 | 0.502 | 0.492 | 0.654 | 0.018 |
| Decision Tree | class_weight | 0.736 | 0.502 | 0.493 | 0.497 | 0.659 | 0.019 |
| Decision Tree | smote | 0.727 | 0.486 | 0.518 | 0.502 | 0.661 | 0.009 |
| Random Forest | none | 0.793 | 0.647 | 0.482 | 0.552 | 0.827 | 0.011 |
| Random Forest | class_weight | 0.778 | 0.570 | 0.654 | 0.609 | 0.828 | 0.010 |
| Random Forest | smote | 0.783 | 0.595 | 0.570 | 0.582 | 0.824 | 0.010 |
| Gradient Boosting | none | 0.803 | 0.661 | 0.532 | 0.589 | 0.848 | 0.012 |
| Gradient Boosting | smote | 0.791 | 0.597 | 0.658 | 0.626 | 0.845 | 0.012 |
| XGBoost | none | 0.802 | 0.660 | 0.520 | 0.582 | 0.845 | 0.009 |
| XGBoost | class_weight | 0.761 | 0.535 | 0.775 | 0.633 | 0.843 | 0.010 |
| XGBoost | smote | 0.792 | 0.608 | 0.614 | 0.611 | 0.842 | 0.010 |
