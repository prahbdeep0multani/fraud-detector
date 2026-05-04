# Credit Card Fraud Detection

A Random Forest classifier trained on the ULB Credit Card Fraud dataset, with EDA, preprocessing, evaluation metrics, and visualizations.

## How to Run

### 1. Get the dataset (free, ~150 MB)

Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
Download `creditcard.csv` and place it inside the `fraud_detection/` folder.

### 2. Install dependencies

```bash
cd ~/fraud_detection
pip install -r requirements.txt
```

### 3. Run

```bash
python fraud_detection.py                      # full run with EDA
python fraud_detection.py --no-eda             # skip EDA plots, faster
python fraud_detection.py --data /path/to/creditcard.csv
```

## What the app does

| Step | Details |
|------|---------|
| EDA | Class balance bar chart, amount distribution, correlation heatmap → saved to `eda.png` |
| Preprocessing | Drops `Time`, `StandardScaler` on `Amount`, stratified 80/20 train/test split |
| Model | Random Forest (100 trees, `class_weight="balanced"` to handle the 0.17% fraud skew) |
| Metrics | Accuracy, Precision, Recall, F1, MCC, ROC-AUC, full classification report |
| Plots | Confusion matrix, ROC curve, top-15 feature importances → saved to `results.png` |
| Demo | Predicts fraud probability on one test transaction |

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Key design choice

`class_weight="balanced"` is used instead of a vanilla Random Forest so the model doesn't just predict "valid" for everything — the naive 99.83% accuracy trap on this heavily imbalanced dataset.
