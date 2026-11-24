# Raw data

Place the **IBM Telco Customer Churn** CSV in this folder. Any file name is accepted:
`src/data.find_csv()` picks up any `*.csv` here that has the expected 21 columns.

Option A: let the pipeline download it from IBM's public GitHub repository:

```bash
python scripts/train.py --download
```

Option B: download it manually from one of these sources and save it here:

- Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- IBM: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv

Expected shape: 7,043 rows × 21 columns. The CSV is git-ignored and should not be committed. Check the
source's terms before redistributing it.
