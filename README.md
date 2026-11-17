# Credit Card Default Prediction (PySpark, Databricks, MLflow)

## Problem
Predict which of 30,000 credit card customers will default next month, using the UCI
"Default of Credit Card Clients" dataset, to support credit risk decisions.

## Tools
Databricks, PySpark, Spark ML, Spark SQL, MLflow

## Approach
1. Cleaned the data: grouped undocumented EDUCATION and MARRIAGE codes, checked for nulls
2. EDA in Spark SQL: default rate by repayment status and by age band
3. Engineered features: `avg_bill`, `avg_payment`, `pay_to_bill_ratio`, `utilisation`, `late_count`
4. Trained Logistic Regression and Gradient Boosted Trees with class weights (about 22% of customers default)
5. Tracked runs, parameters and metrics with MLflow

## Results
80/20 train/test split (24,087 train rows, 5,913 test rows), 28 features.

| Model | AUC |
|---|---|
| Logistic Regression | 0.7526 |
| Gradient Boosted Trees | 0.7821 |

Best model (GBT) on the test set: Precision 0.456, Recall 0.617, Accuracy 0.761.

| | Predicted no default | Predicted default |
|---|---|---|
| **Actual no default** | 3,720 | 930 |
| **Actual default** | 484 | 779 |

## Key Insights
- **Repayment delay is the strongest signal.** Customers one month late (PAY_1 = 1) default at 33.9% and two months late at 69.1%, versus 12.8% for the largest group, customers with no delay (PAY_1 = 0, 14,737 customers).
- **The engineered `late_count` feature separates risk clearly.** Default rate rises from 11.7% for customers never late (19,931 customers) to 70.3% for customers late in all six months (1,341 customers), about 6x higher.
- **`late_count` was the single most important predictor** in the GBT model (importance 0.132). Three more engineered features (`pay_to_bill_ratio`, `utilisation`, `avg_payment`) are in the top seven.
- **Age is a weak signal on its own.** Default rate ranges only from 20.3% (ages 30-39) to 25.3% (ages 50+).
- **Business view:** the model flags 1,709 of 5,913 test customers and catches 779 of the 1,263 actual defaulters (61.7%). Flagged customers default at 45.6%, about 2.1x the 21.4% base rate in the test set.

## Repository contents
- `credit-default-prediction-pyspark-databricks.ipynb`: the full Databricks notebook
- `UCI_Credit_Card.csv`: the dataset, included so the notebook can be reproduced
- `screenshots/`: key outputs (repayment-status default rates, MLflow run comparison, feature importances)

## Limitations
- Single train/test split with default model settings: no cross-validation or hyperparameter tuning yet
- Predictions use the default 0.5 decision threshold, which is not tuned to the cost of a missed default
- Demographic features (age, sex, marital status, education) are included. A production credit model would need a fairness and regulatory review before using them.

## Next Steps
- Cross-validation and hyperparameter tuning; tune the decision threshold against business cost
- Store the data in Amazon S3 and validate the analysis with Amazon Athena (SQL)
- Deploy the model behind an API on AWS (Lambda or SageMaker)
- Add CI/CD with GitHub Actions and automated tests
