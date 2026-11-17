# Credit Card Default Prediction (PySpark, Databricks, MLflow)

## Problem
Predict which of 30,000 credit card customers will default next month, using the UCI
"Default of Credit Card Clients" dataset, to support credit risk decisions.

## Tools
Databricks, PySpark, Spark ML, Spark SQL, MLflow

## Approach
1. Cleaned the data (fixed undocumented category codes, checked for nulls)
2. EDA in Spark SQL: default rate by repayment status and age band
3. Engineered features: avg_bill, avg_payment, pay_to_bill_ratio, utilisation, late_count
4. Trained Logistic Regression and Gradient Boosted Trees with class weights (~22% default rate)
5. Tracked runs and metrics with MLflow

## Results
| Model | AUC |
|---|---|
| Logistic Regression | 0.7526 |
| Gradient Boosted Trees | 0.7821 |

Best model (GBT): Precision 0.456, Recall 0.617, Accuracy 0.761.

## Key Insights
- Customers with repayment status ___ default at ___% versus ___% for on-time payers
- `late_count`, my engineered feature, was the single most important predictor (0.132)
- Customers aged ___ had the highest default rate (___%)
- The model catches ~62% of defaulters; flagged customers default at ~2x the average rate

## Next Steps
- Deploy the model behind an API on AWS (Lambda / SageMaker)
- Add CI/CD with GitHub Actions and automated tests
- Tune the decision threshold to balance precision and recall for business cost
