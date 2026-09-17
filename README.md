# Insurance Service Call Prediction

Final project for DSC 4310 Machine Learning at Baylor University. The goal is to predict which insurance policyholders will call customer service so a proactive outreach campaign can reach likely callers first.

## Data

130,086 policyholder records across eight processing dates in May 2014, with a 3.66% overall call rate. The dataset is not included in this repository. To run the code, place `data.csv` in the project folder or set the `DATA_DIR` environment variable.

## Approach

The six earliest dates are used for development and the last two are held out as a forward test set. Cleaning removes a constant column and 278 duplicate rows, and fills a shared block of missing six month values with zero plus an indicator. Logistic regression, a decision tree, and a random forest were tuned with 5 fold stratified cross validation on ROC-AUC. The random forest on the raw feature set was selected. Random over and under sampling were also tested for class imbalance.

## Results

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | 0.864 | 0.851 |

Contacting the top 10% of customers by model score reached 53% of all test period callers. At the same contact volume as a simple Channel 4 payment rule, the model reached 72% of callers against 63% for the rule.

## Files

`insurance_service_call_prediction.ipynb` is the full notebook with outputs. `insurance_service_call_prediction.py` is the same code as a script.

## Running

```
pip install -r requirements.txt
jupyter notebook insurance_service_call_prediction.ipynb
```
