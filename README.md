# Insurance Service Call Prediction

Final project for DSC 4310 Machine Learning at Baylor University. The goal is to rank insurance policyholders
by how likely they are to make a service payment call in the next five days, so a proactive email campaign can
reach likely callers first and have them pay using online self service.

## Data

130,086 policyholder records across eight processing dates in May 2014, with a 3.66% overall call rate. The
dataset is not included in this repository. To run the code, place `data.csv` in the project folder or set the
`INSURANCE_DATA_PATH` environment variable.

## Approach

A record's outcome is only usable once its five day window has closed, so the notebook waits six calendar days
before training on it. Validation trains on May 13 and scores May 19. The final test trains on May 13 and
May 14 and scores May 20. The model is fixed in advance, since May 19 outcomes would not be known in time to
choose the model that scores May 20. Added inputs are a missing payment history flag, Channel 4's share of
payments, and whether the customer logs into self service. Histogram gradient boosting is compared against
logistic regression and a Channel 4 payment rule at the same contact budget.

## Results

| Metric | Validation, May 19 | Test, May 20 |
| :-- | :-- | :-- |
| Capture at 10% | 55.35% | 48.17% |
| ROC-AUC | | 0.839 |

Contacting the top 10% of records by model score reached 184 of 382 callers on the test date, at a 12.27% call
rate among contacted customers and 4.82 times the rate of random outreach. That is 11.26 percentage points more
callers than the Channel 4 payment rule at the same contact volume.

Call rates fall on May 19 and May 20, which may mean their outcomes are incomplete. Results assume complete
tracking through May 25. There is no customer ID in the data, so repeat customers cannot be identified and the
ranked output needs a duplicate contact check before use.

## Files

`insurance_service_call_prediction.ipynb` is the full notebook with outputs.
`insurance_service_call_prediction.py` is the same code as a script.

## Running

```bash
pip install -r requirements.txt
jupyter notebook insurance_service_call_prediction.ipynb
```
