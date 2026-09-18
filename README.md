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

On the held out test date the model contacts 1,499 records, 10% of that day's book, and finds **48% of
everyone who called**. Reaching the same callers at random would take half the book.

| Approach, top 10% of May 20 | Callers reached | Share of all 382 callers | Call rate among contacted | Lift |
| :-- | :-- | :-- | :-- | :-- |
| Model | 184 | 48.2% | 12.3% | 4.8x |
| Channel 4 payment rule | 141 | 36.9% | 9.4% | 3.7x |
| Random outreach | 38 | 10.0% | 2.5% | 1.0x |

The model beats the business rule by 43 extra callers at identical contact cost, an 11.3 point gain in
capture, and it beats random outreach by nearly five times. ROC-AUC is 0.839 on the test date and capture at
10% was 55.4% on the validation date.

Call rates fall on May 19 and May 20, which may mean their outcomes are incomplete, so those two dates are
reported as a floor rather than as final truth. There is no customer ID in the data, so repeat customers
cannot be identified and the ranked output needs a duplicate contact check before use.

## Files

`insurance_service_call_prediction.ipynb` is the full notebook with outputs.
`insurance_service_call_prediction.py` is the same code as a script.

## Running

```bash
pip install -r requirements.txt
jupyter notebook insurance_service_call_prediction.ipynb
```
