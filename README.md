# Insurance Service Call Prediction

Final project for DSC 4310 Machine Learning at Baylor University. The goal is to rank insurance policyholders by how likely they are to make a service payment call in the next five days, so a proactive email campaign can reach likely callers first and encourage them to pay through online self-service.

## Data

The data has 130,086 policyholder records across eight processing dates, May 13 to May 20, 2014, with a 3.66% overall call rate. `data.csv` holds the records and `data-dictionary.pdf` describes each column. The notebook reads `data.csv` from the project folder by default. To load it from somewhere else, set the `INSURANCE_DATA_PATH` environment variable.

## Approach

A record's outcome is only usable once its five-day window has closed, so the notebook waits six calendar days before training on it (five days for calls to occur plus one day because exact times are missing). Validation trains on May 13 and scores May 19. The final test trains on May 13 and May 14 and scores May 20.

Histogram gradient boosting was chosen in advance and kept fixed, since May 19 outcomes would not be known in time to pick the model that scores May 20. It is compared against logistic regression and a simple business rule that ranks customers by their Channel 4 payments over the last six months. Every approach gets the same contact budget: the top 10% of each day's records.

Three inputs are added to the raw data: a flag for missing payment history, Channel 4's share of all payments over six months, and whether the customer has logged into self-service. Imputation, scaling, and encoding are fit on training data only.

## Results

On the held-out test date the model contacts 1,499 records, 10% of that day's book, and finds **48% of everyone who called**. Finding the same share of callers at random would mean contacting about half the book.

| Approach, top 10% of May 20 | Callers reached | Share of all 382 callers | Call rate among contacted | Lift |
|---|---|---|---|---|
| Gradient boosting model | 184 | 48.2% | 12.3% | 4.8x |
| Channel 4 payment rule | 141 | 36.9% | 9.4% | 3.7x |
| Random outreach (expected) | 38 | 10.0% | 2.5% | 1.0x |

The model reaches 43 more callers than the business rule at the same contact cost, an 11.3 point gain in capture, and its call rate among contacted customers is nearly five times the random baseline. ROC-AUC is 0.839 on the test date.

On the May 19 validation date, gradient boosting captured 55.4% of callers, logistic regression 54.4%, and the Channel 4 rule 44.8%. The gap between the two models is under one point on a single date, so it does not show that one is consistently better. Channel 4's share of payments was by far the most important input, followed by login count.

## Limitations

Call rates drop to 3.19% on May 19 and 2.55% on May 20, compared with 3.7% to 4.2% on earlier dates. Those outcomes may be incomplete, so results on those two dates should be read as a floor. There is no customer ID in the data, so repeat customers cannot be identified and the ranked list needs a duplicate contact check before use. The model predicts who will call; it does not show whether an email would prevent the call. A randomized email test would be needed to measure that.

## Files

`insurance_service_call_prioritization_revised.ipynb` is the full notebook with outputs. `insurance_service_call_prioritization_revised.py` is the same code exported as a script. `data.csv` is the dataset and `data-dictionary.pdf` is its column reference.

## Running

The notebook was run with Python 3.14, pandas 3.0, NumPy 2.4, and scikit-learn 1.9.

    pip install numpy pandas scikit-learn matplotlib seaborn jupyter
    jupyter notebook insurance_service_call_prioritization_revised.ipynb
