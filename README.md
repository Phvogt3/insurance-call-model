# Insurance Service Call Prediction

Final project for DSC 4310 Machine Learning at Baylor University. The goal is to rank insurance policyholders by how likely they are to make a service payment call in the next five days, so a proactive email campaign can reach likely callers first and encourage them to pay through online self service.

## Data

The data has 130,086 policyholder records across eight processing dates from May 13 to May 20, 2014, with a 3.66% overall call rate. `data.csv` holds the records and `data-dictionary.pdf` describes each column. The notebook reads `data.csv` from the project folder by default, and the `INSURANCE_DATA_PATH` environment variable can point it to another location.

## Approach

A record's outcome is only usable once its five day window has closed, so the notebook waits six calendar days before training on it, which covers five days for calls to occur plus one day because exact times are missing. Validation trains on May 13 and scores May 19. The final test trains on May 13 and May 14 and scores May 20.

Histogram gradient boosting was chosen in advance and kept fixed because May 19 outcomes would not be known in time to pick the model that scores May 20. It is compared against logistic regression and a simple business rule that ranks customers by their Channel 4 payments over the last six months, and every approach contacts the top 10% of each day's records.

The notebook adds a flag for missing payment history and a measure of Channel 4's share of all payments over six months. It also adds a flag for whether the customer has logged into self service. All preprocessing is fit on training data only.

## Results

On the held out test date the model contacts 1,499 records, 10% of that day's book, and finds **48% of everyone who called**. Finding the same share of callers at random would mean contacting about half the book.

<table>
  <tr>
    <th>Approach, top 10% of May 20</th>
    <th>Callers reached</th>
    <th>Share of all 382 callers</th>
    <th>Call rate among contacted</th>
    <th>Lift</th>
  </tr>
  <tr>
    <td>Gradient boosting model</td>
    <td>184</td>
    <td>48.2%</td>
    <td>12.3%</td>
    <td>4.8x</td>
  </tr>
  <tr>
    <td>Channel 4 payment rule</td>
    <td>141</td>
    <td>36.9%</td>
    <td>9.4%</td>
    <td>3.7x</td>
  </tr>
</table>

Random outreach at the same budget would be expected to reach about 38 callers, or 10% of them. The model reaches 43 more callers than the business rule at the same contact cost, which is an 11.3 point gain in capture. Its call rate among contacted customers is nearly five times the random baseline, and its ROC AUC is 0.839 on the test date.

On the May 19 validation date, gradient boosting captured 55.4% of callers compared with 54.4% for logistic regression. That gap is under one point on a single date, so more dates are needed to show whether one model is consistently better. The Channel 4 rule captured 44.8% on the same date. Channel 4's share of payments was by far the most important input, followed by login count.

## Limitations

Call rates drop to 3.19% on May 19 and 2.55% on May 20, compared with 3.7% to 4.2% on earlier dates. Those outcomes may be incomplete, so results on those two dates should be read as a floor. There is no customer ID in the data, so repeat customers cannot be identified and the ranked list needs a duplicate contact check before use. Measuring whether an email actually prevents calls would take a randomized email test.

## Files

`insurance_service_call_prioritization_revised.ipynb` is the full notebook with outputs, and `insurance_service_call_prioritization_revised.py` is the same code exported as a script. `data.csv` is the dataset and `data-dictionary.pdf` is its column reference.

## Running

The notebook was run with Python 3.14, pandas 3.0, NumPy 2.4, and sklearn 1.9.

    pip install numpy pandas scikit_learn matplotlib seaborn jupyter
    jupyter notebook insurance_service_call_prioritization_revised.ipynb
