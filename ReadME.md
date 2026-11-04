# Peloton Subscriber Churn Analysis

Why do subscribers of a fitness app cancel, and what can the business do to keep them? This project cleans and combines subscription and usage data, measures churn fairly across customer groups, builds machine learning models that predict which members will cancel next month, and turns the findings into an evidence-backed retention plan.

## Dataset

| File | Rows | Contents |
| --- | --- | --- |
| `datasets/subscriptions.csv` | 4,200 | One row per customer: signup date, plan type and tier, price, acquisition channel, device, age group, churn status, churn date, cancel reason |
| `datasets/usage.csv` | 30,685 | One row per customer per month: workouts, minutes active, classes booked, support tickets |

Data covers **July 2024 to December 2025**.

## Tools

- **Python:** pandas, matplotlib, seaborn (Jupyter notebook)
- **Machine learning:** scikit-learn (logistic regression, random forest)

## Method

1. **Cleaning:** removed 160 duplicate usage rows, standardised plan tier labels (`Basic` to `basic`), filled missing cancel reasons (`unknown` for churners, `not_churned` for active members), and estimated 506 missing `minutes_active` values from workouts (about 30 minutes per workout), with a flag column for the filled rows.
2. **Fair churn measure:** recent signups have had less time to cancel, so overall churn makes them look more loyal. I used a **monthly churn rate** instead: cancellations in a month divided by members active that month.
3. **Customer-month table:** one row per customer per active month, with a `cancelled` flag (1 only in the month they left) and a `tenure_month` column (1st, 2nd, 3rd month of membership).
4. **Group comparisons:** churn by plan type, plan tier, acquisition channel, device, age group and tenure.
5. **Usage join:** matched monthly usage to each member-month to compare engagement and support contact with churn, including first month behaviour.
6. **Tenure check:** re-tested each factor within tenure bands to confirm it holds at every stage of membership.
7. **Cancel reasons:** compared what churners said with what their data showed.
8. **Prediction:** trained models to predict whether an active member will cancel next month.

## Key Findings

| Finding | Result |
| --- | --- |
| Average monthly churn | **5.6%** of active members cancel each month |
| Low engagement | 0 workouts in a month: **13.8%** churn; 6 or more: **2.8%** |
| First month matters | No workouts in month 1: **59%** eventually churn; 6 or more: **25%** |
| Support contact | Raised a ticket: **11.7%** churn that month; no ticket: **5.0%** |
| Promo signups | Highest churn by channel (**7.3%**); referrals lowest (**4.1%**) |
| Risk points | First 3 months, and month 12, when **55%** of annual members cancel at renewal |
| No real effect | Plan tier, device and age group all churn at about 5 to 6% |

**Top 3 factors (holding within every tenure band):** low engagement, support contact, promo acquisition.

**Plan type does not hold up.** Monthly members churn more each month (7.5% vs 3.0%), but annual members leave at renewal instead, so both plans keep about **41%** of members after 12 months. Plan type changes when members leave, not whether they leave.

**Cancel reasons vs data:** "not using enough" and "technical issues" match the data (low workouts; more support tickets and Android users). "Too expensive" customers paid no more than others, but 42% came from promo offers, which points to price shock when a discount ends. Around 40% of churners who gave other reasons had also nearly stopped using the app, so low engagement is under-reported.

## Predicting Churn

**Goal:** predict whether an active member will cancel **next month**, so the retention team can act before they leave.

**Setup**
- **Data:** 26,464 member-months; 6.1% are followed by a cancellation.
- **Target:** whether the member cancels in the following month.
- **Features:** plan type and tier, price, acquisition channel, device, age group, tenure month, a renewal-next-month flag for annual members, and usage (workouts, minutes, classes, support tickets, plus last month's workouts, the change since last month and a 3-month average).
- **Leakage avoided:** churn date, churned flag and cancel reason are excluded.
- **Split by time:** trained on Jul 2024 to Aug 2025, tested on Sep to Nov 2025, so the model is always tested on months it has never seen.
- **Class imbalance:** handled with `class_weight='balanced'`; models are judged on recall, precision and ROC-AUC rather than accuracy.

**Results (test months)**

| Model | Recall | Precision | ROC-AUC |
| --- | --- | --- | --- |
| Logistic regression | 0.81 | 0.16 | 0.85 |
| Random forest | 0.67 | 0.24 | 0.85 |

- Both models rank members well (ROC-AUC 0.85). Logistic regression catches more churners; random forest raises fewer false alarms.
- **Top predictors:** tenure month, the upcoming annual renewal, plan type and recent workout levels, confirming the earlier analysis.
- **Business value:** the top 10% of members by risk score cancel at about 5 times the average rate and contain about half of all next-month cancellations.
- Time-based cross-validation and grid search tuning were tested; they did not improve test performance, so results are limited by the data available rather than model settings.

## Recommendations

1. **First 30 day onboarding goal** of 6 workouts, with reminders and a starter programme.
2. **Low activity alerts** when a member does 0 to 2 workouts in a month.
3. **Support ticket follow-up** within 48 hours, and fix the top Android issues.
4. **Promo exit path:** offer a step-down price before a discount ends.
5. **Renewal campaign** for annual members, starting 60 days before month 12.

The risk scores from the prediction model can decide who receives each action each month. Each action would be tested with an A/B test and measured with the monthly churn rate, compared within the same tenure month. The full plan, with targets, is in the final section of the notebook.

## Repository Structure

```
Peloton_churn_analysis/
├── datasets/
│   ├── subscriptions.csv
│   └── usage.csv
├── evaluation.ipynb            # full analysis and prediction models in Python
└── README.md
```

## How to Run

**Notebook**
1. Install the libraries: `pip install pandas matplotlib seaborn scikit-learn jupyter`
2. In the second cell, change the file paths to `datasets/subscriptions.csv` and `datasets/usage.csv`.
3. Open `evaluation.ipynb` and run all cells from the top.


## Author

Karan Thapa, Bachelor of Computer Science, Charles Darwin University