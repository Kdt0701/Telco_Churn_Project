# Dataset Schema — IBM Telco Customer Churn

## Source
IBM Cognos Analytics Telco Customer Churn sample. The IBM public data directory contains the extended source files for Demographics, Location, Population, Services and Status.

## Source tables used

| Table | Join key | Main role |
|---|---|---|
| Demographics | Customer ID | Customer profile and demographic attributes |
| Location | Customer ID | State, City, ZIP, Latitude, Longitude |
| Services | Customer ID | Tenure, services, contract, payment, charges |
| Status | Customer ID | Churn outcome, satisfaction, churn-related status |
| Population | Zip Code | ZIP-level population context |

## Internal standard
`Customer ID` is normalized to `CustomerID` before Join.

## Join design

```text
Demographics ── CustomerID ── Location
      │
      ├──── CustomerID ── Services
      │
      └──── CustomerID ── Status

Location ── Zip Code ── Population
```

The final analytical dataset is one row per customer. Population is joined defensively as a many-to-one ZIP lookup.

## Important raw fields

### Demographics
`Gender`, `Age`, `Senior Citizen`, `Married`, `Dependents`, `Number of Dependents`

### Location
`Country`, `State`, `City`, `Zip Code`, `Latitude`, `Longitude`

### Services
`Tenure in Months`, `Internet Service`, `Internet Type`, `Online Security`, `Online Backup`, `Device Protection Plan`, `Premium Tech Support`, `Streaming TV`, `Streaming Movies`, `Streaming Music`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charge`, `Total Charges`, `Total Revenue`

### Status
`Satisfaction Score`, `Customer Status`, `Churn Label`, `Churn Value`, `Churn Score`, `CLTV`, `Churn Category`, `Churn Reason`

## Derived fields

- `Tenure_Group`: `<1 year`, `1-3 years`, `>3 years`
- `CLV_Estimated = Monthly Charge × Tenure in Months`
- `Churn_Flag`: binary target derived from the source churn value/label
- `Monthly Charge_Capped`, `Total Charges_Capped`: IQR-capped versions for outlier treatment
- `Churn_Probability`: Logistic Regression output
- `Risk_Level`: Low `[0.00, 0.30)`, Medium `[0.30, 0.60)`, High `[0.60, 1.00]`

## Leakage policy
Do not use `Churn Score`, `Churn Category`, `Churn Reason`, `Churn Label`, `Churn Value`, or `Customer Status` as predictive features. `CLTV` is also excluded from the baseline model because it is a status-related/customer-value field rather than a clean pre-outcome input.
