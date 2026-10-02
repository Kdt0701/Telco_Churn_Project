# Dataset Schema — working design

## Source tables
IBM's extended Telco customer churn sample is organized into five tables: Demographics, Location, Population, Services and Status.

## Planned analytical table
`customer_churn` will be the merged customer-level table used by EDA and modeling.

## Join key
Use the customer identifier field (`Customer ID` / equivalent source spelling), standardized internally to `CustomerID`.

## Planned role of each table
| Source table | Main analytical role |
|---|---|
| Demographics | Gender, age/senior status, partner/dependents |
| Services | Contract, Internet Service, add-on services, payment and charges |
| Status | Churn label/value, satisfaction, CLTV and churn-related status |
| Location | State, City, ZIP, Latitude, Longitude |
| Population | ZIP-level population context |

## Model target
`Churn_Flag` — 1 for churned, 0 for non-churned.

## Model candidates
Use customer/service variables known before the churn outcome. Exclude identifiers and leakage-prone outcome-derived fields.
