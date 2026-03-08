# Data Dictionary

| Column | Type | Meaning | Allowed Values | Notes |
|---|---|---|---|---|
| feature_1 | float | Example numeric feature | Any real number | Check outliers |
| feature_2 | category | Example categorical feature | A, B, C | Verify category drift |
| target | int | Binary label | 0, 1 | Confirm no leakage |

## Leakage and Missingness Warnings

- Do not include post-outcome fields as model features.
- Track missingness patterns before and after preprocessing.
