# Dataset Documentation

## Source

**Primary**: Kaggle - Startup Success Prediction
URL: https://www.kaggle.com/datasets/manishkc06/startup-success-prediction

**Alternative**: Kaggle - Startup Investments (Crunchbase)
URL: https://www.kaggle.com/datasets/arindamkhandai/startup-investments-crunchbase

## Description

Public business information on startups including:
- Company name, description, category/market
- Funding total (USD), number of funding rounds
- Founding date (year, month)
- First and last funding dates
- Country, state, region, city
- Status (acquired, IPO, closed, operating)

## Target Variable

Binary classification derived from `status`:
- **1 (Success)**: acquired or IPO
- **0 (Failure)**: closed (ceased operations)

Startups with status "operating" are excluded from the binary
classification task as their outcome is unresolved.

## Licensing

Kaggle Open Data license. Derived from Crunchbase public data.
No personal data. Company-level information only.

## Privacy

No personal identifiable information (PII). Company-level data only.
Suitable for research without IRB approval.

## Known Limitations

1. Survivorship bias — Crunchbase over-represents funded startups
2. Date censorship — some founded/first-funding dates are approximate
3. Geographical bias — US-based companies are over-represented
4. Temporal drift — startup dynamics change over time
5. No causal claims — observational data, correlations only
