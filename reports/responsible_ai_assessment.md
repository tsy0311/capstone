# Responsible AI Assessment

## LLM-Based Startup Risk Profiling and Funding Outcome Prediction

---

## 1. Data Provenance and Lawful Use

**Source**: Kaggle — Startup Success Prediction dataset (derived from Crunchbase public data)
**License**: Kaggle Open Data license
**Data type**: Company-level public business information
**Personal data**: None — no personal identifiable information (PII)

The dataset contains publicly available information about companies. No personal data about founders, employees, or individuals is included. Data is lawfully used for research and educational purposes.

## 2. Personal Data Minimisation

| Data Category | Present? | Action |
|---------------|----------|--------|
| Company name | Yes | Used as identifier, not feature |
| Founder names | No | N/A |
| Contact information | No | N/A |
| Financial details | Yes (company-level) | Used for modelling |
| Personal demographics | No | N/A |

**Assessment**: No personal data minimisation required — dataset contains only company-level information.

## 3. Access Control and Secure Storage

- Model artifacts saved in `models/` directory (`.gitignore` excludes from version control)
- No secrets, API keys, or credentials embedded in source code
- API endpoints do not expose internal model parameters beyond metadata
- Application runs locally — no cloud storage without explicit configuration

## 4. Fairness and Subgroup Analysis

### 4.1 Potential Bias Sources

| Bias Type | Description | Risk Level |
|-----------|-------------|------------|
| Survivorship bias | Crunchbase over-represents funded startups | High |
| Geographic bias | US-based companies dominate (~60%+) | Medium |
| Market bias | Software startups over-represented | Medium |
| Temporal bias | Older startups have more time to achieve exits | Medium |
| Size bias | Better-funded startups have more complete data | Low |

### 4.2 Subgroup Performance Analysis

Subgroup performance should be evaluated by:
- **Country**: ROC-AUC by country (US vs. non-US)
- **Market**: ROC-AUC by market category (Software, Biotech, Hardware)
- **Founding decade**: ROC-AUC by founding decade (2000s, 2010s, 2020s)

If subgroup performance varies by >10%, fairness mitigation strategies should be considered.

### 4.3 Proxy Discrimination

No protected attributes (race, gender, age) are present in the data. However, geographic location (country_code, state_code) may act as a proxy for demographic composition. This is monitored but not used for discrimination — it is used for contextual prediction.

## 5. Model Explainability and Transparency

- **Global**: SHAP summary plot shows feature importance across all predictions
- **Local**: SHAP waterfall explains individual predictions
- **API**: `/model/info` endpoint exposes model type, parameters, and pipeline structure
- **Dashboard**: Risk category labels and SHAP explanation table displayed for each prediction
- **Documentation**: Full methodology, feature engineering, and evaluation documented

## 6. False Positives and False Negatives

| Error Type | Consequence | Cost Estimate | Mitigation |
|------------|-------------|---------------|-----------|
| False Positive | Unnecessary due diligence on startup that would fail | $50,000 | Threshold optimization, human review |
| False Negative | Missed investment in successful startup | $5,000,000 | Lower threshold for high-value decisions |

**Threshold approach**: Cost-sensitive optimization minimises FP×$50K + FN×$5M. Users can adjust threshold via dashboard/API.

## 7. Data Drift and Model Monitoring

| Monitoring Aspect | Method | Alert Threshold |
|-------------------|--------|-----------------|
| Feature drift | Population Stability Index (PSI) | PSI > 0.2 |
| Label drift | Success/failure rate change | >5% change |
| Performance drift | ROC-AUC on new labelled data | >5% degradation |
| Concept drift | Feature-outcome relationship change | Quarterly evaluation |

**Retraining trigger**: PSI > 0.2 OR performance degradation >5%

## 8. PDPA Malaysia Compliance

### Personal Data Protection Act 2010 (PDPA) Assessment

| PDPA Principle | Applicability | Status |
|----------------|---------------|--------|
| **General Principle** (Section 6) | Personal data processed only with consent | N/A — no personal data |
| **Notice and Choice** (Sections 7-8) | Data subject notified of processing | N/A — no personal data |
| **Purpose Limitation** (Section 9) | Data used only for stated purpose | ✅ — research purpose stated |
| **Data Minimisation** (Section 10) | Collect only necessary data | ✅ — company-level only |
| **Data Quality** (Section 11) | Accurate, current, complete | ✅ — validation module |
| **Security Safeguards** (Section 12) | Reasonable security measures | ✅ — local storage, no secrets |
| **Retention Limitation** (Section 13) | Retain only as long as necessary | ✅ — data deleted post-research |
| **Access and Correction** (Sections 14-15) | Data subject access rights | N/A — no personal data |
| **Data Transfer** (Section 129) | Cross-border transfer restrictions | ✅ — public dataset |

**Recommendation**: A Data Protection Impact Assessment (DPIA) is recommended before production deployment, even though no personal data is processed. This ensures compliance with evolving regulatory requirements.

**Important note**: This assessment is for educational purposes and does not constitute legal advice. Consult a qualified legal professional for formal compliance verification.

## 9. Human Oversight and Limitations

- **Decision-support only**: Predictions inform, not replace, human judgement
- **Threshold transparency**: Users can adjust decision threshold based on risk appetite
- **Explanation availability**: SHAP explanations provided for every prediction
- **Limitation disclosure**: Application clearly states predictions are probabilistic, not certain
- **No automated decisions**: System does not make funding decisions — it provides information to support human decisions

## 10. Risks of Unfair Treatment

| Risk | Mitigation |
|------|-----------|
| Geographic bias in predictions | Subgroup analysis, transparency about limitations |
| Market category bias | Subgroup analysis by market |
| Temporal bias (recent startups harder to predict) | Transparency about prediction uncertainty |
| Self-fulfilling prophecy (model reinforces existing biases) | Human oversight, periodic bias audit |

## 11. Security

- No secrets in source code
- API input validation via Pydantic
- Error handling prevents information leakage
- Model artifacts not committed to version control
- Application runs locally by default
- No external API dependencies for core functionality

## 12. Limitations of This Assessment

1. This is a self-assessment, not an external audit
2. PDPA compliance analysis is educational, not legal advice
3. Subgroup performance analysis requires execution against actual data
4. Fairness metrics depend on the specific dataset distribution
5. Monitoring strategy is proposed but not yet implemented
