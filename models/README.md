# Models Directory

Trained model artifacts are saved here after running `python -m src.train`.

## Expected Artifacts

After training:
- `best_model.joblib` — Best performing model (pipeline + model)
- `label_encoder.joblib` — Label encoder for target
- `feature_columns.joblib` — Feature column order
- `metrics.json` — Final evaluation metrics

## Loading

```python
import joblib
model = joblib.load("models/best_model.joblib")
```
