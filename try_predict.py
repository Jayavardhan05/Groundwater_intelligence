import pandas as pd
from ml.src.inference import GroundwaterLevelPredictor

p = GroundwaterLevelPredictor("ml/artifacts")
df = pd.read_csv("ml/data/processed/groundwater_cleaned_all_records.csv", parse_dates=["Date"])
history = df.groupby("Well ID").tail(p.required_history_steps + 5)

out = p.predict_next(history)
pd.DataFrame(out["predictions"]).to_csv("ml/artifacts/forecasts.csv", index=False)
print(f"{len(out['predictions'])} wells forecast, {len(out['skipped_wells'])} skipped")
print(pd.DataFrame(out["predictions"]).head(10))