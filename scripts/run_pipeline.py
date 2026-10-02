from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import build_customer_churn_dataset, save_processed_dataset
from src.model import train_model, save_model

raw_dir = ROOT / "data" / "raw"
processed = ROOT / "data" / "processed"

df = build_customer_churn_dataset(raw_dir)
save_processed_dataset(df, processed / "final_dataset.csv")
model, scored, metrics = train_model(df)
scored.to_csv(processed / "model_scored.csv", index=False, encoding="utf-8-sig")
save_model(model, ROOT / "models" / "logistic_regression.joblib")

print(f"Customers: {len(scored):,}")
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
print("Outputs:")
print(processed / "final_dataset.csv")
print(processed / "model_scored.csv")
print(ROOT / "models" / "logistic_regression.joblib")
