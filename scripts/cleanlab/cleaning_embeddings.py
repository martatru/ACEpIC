import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from cleanlab.datalab.datalab import Datalab

# 1. Define paths
base_dir = "/home/marta/Pulpit/ACEpIC/embeddings_data"
csv_path = f"{base_dir}/cleaned_peptides.csv"
npy_path = f"{base_dir}/esm2_650m_embeddings.npy"
out_path = f"{base_dir}/cleanlab_issues_report.csv"

# 2. Load data
print("Loading data...")
df = pd.read_csv(csv_path)
X = np.load(npy_path)

# 3. Synchronize X and y (remove rows where pIC50 is NaN)
valid_mask = df["pIC50"].notna()
df_valid = df[valid_mask].reset_index(drop=True)
X_valid = X[valid_mask]
y_valid = df_valid["pIC50"].values

print(f"Valid samples for Cleanlab: {len(y_valid)} (Filtered out {len(df) - len(y_valid)} NaNs)")

# 4. Generate out-of-sample predictions using Ridge Regression
print("Training model and generating predictions (CV)...")
model = Ridge(alpha=1.0)
predictions = cross_val_predict(model, X_valid, y_valid, cv=5)

# 5. Run Cleanlab Datalab for regression
print("Running Cleanlab analysis...")
lab = Datalab(data={"pIC50": y_valid}, label_name="pIC50", task="regression")
lab.find_issues(features=X_valid, pred_probs=predictions)

# 6. Extract results and save
issues_df = lab.get_issues("label")
df_valid["is_label_issue"] = issues_df["is_label_issue"]
df_valid["label_quality_score"] = issues_df["label_score"]

# Sort by quality score (lowest score = most likely an error)
df_valid = df_valid.sort_values("label_quality_score")
df_valid.to_csv(out_path, index=False)

print(f"Done! Found {issues_df['is_label_issue'].sum()} potential label issues.")
print(f"Report saved to: {out_path}")