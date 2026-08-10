import pandas as pd
import numpy as np

file_path = "/home/marta/Pulpit/ACEpIC/embeddings_data/cleaned_peptides.csv"
df = pd.read_csv(file_path)

col_name = "IC50 (µM)"

# Convert to numeric, forcing any text/errors to NaN
df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

# To maintain 1:1 alignment with already generated .npy embeddings, 
# DO NOT drop rows. Replace invalid values (<=0) with NaN.
df["pIC50"] = np.where(
    df[col_name] > 0,
    6.0 - np.log10(df[col_name]),
    np.nan
)

df.to_csv(file_path, index=False)
print(f"Done. Added 'pIC50'. Total rows: {len(df)} (Unchanged to match .npy matrices).")