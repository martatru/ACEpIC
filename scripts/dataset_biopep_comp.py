import os
import sys
import pandas as pd

# Paths
BASE_DIR = "/home/marta/Pulpit/ACEpIC"
EXP_DATASET = os.path.join(BASE_DIR, "dataset/experimental_dataset_plm4ace.ods")
BIOPEP_DATASET = os.path.join(BASE_DIR, "dataset/biopep_uwm.ods")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset/new_peptides_from_biopep.csv")

def main():
    print("Loading datasets (this might take a sec for .ods files)...")
    
    try:
        df_exp = pd.read_excel(EXP_DATASET, sheet_name=0, engine="odf")
        df_biopep = pd.read_excel(BIOPEP_DATASET, engine="odf")
    except Exception as e:
        print(f"Error loading files: {e}")
        print("Tip: make sure odfpy is installed (pip install odfpy).")
        sys.exit(1)

    # Strip whitespaces from column names just in case
    df_exp.columns = df_exp.columns.astype(str).str.strip()
    df_biopep.columns = df_biopep.columns.astype(str).str.strip()

    # Sanity check
    if 'sequence' not in df_exp.columns or 'Sequence' not in df_biopep.columns:
        print("Error: Missing required sequence columns in the datasets.")
        sys.exit(1)

    # Extract and normalize sequences (uppercase, no trailing whitespaces)
    exp_seqs = set(df_exp['sequence'].dropna().astype(str).str.strip().str.upper())
    biopep_seqs = set(df_biopep['Sequence'].dropna().astype(str).str.strip().str.upper())

    print(f"Unique experimental sequences: {len(exp_seqs)}")
    print(f"Unique BIOPEP sequences: {len(biopep_seqs)}")

    # Find the difference
    new_seqs = biopep_seqs - exp_seqs
    print(f"\nFound {len(new_seqs)} new peptides!")

    if not new_seqs:
        print("No new peptides found. The experimental dataset is up to date.")
        return

    # Filter original biopep dataset to extract full rows for new peptides
    # Adding a temp column for a safe match
    df_biopep['temp_seq'] = df_biopep['Sequence'].astype(str).str.strip().str.upper()
    df_new = df_biopep[df_biopep['temp_seq'].isin(new_seqs)].copy()
    
    # Cleanup temp column
    df_new.drop(columns=['temp_seq'], inplace=True)

    # CLEANUP: Remove all 'Unnamed' garbage columns
    df_new = df_new.loc[:, ~df_new.columns.str.contains('^Unnamed')]
    
    # CLEANUP: Drop columns that are completely empty (all NaN values)
    df_new = df_new.dropna(axis=1, how='all')

    # Save to csv
    df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved new, cleaned peptides to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()