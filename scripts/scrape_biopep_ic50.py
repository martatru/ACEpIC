import os
import sys
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Paths configuration
BASE_DIR = "/home/marta/Pulpit/ACEpIC"
INPUT_CSV = os.path.join(BASE_DIR, "dataset/new_peptides_from_biopep.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset/new_peptides_with_ic50.csv")

def extract_ic50_from_html(html):
    """
    Parses the peptide page HTML and looks for the IC50 entry.
    Returns a tuple: (value, unit) e.g., ('115.70', 'µM')
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    for select in soup.find_all('select'):
        selected_option = select.find('option', selected=True)
        
        if selected_option and 'IC50' in selected_option.text:
            tr = select.find_parent('tr')
            if tr:
                inputs = tr.find_all('input', type='text')
                for inp in inputs:
                    val = inp.get('value', '').strip()
                    if val:
                        next_element = inp.next_sibling
                        unit = next_element.strip() if next_element and isinstance(next_element, str) else ""
                        return val, unit
                        
    return None, None

def main():
    print("Loading peptide list...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"File not found: {INPUT_CSV}")
        sys.exit(1)

    if 'ID' not in df.columns:
        print("Error: The CSV file is missing the 'ID' column.")
        sys.exit(1)

    # Initialize new columns for IC50 data
    df['IC50_value'] = None
    df['IC50_unit'] = None

    session = requests.Session()
    
    print("\nScraping IC50 data from BIOPEP-UWM...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
        peptide_id = row['ID']
        
        if pd.isna(peptide_id):
            continue
            
        peptide_id = int(peptide_id)
        url = f"https://biochemia.uwm.edu.pl/biopep/peptide_data_page1.php?zm_ID={peptide_id}"
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            ic50_val, ic50_unit = extract_ic50_from_html(response.text)
            
            if ic50_val:
                df.at[index, 'IC50_value'] = ic50_val
                df.at[index, 'IC50_unit'] = ic50_unit
                
        except Exception as e:
            tqdm.write(f"Error fetching ID {peptide_id}: {e}")
        
        # Polite delay to prevent hammering the server
        time.sleep(1)

    # Save the enriched dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved enriched dataset to:\n{OUTPUT_CSV}")
    
    found = df['IC50_value'].notna().sum()
    print(f"Summary: Successfully retrieved IC50 for {found} out of {len(df)} peptides.")

if __name__ == "__main__":
    main()