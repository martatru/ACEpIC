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
OUTPUT_CSV = os.path.join(BASE_DIR, "dataset/new_peptides_with_fixed_ic50.csv")

def extract_ic50_from_html(html):
    """
    Parses the peptide page HTML and looks specifically for the IC50 entry,
    avoiding Monoisotopic Mass and strictly filtering out EC50.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # 1. Find the dropdown that selects between IC50 / EC50
    iorelist = soup.find('select', {'name': 'iorelist'})
    if not iorelist:
        return None, None
        
    # 2. Check if IC50 is the currently selected option
    selected_option = iorelist.find('option', selected=True)
    if not selected_option or 'IC50' not in selected_option.text:
        # If it's EC50 or anything else, we return None
        return None, None

    # 3. If it IS IC50, extract the value. 
    # Note: BIOPEP developers named the input field 'txt_ec50' for both!
    val_input = soup.find('input', {'name': 'txt_ec50'})
    
    if val_input:
        val = val_input.get('value', '').strip()
        
        # 4. Extract the unit from the span exactly next to the input
        unit_span = val_input.find_next_sibling('span')
        unit = unit_span.text.strip() if unit_span else ""
        
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

    # Initialize / Reset columns for fixed IC50 data
    df['IC50_value'] = None
    df['IC50_unit'] = None

    session = requests.Session()
    
    print("\nFixing and scraping IC50 data from BIOPEP-UWM...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
        peptide_id = row['ID']
        
        if pd.isna(peptide_id):
            continue
            
        peptide_id = int(peptide_id)
        url = f"https://biochemia.uwm.edu.pl/biopep/peptide_data_page1.php?zm_ID={peptide_id}"
        
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            
            # Pass the HTML to our fixed logic
            ic50_val, ic50_unit = extract_ic50_from_html(response.text)
            
            if ic50_val:
                df.at[index, 'IC50_value'] = ic50_val
                df.at[index, 'IC50_unit'] = ic50_unit
                
        except Exception as e:
            tqdm.write(f"Error fetching ID {peptide_id}: {e}")
        
        # Polite delay to prevent hammering the server
        time.sleep(1)

    # Save the strictly verified dataset
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved the FIXED dataset to:\n{OUTPUT_CSV}")
    
    found = df['IC50_value'].notna().sum()
    print(f"Summary: Successfully retrieved strict IC50 for {found} out of {len(df)} peptides.")

if __name__ == "__main__":
    main()