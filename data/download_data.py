"""
data/download_data.py
---------------------
Load and prepare datasets for HierSolv.

Supports:
    - BigSolDB (Kadivar et al., Nature Chem. 2023) - loaded from local BigSolDBv2.1.csv
    - ESOL (Delaney, J. Chem. Inf. Comput. Sci. 2004)
    - BNNLabs solubility dataset
    - PHYSPROP

Run with:
    python data/download_data.py --dataset bigsoldb --output data/bigsoldb.csv
"""

import argparse
import pandas as pd
import requests
from pathlib import Path


def load_bigsoldb_local(output_path: str = 'data/bigsoldb.csv'):
    """
    Load BigSolDB dataset from local BigSolDBv2.1.csv file.
    
    Maps columns:
        SMILES_Solute -> solute_smiles
        SMILES_Solvent -> solvent_smiles
        LogS(mol/L) -> logS
        Temperature_K -> temperature (convert to Celsius if needed)
        Solvent -> solvent_name
    """
    bigsoldb_path = Path('.') / 'BigSolDBv2.1.csv'
    densities_path = Path('.') / 'BigSolDBv2.1_densities.csv'
    
    if not bigsoldb_path.exists():
        print(f"⚠ BigSolDB file not found at {bigsoldb_path}")
        return None
    
    print(f"Loading BigSolDB from {bigsoldb_path}...")
    
    try:
        # Load main data
        df = pd.read_csv(bigsoldb_path)
        
        # Load densities (optional, for enrichment)
        if densities_path.exists():
            densities = pd.read_csv(densities_path)
            print(f"  Loaded density data for {len(densities)} (solvent, temperature) pairs")
        
        # Rename columns to standard format
        df_processed = pd.DataFrame({
            'solute_smiles': df['SMILES_Solute'],
            'solvent_smiles': df['SMILES_Solvent'],
            'logS': df['LogS(mol/L)'],
            'temperature': df['Temperature_K'],  # In Kelvin
            'solvent_name': df['Solvent'],
            'compound_name': df.get('Compound_Name', ''),
            'cas': df.get('CAS', ''),
            'pubchem_cid': df.get('PubChem_CID', ''),
            'fda_approved': df.get('FDA_Approved', ''),
            'source': df.get('Source', ''),
        })
        
        # Remove rows with missing SMILES or LogS
        df_processed = df_processed.dropna(subset=['solute_smiles', 'solvent_smiles', 'logS'])
        
        df_processed.to_csv(output_path, index=False)
        print(f"✓ Loaded BigSolDB: {len(df_processed):,} samples")
        print(f"  - Unique solvents: {df_processed['solvent_name'].nunique()}")
        print(f"  - Temperature range: {df_processed['temperature'].min():.1f} - {df_processed['temperature'].max():.1f} K")
        print(f"  - LogS range: {df_processed['logS'].min():.2f} - {df_processed['logS'].max():.2f}")
        print(f"  - Saved to: {output_path}")
        
        return df_processed
        
    except Exception as e:
        print(f"⚠ Error loading BigSolDB: {e}")
        return None


def download_esol(output_path: str = 'data/esol.csv'):
    """Download ESOL dataset (Delaney, 2004)."""
    print("Attempting to download ESOL...")
    
    url = "https://www.dropbox.com/s/bnjs7oaxch3f6o2/delaney.csv?dl=1"
    
    try:
        df = pd.read_csv(url)
        # Rename columns
        df = df.rename(columns={'SMILES': 'solute_smiles', 'Solubility': 'logS'})
        df['solvent_smiles'] = 'O'  # All water
        df['solvent_name'] = 'water'
        df['temperature'] = 298.15
        df = df[['solute_smiles', 'solvent_smiles', 'logS', 'temperature', 'solvent_name']]
        df.to_csv(output_path, index=False)
        print(f"✓ Downloaded ESOL: {len(df)} samples")
        return df
    except Exception as e:
        print(f"⚠ ESOL download failed: {e}")
        return None


def download_physprop(output_path: str = 'data/physprop.csv'):
    """
    Download PHYSPROP subset (EPA database).
    Note: PHYSPROP access is restricted; this is a placeholder.
    """
    print("PHYSPROP requires manual download from EPA website.")
    print("Visit: https://www.epa.gov/tsca-cbi-documents/physical-chemical-property-data")
    return None


def main():
    parser = argparse.ArgumentParser(description='Load HierSolv datasets')
    parser.add_argument('--dataset', type=str, default='bigsoldb',
                        choices=['bigsoldb', 'esol', 'physprop'],
                        help='Dataset to load/download')
    parser.add_argument('--output', type=str,
                        help='Output CSV path')
    args = parser.parse_args()

    # Create data directory
    Path('data').mkdir(exist_ok=True)

    if args.output is None:
        args.output = f'data/{args.dataset}.csv'

    if args.dataset == 'bigsoldb':
        df = load_bigsoldb_local(args.output)
    elif args.dataset == 'esol':
        df = download_esol(args.output)
    elif args.dataset == 'physprop':
        df = download_physprop(args.output)

    if df is not None:
        print(f"\nDataset summary:")
        print(f"  Samples: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Saved to: {args.output}")


if __name__ == '__main__':
    main()
