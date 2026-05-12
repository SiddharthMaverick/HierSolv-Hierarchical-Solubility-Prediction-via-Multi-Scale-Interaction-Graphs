"""
data/download_data.py
---------------------
Download and prepare datasets for HierSolv.

Supports:
    - BigSolDB (Kadivar et al., Nature Chem. 2023)
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


def download_bigsoldb(output_path: str = 'data/bigsoldb.csv'):
    """
    Download BigSolDB dataset (Kadivar et al., 2023).
    
    Public access via zenodo or accompanying repository.
    Falls back to a minimal example if unavailable.
    """
    print("Attempting to download BigSolDB...")
    
    # Note: Replace with actual URL from the paper's data repository
    url = "https://zenodo.org/record/XXXXXXX/files/bigsoldb.csv"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            df = pd.read_csv(output_path)
            print(f"✓ Downloaded BigSolDB: {len(df)} samples")
            return df
    except Exception as e:
        print(f"⚠ Download failed: {e}")

    # Fallback: create minimal example dataset
    print("Creating minimal example dataset...")
    data = {
        'solute_smiles': [
            'CC(=O)Oc1ccccc1C(=O)O',  # Aspirin
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'O=C(O)Cc1ccccc1Nc2c(Cl)cccc2Cl',  # Diclofenac
        ],
        'solvent_smiles': [
            'O',  # Water
            'CC(C)O',  # Isopropanol
            'CCO',  # Ethanol
        ],
        'logS': [-2.5, -0.1, -1.8],
        'temperature': [298.15, 298.15, 298.15],
        'solvent_name': ['water', 'isopropanol', 'ethanol'],
    }
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Created example dataset: {len(df)} samples at {output_path}")
    return df


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
    parser = argparse.ArgumentParser(description='Download HierSolv datasets')
    parser.add_argument('--dataset', type=str, default='bigsoldb',
                        choices=['bigsoldb', 'esol', 'physprop'],
                        help='Dataset to download')
    parser.add_argument('--output', type=str,
                        help='Output CSV path')
    args = parser.parse_args()

    # Create data directory
    Path('data').mkdir(exist_ok=True)

    if args.output is None:
        args.output = f'data/{args.dataset}.csv'

    if args.dataset == 'bigsoldb':
        df = download_bigsoldb(args.output)
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
