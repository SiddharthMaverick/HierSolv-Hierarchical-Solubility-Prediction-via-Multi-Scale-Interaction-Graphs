#!/usr/bin/env python
"""Quick analysis of BigSolDB dataset"""
import pandas as pd
from pathlib import Path

print("=" * 70)
print("BIGSOLDB ANALYSIS")
print("=" * 70)

# Load main data
df = pd.read_csv('BigSolDBv2.1.csv')
print(f"\n📊 MAIN DATASET (BigSolDBv2.1.csv)")
print(f"   Total rows: {len(df):,}")
print(f"   Columns: {list(df.columns)}")

# Statistics
print(f"\n🧪 SOLVENT COVERAGE")
print(f"   Unique solvents: {df['Solvent'].nunique()}")
print(f"   Top solvents:")
for solv, count in df['Solvent'].value_counts().head(10).items():
    print(f"     - {solv}: {count:,} samples")

print(f"\n🌡️ TEMPERATURE RANGE")
print(f"   Min: {df['Temperature_K'].min():.1f} K ({df['Temperature_K'].min()-273.15:.1f}°C)")
print(f"   Max: {df['Temperature_K'].max():.1f} K ({df['Temperature_K'].max()-273.15:.1f}°C)")
print(f"   Unique temps: {df['Temperature_K'].nunique()}")

print(f"\n📈 SOLUBILITY (LogS mol/L) DISTRIBUTION")
print(f"   Mean: {df['LogS(mol/L)'].mean():.2f}")
print(f"   Median: {df['LogS(mol/L)'].median():.2f}")
print(f"   Std: {df['LogS(mol/L)'].std():.2f}")
print(f"   Min: {df['LogS(mol/L)'].min():.2f}")
print(f"   Max: {df['LogS(mol/L)'].max():.2f}")

print(f"\n🧬 COMPOUND STATISTICS")
print(f"   Unique solutes: {df['SMILES_Solute'].nunique():,}")
print(f"   FDA approved: {(df['FDA_Approved']=='Yes').sum():,}")
print(f"   Missing values: {df.isnull().sum().sum()}")

print(f"\n💾 DENSITIES FILE")
if Path('BigSolDBv2.1_densities.csv').exists():
    dens = pd.read_csv('BigSolDBv2.1_densities.csv')
    print(f"   Rows: {len(dens):,}")
    print(f"   Solvents: {dens['Solvent'].nunique()}")
    print(f"   Temp range: {dens['Temperature_K'].min():.1f} - {dens['Temperature_K'].max():.1f} K")
else:
    print(f"   Not found")

print("\n" + "=" * 70)
print("✅ Ready for training with data/bigsoldb.csv")
print("=" * 70)
