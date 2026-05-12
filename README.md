# HierSolv: Hierarchical Solubility Prediction via Multi-Scale Interaction Graphs

A novel GNN framework for predicting molecular solubility across diverse solvents.
Advances beyond MolMerger with: multi-site charge interaction graphs (CSGM),
hierarchical two-level message passing (HierGAT), temperature-conditioned GRU (TC-GRU),
and evidential uncertainty quantification (EDL).

## Setup

```bash
# 1. Create environment
conda create -n hiersolv python=3.10 -y
conda activate hiersolv

# 2. Install PyTorch (CPU version shown; use CUDA variant if available)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu

# 3. Install PyTorch Geometric
pip install torch-geometric==2.4.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 4. Install remaining dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Download and prepare data
python data/download_data.py

# Run full pipeline (train + evaluate + generate figures)
python run_experiment.py --config configs/hiersolv_default.yaml

# Ablation study
python experiments/ablation.py --config configs/ablation.yaml

# Predict for new SMILES pairs
python predict.py --solute "CC(=O)Oc1ccccc1C(=O)O" --solvent "O" --temperature 298.15
```

## Project Structure

```
hiersolv/
├── data/
│   ├── download_data.py        # fetch BigSolDB, ESOL, BNNLabs
│   ├── dataset.py              # PyTorch Dataset class
│   └── splits.py               # scaffold + random + stratified splits
├── models/
│   ├── csgm.py                 # Charge Surface Graph Merging
│   ├── hiersolv.py             # Full model architecture
│   ├── baselines.py            # MolMerger, ConcatGCN, RF baselines
│   └── evidential.py           # NIG loss + uncertainty heads
├── utils/
│   ├── featurizer.py           # atom/bond feature vectors
│   ├── metrics.py              # MAE, RMSE, R2, ECE, calibration
│   ├── plotting.py             # all publication figures
│   └── logger.py               # experiment logging + W&B
├── experiments/
│   ├── ablation.py             # component ablation runner
│   ├── ood_eval.py             # out-of-distribution evaluation
│   └── k_sensitivity.py        # CSGM K hyperparameter sweep
├── tests/
│   ├── test_csgm.py
│   ├── test_model.py
│   └── test_metrics.py
├── configs/
│   ├── hiersolv_default.yaml
│   └── ablation.yaml
├── run_experiment.py           # main entry point
├── predict.py                  # inference on new pairs
└── requirements.txt
```

## Citation

```bibtex
@article{hiersolv2024,
  title={HierSolv: Hierarchical Solubility Prediction via Multi-Scale Interaction Graphs},
  author={},
  journal={J. Chem. Inf. Model.},
  year={2024}
}
```
