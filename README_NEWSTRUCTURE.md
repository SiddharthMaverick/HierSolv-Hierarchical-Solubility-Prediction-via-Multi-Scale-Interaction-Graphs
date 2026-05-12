# HierSolv: Hierarchical Solubility Prediction via Multi-Scale Interaction Graphs

A novel GNN framework for predicting molecular solubility across diverse solvents using hierarchical graph neural networks, charge-based graph merging, and evidential uncertainty quantification.

## ✨ Key Features

- **Hierarchical Architecture**: Two-level message passing for intra- and inter-molecular interactions
- **CSGM (Charge Surface Graph Merging)**: Adaptive multi-site anchor selection based on Gasteiger charges
- **Temperature Conditioning**: Sinusoidal embeddings + temperature-conditioned GRU
- **Evidential Uncertainty**: NIG-based epistemic and aleatoric uncertainty quantification
- **Physical Priors**: Van't Hoff law regularization for thermodynamic consistency

## 📦 Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/hiersolv.git
cd hiersolv

# Create environment
conda create -n hiersolv python=3.10 -y
conda activate hiersolv

# Install dependencies
pip install -e .

# Optional: Install development tools
pip install -e ".[dev]"

# Optional: Install W&B for experiment tracking
pip install -e ".[wandb]"
```

### GPU Support (Optional)

```bash
# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric with GPU support
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

## 🚀 Quick Start

### 1. Download & Prepare Data

```bash
python data/download_data.py --dataset bigsoldb --output data/bigsoldb.csv
```

### 2. Train the Model

```bash
python run_experiment.py --config configs/hiersolv_default.yaml --data data/bigsoldb.csv
```

### 3. Make Predictions

```bash
python predict.py --model results/hiersolv_best.pt \
                  --solute "CC(=O)Oc1ccccc1C(=O)O" \
                  --solvent "O" \
                  --temperature 298.15
```

## 📁 Project Structure

```
hiersolv/
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── csgm.py               # Charge Surface Graph Merging
│   ├── hiersolv.py           # Main HierSolv architecture
│   ├── evidential.py         # EDL uncertainty heads
│   └── baselines.py          # Baseline models (MolMerger, ConcatGCN, RF, MLP)
├── data/                      # Data loading and processing
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset class
│   ├── splits.py             # Data splitting strategies
│   └── download_data.py      # Dataset download utilities
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── featurizer.py         # Molecular featurization
│   ├── metrics.py            # Evaluation metrics
│   ├── trainer.py            # Training loop
│   ├── logger.py             # Experiment logging
│   └── plotting.py           # Visualization (optional)
├── experiments/               # Experiment runners
│   ├── __init__.py
│   ├── ablation.py           # Ablation study
│   ├── ood_eval.py           # Out-of-distribution evaluation
│   └── k_sensitivity.py      # K hyperparameter sweep
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_csgm.py
│   ├── test_model.py
│   └── test_metrics.py
├── configs/                   # Configuration files
│   ├── hiersolv_default.yaml
│   └── ablation.yaml
├── run_experiment.py         # Main training script
├── predict.py                # Inference script
├── setup.py                  # Package setup
├── README.md                 # This file
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
├── requirements.txt          # Dependencies
└── .gitignore               # Git ignore rules
```

## 🧬 Model Architecture

### HierSolv Overview

1. **Level-1 (Intra-molecular)**: Separate GATv2 layers for solute and solvent graphs
2. **Level-2 (Inter-molecular)**: CSGM bipartite edges connect chemically active anchor atoms
3. **Temperature Encoding**: Sinusoidal positional encoding + learned projection
4. **Fusion & Output**: Temperature-conditioned GRU + MLP → Evidential head

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden` | 256 | Hidden dimension |
| `heads` | 8 | Attention heads in GATv2 |
| `L1` | 3 | Intra-molecular layers |
| `L2` | 2 | Inter-molecular layers |
| `K_frac` | 0.30 | Anchor atom selection fraction |
| `tau` | 0.50 | Interaction edge pruning threshold |

## 📊 Training & Evaluation

### Training with Defaults

```bash
python run_experiment.py --config configs/hiersolv_default.yaml
```

### Custom Configuration

Edit `configs/hiersolv_default.yaml` to customize:
- Learning rate, batch size, epochs
- CSGM parameters (K_frac, tau)
- Model architecture (hidden dim, layers)
- Regularization (EDL weight, van't Hoff weight)

### Ablation Studies

```bash
python experiments/ablation.py --config configs/ablation.yaml
```

Ablations included:
- ✓ Full HierSolv
- − Multi-site CSGM (K=1 only)
- − Temperature conditioning
- − Hierarchical structure (single-level)
- − Evidential learning (MSE only)
- − Van't Hoff regularization
- − Residual connections

## 🔬 Experiments

### OOD Generalization

```bash
python experiments/ood_eval.py --model results/hiersolv_best.pt \
                               --ood-solvents "CC(C)O,CCO" \
                               --data data/bigsoldb.csv
```

### K Sensitivity Analysis

```bash
python experiments/k_sensitivity.py --data data/bigsoldb.csv
```

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_model.py -v

# With coverage
pytest tests/ --cov=hiersolv --cov-report=html
```

## 📈 Weights & Biases Integration

Enable experiment tracking with W&B:

```bash
python run_experiment.py --config configs/hiersolv_default.yaml \
                        --use-wandb \
                        --wandb-project hiersolv \
                        --wandb-entity your-entity
```

## 📝 Citation

If you use HierSolv in your research, please cite:

```bibtex
@article{hiersolv2024,
  title={HierSolv: Hierarchical Solubility Prediction via Multi-Scale Interaction Graphs},
  author={Your Names},
  journal={Journal of Chemical Information and Modeling},
  year={2024}
}
```

## 📚 References

- **GATv2**: Brody et al., "Graph Attention Networks", ICLR 2022
- **EDL**: Amini et al., "Deep Evidential Regression", NeurIPS 2020
- **MolMerger**: [Reference Paper], 2024
- **CSGM Concept**: Inspired by charge-aware molecular interactions

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Pull request process
- Bug report templates

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) file for details.

## 💬 Questions & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/hiersolv/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/hiersolv/discussions)
- **Email**: your-email@example.com

## 🙏 Acknowledgments

- Dataset providers: BigSolDB, ESOL, BNNLabs
- PyTorch Geometric team for GNN utilities
- RDKit for molecular representations

---

**Last Updated**: 2024-05-12  
**Maintainers**: Research Team
