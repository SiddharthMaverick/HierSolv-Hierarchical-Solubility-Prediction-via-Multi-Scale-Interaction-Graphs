"""
predict.py
-----------
Inference script for HierSolv on new solute-solvent pairs.

Usage:
    python predict.py --model checkpoints/hiersolv_best.pt \
                      --solute "CC(=O)Oc1ccccc1C(=O)O" \
                      --solvent "O" \
                      --temperature 298.15

Outputs:
    - Point estimate (mean)
    - Uncertainty (std dev of posterior)
    - Aleatoric and epistemic uncertainty components
"""

import argparse
import torch
from pathlib import Path

from models import build_csgm, HierSolv


def load_model(checkpoint_path: str, device: torch.device) -> HierSolv:
    """Load a trained HierSolv model from checkpoint."""
    model = HierSolv()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: HierSolv, solute_smiles: str, solvent_smiles: str,
            temperature: float = 298.15, device: torch.device = None):
    """
    Predict LogS for a solute-solvent pair.

    Args:
        model: trained HierSolv model
        solute_smiles: canonical SMILES of solute
        solvent_smiles: canonical SMILES of solvent
        temperature: temperature in Kelvin
        device: torch device

    Returns:
        dict with keys:
            - pred_logS: point estimate
            - epistemic_unc: epistemic uncertainty
            - aleatoric_unc: aleatoric uncertainty
            - total_unc: total uncertainty (std dev)
    """
    if device is None:
        device = next(model.parameters()).device

    # Build CSGM
    triplet = build_csgm(
        solute_smiles, solvent_smiles,
        logS=0.0,  # placeholder
        temperature=temperature,
    )
    
    if triplet is None:
        return {'error': 'Invalid SMILES'}

    # Convert to batch
    batch = {
        'node_feats_u': triplet.solute.node_feats.unsqueeze(0),
        'edge_index_u': triplet.solute.edge_index,
        'edge_feats_u': triplet.solute.edge_feats.unsqueeze(0),
        'batch_u': torch.tensor([0] * triplet.solute.n_atoms),
        'n_atoms_u': [triplet.solute.n_atoms],
        'node_feats_v': triplet.solvent.node_feats.unsqueeze(0),
        'edge_index_v': triplet.solvent.edge_index,
        'edge_feats_v': triplet.solvent.edge_feats.unsqueeze(0),
        'batch_v': torch.tensor([0] * triplet.solvent.n_atoms),
        'inter_edge_index': triplet.inter_edge_index,
        'inter_edge_weights': triplet.inter_edge_weights.unsqueeze(0),
        'temperature': torch.tensor([temperature]),
    }

    # Move to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # Forward pass
    with torch.no_grad():
        output = model(
            batch['node_feats_u'],
            batch['edge_index_u'],
            batch['edge_feats_u'],
            batch['batch_u'],
            batch['n_atoms_u'],
            batch['node_feats_v'],
            batch['edge_index_v'],
            batch['edge_feats_v'],
            batch['batch_v'],
            batch['inter_edge_index'],
            batch['inter_edge_weights'],
            batch['temperature'],
        )

    if isinstance(output, tuple):  # EDL output
        gamma, nu, alpha, beta = output
        from models.evidential import decompose_uncertainty
        aleatoric, epistemic = decompose_uncertainty(nu, alpha, beta)
        total_unc = torch.sqrt(aleatoric + epistemic)
        
        return {
            'pred_logS': gamma.item(),
            'epistemic_unc': epistemic.item(),
            'aleatoric_unc': aleatoric.item(),
            'total_unc': total_unc.item(),
        }
    else:  # Point estimate
        return {
            'pred_logS': output.item(),
            'epistemic_unc': 0.0,
            'aleatoric_unc': 0.0,
            'total_unc': 0.0,
        }


def main():
    parser = argparse.ArgumentParser(description='Predict LogS with HierSolv')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--solute', type=str, required=True,
                        help='Solute SMILES')
    parser.add_argument('--solvent', type=str, required=True,
                        help='Solvent SMILES')
    parser.add_argument('--temperature', type=float, default=298.15,
                        help='Temperature in Kelvin')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    args = parser.parse_args()

    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    
    # Load model
    model = load_model(args.model, device)
    
    # Predict
    result = predict(model, args.solute, args.solvent, args.temperature, device)
    
    # Print results
    print("\n" + "="*60)
    print("HierSolv Prediction Results")
    print("="*60)
    print(f"Solute (SMILES):     {args.solute}")
    print(f"Solvent (SMILES):    {args.solvent}")
    print(f"Temperature (K):     {args.temperature:.2f}")
    print("-"*60)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Predicted LogS:      {result['pred_logS']:.4f}")
        print(f"Epistemic Unc:       {result['epistemic_unc']:.4f}")
        print(f"Aleatoric Unc:       {result['aleatoric_unc']:.4f}")
        print(f"Total Unc (1σ):      {result['total_unc']:.4f}")
        print(f"95% Confidence Int:  [{result['pred_logS'] - 1.96*result['total_unc']:.4f}, "
              f"{result['pred_logS'] + 1.96*result['total_unc']:.4f}]")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
