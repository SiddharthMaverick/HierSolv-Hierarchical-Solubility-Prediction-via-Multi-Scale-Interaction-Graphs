#!/usr/bin/env python
"""
verify_project_structure.py
---------------------------
Verify that all required files and directories exist for a complete HierSolv project.

Run with: python verify_project_structure.py
"""

import os
from pathlib import Path


def verify_structure():
    """Verify project structure."""
    base_dir = Path('.')
    
    required_dirs = [
        'models',
        'data',
        'utils',
        'experiments',
        'tests',
        'configs',
        'results',
    ]
    
    required_files = {
        'root': [
            'README.md',
            'README_NEWSTRUCTURE.md',
            'setup.py',
            'LICENSE',
            'CONTRIBUTING.md',
            '.gitignore',
            'requirements.txt',
            'run_experiment.py',
            'predict.py',
        ],
        'models': [
            '__init__.py',
            'csgm.py',
            'hiersolv.py',
            'evidential.py',
            'baselines.py',
        ],
        'data': [
            '__init__.py',
            'dataset.py',
            'splits.py',
            'download_data.py',
        ],
        'utils': [
            '__init__.py',
            'featurizer.py',
            'metrics.py',
            'trainer.py',
            'logger.py',
            'plotting.py',
        ],
        'experiments': [
            '__init__.py',
            'ablation.py',
            'ood_eval.py',
            'k_sensitivity.py',
        ],
        'tests': [
            '__init__.py',
            'test_csgm.py',
            'test_model.py',
            'test_metrics.py',
        ],
        'configs': [
            'hiersolv_default.yaml',
            'ablation.yaml',
        ],
    }
    
    print("🔍 Verifying HierSolv Project Structure\n")
    print("="*60)
    
    # Check directories
    print("\n📁 DIRECTORIES:")
    missing_dirs = []
    for d in required_dirs:
        path = base_dir / d
        if path.is_dir():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ (MISSING)")
            missing_dirs.append(d)
    
    # Check files
    all_ok = True
    for location, files in required_files.items():
        if location == 'root':
            check_dir = base_dir
            print(f"\n📄 ROOT FILES:")
        else:
            check_dir = base_dir / location
            print(f"\n📄 {location.upper()}/ FILES:")
        
        for f in files:
            path = check_dir / f
            if path.is_file():
                size = path.stat().st_size
                print(f"  ✓ {f:<30} ({size:>8,} bytes)")
            else:
                print(f"  ✗ {f:<30} (MISSING)")
                all_ok = False
    
    # Summary
    print("\n" + "="*60)
    if all_ok and not missing_dirs:
        print("✅ Project structure is COMPLETE and ready!\n")
        return True
    else:
        print("⚠️  Project structure is INCOMPLETE.\n")
        if missing_dirs:
            print(f"Missing directories: {', '.join(missing_dirs)}")
        print("\nTo complete the project, run:")
        print("  python data/download_data.py")
        print("  python run_experiment.py --config configs/hiersolv_default.yaml")
        return False


def show_next_steps():
    """Show next steps for using the project."""
    print("\n🚀 NEXT STEPS:")
    print("-" * 60)
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Download data:")
    print("   python data/download_data.py --dataset bigsoldb")
    print()
    print("3. Run tests:")
    print("   pytest tests/ -v")
    print()
    print("4. Train the model:")
    print("   python run_experiment.py --config configs/hiersolv_default.yaml")
    print()
    print("5. Make predictions:")
    print("   python predict.py --model results/hiersolv_best.pt \\")
    print("                     --solute 'CC(=O)O' --solvent 'O'")
    print()
    print("📖 For details, see README_NEWSTRUCTURE.md")
    print()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if verify_structure():
        show_next_steps()
