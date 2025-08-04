#!/usr/bin/env python3
"""
Setup and Run Script for EGO-Moment-CLE-ViT

This script provides a complete workflow for getting started:
1. Check dependencies
2. Download dataset
3. Verify configuration
4. Run training
5. Evaluate results

Usage:
    python setup_and_run.py --dataset cotton80
    python setup_and_run.py --dataset cotton80 --epochs 5 --batch_size 16
"""

import argparse
import sys
import os
import yaml
from pathlib import Path
import subprocess
import time

def run_command(command, description="", capture_output=True):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"[*] {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
        else:
            result = subprocess.run(command, shell=True, check=True)
        
        print(f"[+] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] {description} failed with exit code {e.returncode}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n[INFO] Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision', 
        'timm': 'Torch Image Models',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'einops': 'einops',
        'numpy': 'NumPy'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[MISSING] {name}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[WARNING] Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    print("\n[OK] All dependencies are installed!")
    return True

def check_dataset_available(dataset_name):
    """Check if dataset is available for download"""
    print(f"\n[INFO] Checking dataset availability: {dataset_name}")
    command = "python download_simple.py --list"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Look for the dataset name in the output
            lines = result.stdout.split('\n')
            for line in lines:
                if f'[*] {dataset_name}' in line:
                    print(f"[OK] Dataset '{dataset_name}' is available")
                    return True
            
            print(f"[ERROR] Dataset '{dataset_name}' not found")
            print("Available datasets:")
            # Extract dataset names from output
            for line in lines:
                if '[*]' in line and line.strip():
                    dataset_line = line.strip()
                    if dataset_line.startswith('[*]'):
                        print(f"  - {dataset_line[3:].strip()}")
            return False
        else:
            print("[ERROR] Failed to check available datasets")
            if result.stderr:
                print(result.stderr)
            return False
    except subprocess.CalledProcessError as e:
        print("[ERROR] Failed to check available datasets")
        if e.stderr:
            print(e.stderr)
        return False

def download_dataset(dataset_name, root="./data"):
    """Download dataset using the CLI tool"""
    command = f"python download_simple.py --dataset {dataset_name} --root {root}"
    return run_command(command, f"Downloading dataset: {dataset_name}", capture_output=False)

def create_quick_config(dataset_name, epochs=10, batch_size=32, config_name="quick_start"):
    """Create a quick start configuration file"""
    print(f"\n📝 Creating configuration for {config_name}...")
    
    # Load base config
    base_config_path = "configs/ufg_base.yaml"
    if not os.path.exists(base_config_path):
        print(f"❌ Base config not found: {base_config_path}")
        return None
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for quick start
    config['dataset']['name'] = dataset_name
    config['training']['epochs'] = epochs
    config['training']['batch_size'] = batch_size
    config['training']['val_frequency'] = max(1, epochs // 5)  # Validate 5 times during training
    config['training']['save_frequency'] = max(1, epochs // 2)  # Save 2 checkpoints
    config['experiment']['name'] = f"{config_name}_{dataset_name}_{epochs}epochs"
    
    # Adjust for quick training
    if epochs <= 10:
        config['training']['val_frequency'] = 1  # Validate every epoch for short runs
        config['experiment']['log_frequency'] = 10  # More frequent logging
    
    # Save quick config
    quick_config_path = f"configs/{config_name}.yaml"
    with open(quick_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[+] Configuration saved: {quick_config_path}")
    print(f"  - Dataset: {dataset_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Validation frequency: {config['training']['val_frequency']}")
    
    return quick_config_path

def run_training(config_path):
    """Run training with the specified config"""
    command = f"python train.py --config {config_path}"
    return run_command(command, "Training EGO-Moment-CLE-ViT", capture_output=False)

def run_evaluation(config_path):
    """Run evaluation if training completed successfully"""
    # Look for the best model checkpoint
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        best_model_path = checkpoints_dir / "best_model.pth"
        if best_model_path.exists():
            command = f"python eval.py --config {config_path} --checkpoint {best_model_path}"
            return run_command(command, "Evaluating trained model", capture_output=False)
    
    print("⚠️  No trained model found, skipping evaluation")
    return False

def show_results():
    """Show training results and next steps"""
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    # Check for outputs
    outputs_dir = Path("outputs")
    logs_dir = Path("logs")
    checkpoints_dir = Path("checkpoints")
    
    if outputs_dir.exists():
        output_files = list(outputs_dir.glob("*"))
        print(f"📁 Output files ({len(output_files)}):")
        for file in output_files:
            print(f"  - {file}")
    
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        print(f"📝 Log files ({len(log_files)}):")
        for file in log_files:
            print(f"  - {file}")
    
    if checkpoints_dir.exists():
        checkpoint_files = list(checkpoints_dir.glob("*.pth"))
        print(f"💾 Checkpoint files ({len(checkpoint_files)}):")
        for file in checkpoint_files:
            print(f"  - {file}")

def main():
    parser = argparse.ArgumentParser(
        description="Setup and run EGO-Moment-CLE-ViT training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with cotton80 dataset
  python setup_and_run.py --dataset cotton80
  
  # Quick training (5 epochs, small batch)
  python setup_and_run.py --dataset cotton80 --epochs 5 --batch_size 16
  
  # Skip dataset download (if already downloaded)
  python setup_and_run.py --dataset cotton80 --skip-download
  
  # Only download dataset and exit
  python setup_and_run.py --dataset cotton80 --download-only
  
  # Full pipeline with evaluation
  python setup_and_run.py --dataset cotton80 --epochs 20 --with-evaluation
        """
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='cotton80',
        help='Dataset name to use (default: cotton80)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch_size', '-b',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--root', '-r',
        type=str,
        default='./data',
        help='Root directory for datasets (default: ./data)'
    )
    parser.add_argument(
        '--config_name',
        type=str,
        default='quick_start',
        help='Name for the generated config file (default: quick_start)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (assume already downloaded)'
    )
    parser.add_argument(
        '--download-only',
        action='store_true',
        help='Only download dataset and exit'
    )
    parser.add_argument(
        '--with-evaluation',
        action='store_true',
        help='Run evaluation after training'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check dependencies and exit'
    )
    
    args = parser.parse_args()
    
    print("[*] EGO-Moment-CLE-ViT Setup and Run")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Data Root: {args.root}")
    print(f"Config Name: {args.config_name}")
    
    start_time = time.time()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    if args.check_only:
        print("\n[+] Dependency check completed. Exiting as requested.")
        sys.exit(0)
    
    # Step 2: Check dataset availability
    if not check_dataset_available(args.dataset):
        print(f"\n❌ Dataset '{args.dataset}' is not available")
        sys.exit(1)
    
    # Step 3: Download dataset
    if not args.skip_download:
        if not download_dataset(args.dataset, args.root):
            print("\n❌ Dataset download failed")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping dataset download")
    
    if args.download_only:
        print("\n[+] Dataset download completed. Exiting as requested.")
        sys.exit(0)
    
    # Step 4: Create configuration
    config_path = create_quick_config(args.dataset, args.epochs, args.batch_size, args.config_name)
    if not config_path:
        print("\n❌ Failed to create configuration")
        sys.exit(1)
    
    # Step 5: Run training
    print(f"\n🎓 Starting training (this may take a while...)")
    if not run_training(config_path):
        print("\n❌ Training failed")
        sys.exit(1)
    
    # Step 6: Run evaluation (optional)
    if args.with_evaluation:
        print(f"\n📊 Running evaluation...")
        run_evaluation(config_path)
    else:
        print("\n⏭️  Skipping evaluation (use --with-evaluation to include)")
    
    # Step 7: Show results
    show_results()
    
    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("🎉 Setup and run completed successfully!")
    print("="*60)
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print("\nNext steps:")
    print("- Check results in: ./outputs/")
    print("- View logs in: ./logs/")
    print("- Find checkpoints in: ./checkpoints/")
    print(f"- Modify configs/{args.config_name}.yaml for custom experiments")
    print("- Use 'python eval.py' for detailed evaluation")

if __name__ == "__main__":
    main()
