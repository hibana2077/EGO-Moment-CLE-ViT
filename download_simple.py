#!/usr/bin/env python3
"""
Simple Dataset Download CLI Tool for EGO-Moment-CLE-ViT (ASCII-safe)

This CLI tool allows users to download datasets used by the EGO-Moment-CLE-ViT model.
It provides an interface to download specific datasets to specified root directories,
maintaining compatibility with the main training pipeline.

Usage:
    python download_simple.py --dataset cotton80 --root ./data
    python download_simple.py --list  # List available datasets
    python download_simple.py --info cotton80  # Get dataset information
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.dataset.ufgvc import UFGVCDataset
except ImportError as e:
    print(f"Error importing UFGVCDataset: {e}")
    print("Please ensure you are running this script from the project root directory.")
    sys.exit(1)


class SimpleDatasetDownloader:
    """Simple Dataset Downloader with ASCII-only output"""
    
    def __init__(self):
        self.available_datasets = UFGVCDataset.DATASETS
    
    def list_datasets(self) -> None:
        """List all available datasets with descriptions"""
        print("\n" + "="*80)
        print("AVAILABLE DATASETS")
        print("="*80)
        
        for name, config in self.available_datasets.items():
            print(f"\n[*] {name}")
            print(f"    Description: {config['description']}")
            print(f"    Filename: {config['filename']}")
        
        print(f"\nTotal available datasets: {len(self.available_datasets)}")
        print("="*80)
    
    def get_dataset_info(self, dataset_name: str, root: str = "./data") -> None:
        """Get detailed information about a specific dataset"""
        if dataset_name not in self.available_datasets:
            print(f"[ERROR] Dataset '{dataset_name}' not found.")
            self._suggest_similar_datasets(dataset_name)
            return
        
        config = self.available_datasets[dataset_name]
        filepath = Path(root) / config['filename']
        
        print("\n" + "="*80)
        print(f"DATASET INFORMATION: {dataset_name}")
        print("="*80)
        
        print(f"\n[INFO] Basic Information:")
        print(f"   Name: {dataset_name}")
        print(f"   Description: {config['description']}")
        print(f"   Filename: {config['filename']}")
        
        print(f"\n[INFO] File Information:")
        print(f"   Root directory: {root}")
        print(f"   Full path: {filepath}")
        print(f"   File exists: {'YES' if filepath.exists() else 'NO'}")
        
        if filepath.exists():
            file_size = filepath.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            print(f"   File size: {file_size_mb:.2f} MB")
            
            # Try to get detailed dataset statistics
            try:
                print(f"\n[INFO] Dataset Statistics:")
                
                # Create a temporary dataset instance to get splits info
                temp_dataset = UFGVCDataset(
                    dataset_name=dataset_name,
                    root=root,
                    split='train',  # Use train split as default
                    download=False  # Don't re-download
                )
                
                info = temp_dataset.get_dataset_info()
                
                print(f"   Total samples: {info['total_samples']:,}")
                print(f"   Total classes: {info['total_classes']}")
                print(f"   Split distribution:")
                for split, count in info['split_distribution'].items():
                    print(f"     - {split}: {count:,} samples")
                
                if len(info['classes']) <= 20:
                    print(f"   Classes: {', '.join(info['classes'])}")
                else:
                    print(f"   Classes (first 10): {', '.join(info['classes'][:10])}...")
                
            except Exception as e:
                print(f"   [WARNING] Could not load dataset statistics: {e}")
        
        print("="*80)
    
    def download_dataset(
        self, 
        dataset_name: str, 
        root: str = "./data",
        force: bool = False,
        verify: bool = True
    ) -> bool:
        """Download a specific dataset"""
        if dataset_name not in self.available_datasets:
            print(f"[ERROR] Dataset '{dataset_name}' not found.")
            self._suggest_similar_datasets(dataset_name)
            return False
        
        config = self.available_datasets[dataset_name]
        root_path = Path(root)
        filepath = root_path / config['filename']
        
        print("\n" + "="*60)
        print(f"DOWNLOADING DATASET: {dataset_name}")
        print("="*60)
        
        print(f"\n[INFO] Download Information:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Description: {config['description']}")
        print(f"   Destination: {filepath}")
        
        # Check if file already exists
        if filepath.exists() and not force:
            print(f"\n[OK] Dataset already exists at: {filepath}")
            file_size = filepath.stat().st_size / (1024 * 1024)
            print(f"   File size: {file_size:.2f} MB")
            print("   Use --force to re-download")
            
            if verify:
                print("\n[INFO] Verifying dataset integrity...")
                if self._verify_dataset(dataset_name, root):
                    print("[OK] Dataset verification passed")
                    return True
                else:
                    print("[ERROR] Dataset verification failed, consider re-downloading with --force")
                    return False
            return True
        
        # Create directory if it doesn't exist
        root_path.mkdir(parents=True, exist_ok=True)
        
        # Download using UFGVCDataset
        try:
            print(f"\n[INFO] Starting download...")
            
            # Create dataset instance which will trigger download
            dataset = UFGVCDataset(
                dataset_name=dataset_name,
                root=root,
                split='train',  # Use train split for download trigger
                download=True
            )
            
            print(f"[OK] Download completed successfully!")
            
            # Verify dataset if requested
            if verify:
                print(f"\n[INFO] Verifying dataset integrity...")
                if self._verify_dataset(dataset_name, root):
                    print("[OK] Dataset verification passed")
                else:
                    print("[WARNING] Dataset verification failed, but download completed")
                    return False
            
            # Show dataset info
            print(f"\n[INFO] Dataset Information:")
            info = dataset.get_dataset_info()
            print(f"   Total samples: {info['total_samples']:,}")
            print(f"   Total classes: {info['total_classes']}")
            print(f"   Available splits: {list(info['split_distribution'].keys())}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
            
            # Clean up incomplete file
            if filepath.exists():
                try:
                    filepath.unlink()
                    print(f"[INFO] Cleaned up incomplete file")
                except:
                    pass
            
            return False
    
    def _verify_dataset(self, dataset_name: str, root: str) -> bool:
        """Verify dataset integrity by trying to load it"""
        try:
            # Try to create dataset instance and load some basic info
            dataset = UFGVCDataset(
                dataset_name=dataset_name,
                root=root,
                split='train',
                download=False
            )
            
            # Basic checks
            assert len(dataset) > 0, "Dataset is empty"
            assert len(dataset.classes) > 0, "No classes found"
            
            # Try to load first sample
            dataset[0]
            
            return True
            
        except Exception as e:
            print(f"   Verification error: {e}")
            return False
    
    def _suggest_similar_datasets(self, dataset_name: str) -> None:
        """Suggest similar dataset names if exact match not found"""
        suggestions = []
        for name in self.available_datasets.keys():
            if dataset_name.lower() in name.lower() or name.lower() in dataset_name.lower():
                suggestions.append(name)
        
        if suggestions:
            print(f"   Did you mean one of these? {', '.join(suggestions)}")
        else:
            print(f"   Use --list to see all available datasets")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Download datasets for EGO-Moment-CLE-ViT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python download_simple.py --list
  
  # Download cotton80 dataset to default directory (./data)
  python download_simple.py --dataset cotton80
  
  # Download to specific directory
  python download_simple.py --dataset cotton80 --root /path/to/data
  
  # Force re-download even if file exists
  python download_simple.py --dataset cotton80 --force
  
  # Get information about a dataset
  python download_simple.py --info cotton80
        """
    )
    
    # Main action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        '--dataset', '-d',
        type=str,
        help='Name of the dataset to download'
    )
    action_group.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available datasets'
    )
    action_group.add_argument(
        '--info', '-i',
        type=str,
        metavar='DATASET',
        help='Get detailed information about a specific dataset'
    )
    
    # Optional arguments
    parser.add_argument(
        '--root', '-r',
        type=str,
        default='./data',
        help='Root directory to save datasets (default: ./data)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if file already exists'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip dataset verification after download'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = SimpleDatasetDownloader()
    
    # Handle different actions
    try:
        if args.list:
            downloader.list_datasets()
            
        elif args.info:
            success = downloader.get_dataset_info(args.info, args.root)
            # Return error code if dataset not found
            if args.info not in downloader.available_datasets:
                sys.exit(1)
            
        elif args.dataset:
            success = downloader.download_dataset(
                args.dataset, 
                args.root, 
                args.force, 
                not args.no_verify
            )
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n\n[ERROR] Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
