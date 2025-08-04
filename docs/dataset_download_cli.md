# Dataset Download CLI Tool

This document provides information about the `download_dataset.py` CLI tool for downloading datasets used in the EGO-Moment-CLE-ViT project.

## Quick Start

```bash
# List available datasets
python download_dataset.py --list

# Download a specific dataset
python download_dataset.py --dataset cotton80

# Get dataset information
python download_dataset.py --info cotton80
```

## Features

- **Dataset Download**: Download individual or all datasets
- **Dataset Information**: Get detailed information about datasets
- **Dataset Listing**: List all available datasets with descriptions
- **Verification**: Automatic dataset integrity verification
- **Force Download**: Re-download existing datasets
- **Custom Root Directory**: Specify custom download location

## Available Commands

### List Datasets
```bash
python download_dataset.py --list
python download_dataset.py -l
```
Shows all available datasets with their descriptions.

### Download Dataset
```bash
python download_dataset.py --dataset DATASET_NAME [--root PATH] [--force] [--no-verify]
python download_dataset.py -d DATASET_NAME -r PATH
```

**Options:**
- `--dataset, -d`: Name of the dataset to download
- `--root, -r`: Root directory (default: `./data`)
- `--force, -f`: Force re-download even if file exists
- `--no-verify`: Skip verification after download

**Examples:**
```bash
# Download cotton80 to default directory
python download_dataset.py --dataset cotton80

# Download to specific directory
python download_dataset.py --dataset cotton80 --root /path/to/data

# Force re-download
python download_dataset.py --dataset cotton80 --force
```

### Get Dataset Information
```bash
python download_dataset.py --info DATASET_NAME [--root PATH]
python download_dataset.py -i DATASET_NAME
```

Shows detailed information about a dataset including:
- Basic information (name, description, URL)
- File information (path, size, existence)
- Dataset statistics (samples, classes, splits)

### Download All Datasets
```bash
python download_dataset.py --all [--root PATH] [--force]
python download_dataset.py -a
```

Downloads all available datasets sequentially.

## Available Datasets

The tool supports the following datasets from the UFGVC collection:

| Dataset Name | Description |
|--------------|-------------|
| `cotton80` | Cotton classification dataset with 80 classes |
| `soybean` | Soybean classification dataset |
| `soy_ageing_r1` | Soybean ageing dataset - Round 1 |
| `soy_ageing_r3` | Soybean ageing dataset - Round 3 |
| `soy_ageing_r4` | Soybean ageing dataset - Round 4 |
| `soy_ageing_r5` | Soybean ageing dataset - Round 5 |
| `soy_ageing_r6` | Soybean ageing dataset - Round 6 |

## Integration with Main Training Pipeline

The downloaded datasets are fully compatible with the main training pipeline:

```bash
# Download dataset
python download_dataset.py --dataset cotton80 --root ./data

# Use in training with config file
python train.py --config configs/ufg_base.yaml
```

The tool ensures that:
- Datasets are downloaded to the correct directory structure
- File formats match the expected parquet format
- Dataset verification ensures integrity
- Class mappings and splits are preserved

## Configuration Compatibility

The tool is designed to work seamlessly with the training configuration:

```yaml
# In your config file (e.g., configs/ufg_base.yaml)
dataset:
  name: "cotton80"  # Use any dataset name from the available list
  root: "./data"    # Must match the --root parameter used in download
  download: true    # Can be set to false after manual download
```

## Error Handling

The tool provides comprehensive error handling:

- **Dataset not found**: Suggests similar dataset names
- **Download failures**: Cleans up incomplete files
- **Verification failures**: Reports specific issues
- **Interrupted downloads**: Graceful cleanup on Ctrl+C

## Advanced Usage

### Batch Download Script
```bash
#!/bin/bash
# Download multiple specific datasets
datasets=("cotton80" "soybean" "soy_ageing_r1")
for dataset in "${datasets[@]}"; do
    python download_dataset.py --dataset "$dataset" --root ./data
done
```

### Verification Only
```bash
# Download without verification (faster)
python download_dataset.py --dataset cotton80 --no-verify

# Later verify manually
python download_dataset.py --info cotton80
```

### Custom Integration
```python
# Use in Python scripts
from download_dataset import DatasetDownloader

downloader = DatasetDownloader()
success = downloader.download_dataset("cotton80", "./data")
if success:
    print("Dataset ready for training!")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the project root directory
2. **Permission Error**: Ensure write permissions to the target directory
3. **Network Error**: Check internet connection for downloads
4. **Disk Space**: Ensure sufficient disk space for large datasets

### Debug Mode
```bash
# Run with detailed error information
python download_dataset.py --dataset cotton80 --quiet false
```

## Requirements

- Python 3.7+
- Dependencies from `requirements.txt`
- Internet connection for downloads
- Sufficient disk space (datasets range from 50MB to 500MB+)

## File Structure

After downloading, your directory structure will look like:
```
data/
├── cotton80_dataset.parquet
├── soybean_dataset.parquet
└── soy_ageing_R1_dataset.parquet
```

This structure is compatible with the main training pipeline and configuration files.
