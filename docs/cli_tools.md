# CLI Tools for EGO-Moment-CLE-ViT

This document describes the command-line interface (CLI) tools available for the EGO-Moment-CLE-ViT project.

## Available Tools

### 1. `download_simple.py` - Simple Dataset Downloader

A lightweight CLI tool for downloading datasets with ASCII-only output (Windows-compatible).

#### Usage

```bash
# List all available datasets
python download_simple.py --list

# Download a specific dataset
python download_simple.py --dataset cotton80

# Download to custom directory
python download_simple.py --dataset cotton80 --root /path/to/data

# Get dataset information
python download_simple.py --info cotton80

# Force re-download
python download_simple.py --dataset cotton80 --force
```

#### Available Datasets

- `cotton80` - Cotton classification dataset with 80 classes
- `soybean` - Soybean classification dataset  
- `soy_ageing_r1` - Soybean ageing dataset - Round 1
- `soy_ageing_r3` - Soybean ageing dataset - Round 3
- `soy_ageing_r4` - Soybean ageing dataset - Round 4
- `soy_ageing_r5` - Soybean ageing dataset - Round 5
- `soy_ageing_r6` - Soybean ageing dataset - Round 6

### 2. `setup_and_run.py` - Complete Training Pipeline

An all-in-one script that handles the complete workflow from setup to training.

#### Features

- **Dependency Checking**: Verifies all required packages are installed
- **Dataset Management**: Downloads and verifies datasets
- **Configuration Generation**: Creates optimized configs for quick starts
- **Training Execution**: Runs training with proper logging
- **Result Summary**: Shows training outcomes and next steps

#### Usage

```bash
# Quick start with default settings
python setup_and_run.py --dataset cotton80

# Quick training (5 epochs, small batch)
python setup_and_run.py --dataset cotton80 --epochs 5 --batch_size 16

# Full pipeline with evaluation
python setup_and_run.py --dataset cotton80 --epochs 20 --with-evaluation

# Check dependencies only
python setup_and_run.py --check-only

# Download dataset only
python setup_and_run.py --dataset cotton80 --download-only

# Skip download (if already downloaded)
python setup_and_run.py --dataset cotton80 --skip-download
```

#### Options

- `--dataset, -d`: Dataset name to use (default: cotton80)
- `--epochs, -e`: Number of training epochs (default: 10)
- `--batch_size, -b`: Batch size for training (default: 32)
- `--root, -r`: Root directory for datasets (default: ./data)
- `--config_name`: Name for generated config file (default: quick_start)
- `--skip-download`: Skip dataset download
- `--download-only`: Only download dataset and exit
- `--with-evaluation`: Run evaluation after training
- `--check-only`: Only check dependencies and exit

### 3. `quick_start.py` - Demo and Architecture Overview

A simple demonstration script that shows model architecture and runs basic tests.

#### Usage

```bash
# Run quick demo
python quick_start.py --demo

# Show architecture overview
python quick_start.py --arch

# Check requirements
python quick_start.py --check
```

## Workflow Examples

### Complete Setup and Training

```bash
# 1. Check everything is ready
python setup_and_run.py --check-only

# 2. Download dataset and run quick training
python setup_and_run.py --dataset cotton80 --epochs 5 --batch_size 16

# 3. Run longer training with evaluation
python setup_and_run.py --dataset cotton80 --epochs 50 --with-evaluation
```

### Manual Dataset Management

```bash
# 1. List available datasets
python download_simple.py --list

# 2. Get detailed dataset information
python download_simple.py --info cotton80

# 3. Download specific dataset
python download_simple.py --dataset cotton80 --root ./my_data

# 4. Use in training
python train.py --config configs/ufg_base.yaml
```

### Custom Configuration

```bash
# 1. Generate custom config
python setup_and_run.py --dataset cotton80 --epochs 20 --batch_size 64 --config_name my_experiment --download-only

# 2. Modify the generated config file
# Edit configs/my_experiment.yaml

# 3. Run training with custom config
python train.py --config configs/my_experiment.yaml
```

## File Organization

After using these tools, your project structure will look like:

```
EGO-Moment-CLE-ViT/
├── data/                           # Downloaded datasets
│   ├── cotton80_dataset.parquet
│   └── soybean_dataset.parquet
├── configs/                        # Configuration files
│   ├── ufg_base.yaml              # Base configuration
│   └── quick_start.yaml           # Generated quick start config
├── checkpoints/                    # Saved model checkpoints
│   ├── best_model.pth
│   └── checkpoint_epoch_10.pth
├── logs/                          # Training logs
│   └── quickstart_cotton80_10epochs.log
├── outputs/                       # Training outputs
│   └── training_curves.png
└── ...
```

## Configuration Integration

The CLI tools are designed to work seamlessly with the main training pipeline:

### Generated Configuration

When you use `setup_and_run.py`, it automatically generates a configuration file that:

- Sets the correct dataset name and path
- Optimizes training parameters for quick experiments
- Adjusts logging and validation frequency based on epoch count
- Uses meaningful experiment names

### Base Configuration Compatibility

All generated configs inherit from `configs/ufg_base.yaml` and only override necessary parameters, ensuring:

- Model architecture remains consistent
- Advanced features (GPF, moment pooling) are preserved
- Training stability is maintained

## Error Handling and Troubleshooting

### Common Issues

1. **Unicode Encoding Errors**: Use `download_simple.py` instead of `download_dataset.py` on Windows
2. **Import Errors**: Ensure you're running from the project root directory
3. **Permission Errors**: Check write permissions for data, logs, and checkpoint directories
4. **Memory Errors**: Reduce batch size using `--batch_size` parameter

### Debug Mode

For detailed error information:

```bash
# Run with Python's verbose mode
python -v setup_and_run.py --dataset cotton80

# Check individual components
python download_simple.py --info cotton80
python quick_start.py --check
```

## Next Steps

After successfully using these CLI tools:

1. **Explore Results**: Check generated logs and training curves
2. **Customize Training**: Modify generated config files for your needs
3. **Run Evaluation**: Use `eval.py` for detailed model evaluation
4. **Scale Up**: Increase epochs and batch size for full training runs
5. **Experiment**: Try different datasets and hyperparameters

## Integration with Main Scripts

These CLI tools complement the main training scripts:

- `train.py`: Use generated configs from `setup_and_run.py`
- `eval.py`: Evaluate models trained through the CLI workflow
- `test_implementation.py`: Verify everything works correctly

The CLI tools handle the setup and basic training, while the main scripts provide advanced functionality and detailed control.
