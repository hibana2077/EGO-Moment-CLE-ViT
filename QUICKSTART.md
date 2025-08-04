# Quick Start Guide for CLI Tools

This guide helps you get started with the EGO-Moment-CLE-ViT CLI tools quickly.

## 🚀 30-Second Quick Start

```bash
# 1. Check if everything is ready
python setup_and_run.py --check-only

# 2. Download dataset and run quick training (5 epochs)
python setup_and_run.py --dataset cotton80 --epochs 5 --batch_size 16
```

That's it! Your model will be trained and ready.

## 📋 Available Commands

### List Available Datasets
```bash
python download_simple.py --list
```

### Download a Specific Dataset
```bash
python download_simple.py --dataset cotton80
```

### Get Dataset Information
```bash
python download_simple.py --info cotton80
```

### Complete Training Pipeline
```bash
# Quick training (recommended for first time)
python setup_and_run.py --dataset cotton80 --epochs 10

# Full training with evaluation
python setup_and_run.py --dataset cotton80 --epochs 50 --with-evaluation

# Large batch training
python setup_and_run.py --dataset cotton80 --epochs 20 --batch_size 64
```

### Test the Tools
```bash
python test_cli_tools.py
```

## 📊 Available Datasets

| Dataset | Description | Classes | Samples |
|---------|-------------|---------|---------|
| `cotton80` | Cotton classification | 80 | 720 |
| `soybean` | Soybean classification | Variable | Variable |
| `soy_ageing_r1` | Soybean ageing - Round 1 | Variable | Variable |
| `soy_ageing_r3` | Soybean ageing - Round 3 | Variable | Variable |
| `soy_ageing_r4` | Soybean ageing - Round 4 | Variable | Variable |
| `soy_ageing_r5` | Soybean ageing - Round 5 | Variable | Variable |
| `soy_ageing_r6` | Soybean ageing - Round 6 | Variable | Variable |

## 🎯 Recommended Workflows

### For Quick Testing
```bash
python setup_and_run.py --dataset cotton80 --epochs 5 --batch_size 16
```

### For Full Experiments
```bash
python setup_and_run.py --dataset cotton80 --epochs 50 --batch_size 32 --with-evaluation
```

### For Large Scale Training
```bash
python setup_and_run.py --dataset cotton80 --epochs 100 --batch_size 64 --config_name large_scale
```

## 📁 Generated Files

After running the tools, you'll find:

```
├── data/                          # Downloaded datasets
│   └── cotton80_dataset.parquet
├── configs/                       # Generated configurations
│   └── quick_start.yaml
├── checkpoints/                   # Saved models
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── logs/                         # Training logs
│   └── quickstart_cotton80_*epochs.log
└── outputs/                      # Training results
    └── training_curves.png
```

## 🔧 Troubleshooting

### Windows Encoding Issues
If you see Unicode errors, use the "simple" versions:
- `download_simple.py` instead of `download_dataset.py`
- All tools now support ASCII-only output

### Memory Issues
Reduce batch size:
```bash
python setup_and_run.py --dataset cotton80 --batch_size 8
```

### Dependency Issues
Check and install:
```bash
python setup_and_run.py --check-only
pip install -r requirements.txt
```

## 📚 Next Steps

1. **Check Results**: Look at `outputs/training_curves.png`
2. **View Logs**: Check `logs/` directory for detailed training info
3. **Run Evaluation**: Use `python eval.py --config configs/quick_start.yaml --checkpoint checkpoints/best_model.pth`
4. **Experiment**: Try different datasets and parameters

## 🛠️ Advanced Usage

### Custom Configuration
```bash
# Generate config without training
python setup_and_run.py --dataset cotton80 --epochs 20 --download-only

# Edit the config file
# configs/quick_start.yaml

# Train with custom config
python train.py --config configs/quick_start.yaml
```

### Manual Dataset Management
```bash
# Download to custom directory
python download_simple.py --dataset cotton80 --root /custom/path

# Force re-download
python download_simple.py --dataset cotton80 --force

# Skip verification
python download_simple.py --dataset cotton80 --no-verify
```

## 💡 Pro Tips

1. **Start Small**: Always begin with `--epochs 5` for testing
2. **Monitor Resources**: Use Task Manager to check GPU/CPU usage
3. **Save Configs**: Generated configs can be reused and modified
4. **Batch Size**: Start with 16, increase if you have more memory
5. **Multiple Runs**: Change `--config_name` for different experiments

## 🤝 Getting Help

If you encounter issues:
1. Run `python test_cli_tools.py` to verify tools work
2. Check `logs/` directory for error messages
3. Try reducing batch size or epochs
4. Ensure all dependencies are installed

Happy training! 🎉
