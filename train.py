"""
Training script for EGO-Moment-CLE-ViT

This script handles the complete training pipeline including:
- Model initialization and configuration
- Data loading and augmentation  
- Training loop with loss computation
- Validation and evaluation
- Checkpointing and logging
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import EGOMomentCLEViT, CLEViTDataTransforms
from losses import TripletLoss, KernelAlignmentLoss
from utils import set_seed, count_parameters, plot_training_curves
from dataset.ufgvc import UFGVCDataset

# Optional dependencies
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("wandb not available. Logging to wandb disabled.")

try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_AMP = True
except ImportError:
    HAS_AMP = False
    warnings.warn("AMP not available. Mixed precision disabled.")


class Trainer:
    """
    Trainer class for EGO-Moment-CLE-ViT
    
    Handles the complete training pipeline including data loading,
    model training, validation, and logging.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Set random seed
        set_seed(config['experiment']['seed'])
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Setup directories
        self._setup_directories()
        
        # Initialize wandb if enabled
        if config['experiment']['wandb']['enabled'] and HAS_WANDB:
            self._setup_wandb()
    
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        device_config = self.config['experiment']['device']
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        if device.type == 'cuda':
            gpu_ids = self.config['experiment']['gpu_ids']
            if len(gpu_ids) > 1:
                self.logger.info(f"Using multiple GPUs: {gpu_ids}")
            else:
                torch.cuda.set_device(gpu_ids[0])
                self.logger.info(f"Using GPU: {gpu_ids[0]}")
        
        self.logger.info(f"Device: {device}")
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('EGOMomentCLEViT')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = log_dir / f"{self.config['experiment']['name']}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_directories(self):
        """Create necessary directories"""
        for dir_name in ['output_dir', 'save_dir', 'log_dir']:
            dir_path = Path(self.config['experiment'][dir_name])
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = self.config['experiment']['wandb']
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            name=self.config['experiment']['name'],
            config=self.config
        )
    
    def setup_data(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")
        
        data_config = self.config['data']
        dataset_config = self.config['dataset']
        
        # Create transforms
        train_transforms = CLEViTDataTransforms(
            input_size=data_config['input_size'],
            resize_size=data_config['resize_size'],
            is_training=True,
            mask_ratio=tuple(data_config['mask_ratio'])
        )
        
        val_transforms = CLEViTDataTransforms(
            input_size=data_config['input_size'],
            resize_size=data_config['resize_size'],
            is_training=False
        )
        
        # Load dataset
        full_dataset = UFGVCDataset(
            dataset_name=dataset_config['name'],
            root=dataset_config['root'],
            split='train',  # We'll split manually
            transform=None,  # Will be set per split
            download=dataset_config['download']
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(dataset_config['train_split'] * total_size)
        val_size = total_size - train_size
        
        generator = torch.Generator().manual_seed(dataset_config['random_seed'])
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        
        # Set transforms for each split
        train_dataset.dataset.transform = train_transforms
        val_dataset.dataset.transform = val_transforms
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            persistent_workers=data_config['persistent_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory'],
            persistent_workers=data_config['persistent_workers']
        )
        
        # Update num_classes in config
        self.config['model']['num_classes'] = full_dataset.num_classes
        
        self.logger.info(f"Dataset: {dataset_config['name']}")
        self.logger.info(f"Train samples: {len(train_dataset)}")
        self.logger.info(f"Val samples: {len(val_dataset)}")
        self.logger.info(f"Num classes: {full_dataset.num_classes}")
    
    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        self.logger.info("Setting up model...")
        
        model_config = self.config['model']
        
        # Create model
        self.model = EGOMomentCLEViT(
            num_classes=model_config['num_classes'],
            backbone_name=model_config['backbone_name'],
            pretrained=model_config['pretrained'],
            gpf_degree_p=model_config['gpf']['degree_p'],
            gpf_degree_q=model_config['gpf']['degree_q'],
            gpf_similarity=model_config['gpf']['similarity'],
            moment_d_out=model_config['moment']['d_out'],
            use_third_order=model_config['moment']['use_third_order'],
            isqrt_iterations=model_config['moment']['isqrt_iterations'],
            sketch_dim=model_config['moment']['sketch_dim'],
            classifier_fusion=model_config['classifier']['fusion_type'],
            classifier_hidden=model_config['classifier']['hidden_dim'],
            lambda_triplet=self.config['training']['loss']['lambda_triplet'],
            lambda_align=self.config['training']['loss']['lambda_align'],
            margin=self.config['training']['loss']['margin'],
            dropout=model_config['classifier']['dropout']
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Multi-GPU setup
        gpu_ids = self.config['experiment']['gpu_ids']
        if len(gpu_ids) > 1 and torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
        
        # Log model info
        num_params = count_parameters(self.model)
        self.logger.info(f"Model parameters: {num_params:,}")
        
        # Setup optimizer
        train_config = self.config['training']
        opt_config = train_config['optimizer']
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas'],
            eps=opt_config['eps']
        )
        
        # Setup scheduler
        sched_config = train_config['scheduler']
        if sched_config['name'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['epochs'],
                eta_min=sched_config['min_lr']
            )
        
        # Setup mixed precision scaler
        if train_config['amp'] and HAS_AMP:
            self.scaler = GradScaler()
            self.logger.info("Mixed precision training enabled")
        
        self.logger.info("Model setup completed")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        train_config = self.config['training']
        log_freq = self.config['experiment']['log_frequency']
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle batch format from UFG dataset
            if len(batch) == 4:  # anchor, positive, label, idx
                anchor, positive, labels, _ = batch
            else:  # Standard format
                anchor, positive, labels = batch
            
            anchor = anchor.to(self.device, non_blocking=True)
            positive = positive.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            if train_config['amp'] and self.scaler is not None:
                with autocast():
                    output = self.model(anchor, positive, labels)
                    loss = output['loss']
            else:
                output = self.model(anchor, positive, labels)
                loss = output['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if train_config['amp'] and self.scaler is not None:
                self.scaler.scale(loss).backward()
                if train_config['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), train_config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if train_config['grad_clip'] > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), train_config['grad_clip'])
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Compute accuracy using main logits
            with torch.no_grad():
                _, predicted = torch.max(output['logits'], 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            
            # Update progress bar
            current_acc = 100.0 * total_correct / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            # Logging
            if batch_idx % log_freq == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%"
                )
                
                if HAS_WANDB and self.config['experiment']['wandb']['enabled']:
                    wandb.log({
                        'train_loss_step': loss.item(),
                        'train_acc_step': current_acc,
                        'epoch': self.current_epoch,
                        'step': self.current_epoch * len(self.train_loader) + batch_idx
                    })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Handle batch format
                if len(batch) == 4:
                    anchor, positive, labels, _ = batch
                else:
                    anchor, positive, labels = batch
                
                anchor = anchor.to(self.device, non_blocking=True)
                positive = positive.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(anchor, positive, labels)
                loss = output['loss']
                
                total_loss += loss.item()
                
                # Compute accuracy
                _, predicted = torch.max(output['logits'], 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        save_dir = Path(self.config['experiment']['save_dir'])
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation accuracy: {self.best_val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        num_epochs = self.config['training']['epochs']
        val_freq = self.config['training']['val_frequency']
        save_freq = self.config['training']['save_frequency']
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            if epoch % val_freq == 0:
                val_loss, val_acc = self.validate()
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # Check if best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                
                self.logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
                )
                
                # Wandb logging
                if HAS_WANDB and self.config['experiment']['wandb']['enabled']:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc,
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                
                # Save checkpoint
                if epoch % save_freq == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
        
        # Save final checkpoint
        self.save_checkpoint()
        
        # Plot training curves
        output_dir = Path(self.config['experiment']['output_dir'])
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            title=f"Training Curves - {self.config['experiment']['name']}",
            save_path=str(output_dir / "training_curves.png")
        )
        
        self.logger.info(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train EGO-Moment-CLE-ViT")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, help='Override device setting')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device:
        config['experiment']['device'] = args.device
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    
    # Create trainer
    trainer = Trainer(config)
    
    # Setup data and model
    trainer.setup_data()
    trainer.setup_model()
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if trainer.scaler and checkpoint['scaler_state_dict']:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_val_acc = checkpoint['best_val_acc']
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
