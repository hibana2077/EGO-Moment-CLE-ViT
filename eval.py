"""
Evaluation script for EGO-Moment-CLE-ViT

This script handles model evaluation including:
- Loading trained models
- Computing evaluation metrics
- Generating visualizations
- Ablation studies
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import EGOMomentCLEViT, CLEViTDataTransforms
from utils import (
    set_seed, 
    plot_confusion_matrix, 
    plot_feature_embeddings,
    visualize_moment_features,
    plot_graph_weights,
    plot_polynomial_coefficients
)
from dataset.ufgvc import UFGVCDataset

# Optional dependencies
try:
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class Evaluator:
    """
    Evaluator class for EGO-Moment-CLE-ViT
    
    Handles model evaluation, metric computation, and visualization generation.
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # Set random seed
        set_seed(config['experiment']['seed'])
        
        # Initialize components
        self.model = None
        self.test_loader = None
        
        # Results storage
        self.results = {}
        
        # Setup directories
        self._setup_directories()
    
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        device_config = self.config['experiment']['device']
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        self.logger.info(f"Device: {device}")
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('EGOMomentCLEViT_Eval')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_directories(self):
        """Create necessary directories"""
        eval_dir = Path(self.config['experiment']['output_dir']) / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir = eval_dir
    
    def setup_data(self):
        """Setup test data loader"""
        self.logger.info("Setting up test data loader...")
        
        data_config = self.config['data']
        dataset_config = self.config['dataset']
        
        # Create test transforms (no augmentation)
        test_transforms = CLEViTDataTransforms(
            input_size=data_config['input_size'],
            resize_size=data_config['resize_size'],
            is_training=False
        )
        
        # Load test dataset
        try:
            test_dataset = UFGVCDataset(
                dataset_name=dataset_config['name'],
                root=dataset_config['root'],
                split='test',
                transform=test_transforms,
                download=False  # Should already be downloaded
            )
        except:
            # If no test split, use validation split
            self.logger.warning("No test split found, using validation split")
            test_dataset = UFGVCDataset(
                dataset_name=dataset_config['name'],
                root=dataset_config['root'],
                split='val',
                transform=test_transforms,
                download=False
            )
        
        # Create data loader
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        self.num_classes = test_dataset.num_classes
        self.class_names = getattr(test_dataset, 'class_names', None)
        
        self.logger.info(f"Test samples: {len(test_dataset)}")
        self.logger.info(f"Num classes: {self.num_classes}")
    
    def load_model(self):
        """Load trained model from checkpoint"""
        self.logger.info(f"Loading model from {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Update config with checkpoint config if needed
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Create model
        model_config = self.config['model']
        self.model = EGOMomentCLEViT(
            num_classes=self.num_classes,
            backbone_name=model_config['backbone_name'],
            pretrained=False,  # Don't need pretrained weights when loading
            gpf_degree_p=model_config['gpf']['degree_p'],
            gpf_degree_q=model_config['gpf']['degree_q'],
            gpf_similarity=model_config['gpf']['similarity'],
            moment_d_out=model_config['moment']['d_out'],
            use_third_order=model_config['moment']['use_third_order'],
            isqrt_iterations=model_config['moment']['isqrt_iterations'],
            sketch_dim=model_config['moment']['sketch_dim'],
            classifier_fusion=model_config['classifier']['fusion_type'],
            classifier_hidden=model_config['classifier']['hidden_dim'],
            dropout=model_config['classifier']['dropout']
        )
        
        # Load state dict
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("Model loaded successfully")
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics"""
        self.logger.info("Computing evaluation metrics...")
        
        all_predictions = []
        all_labels = []
        all_features = []
        all_moment_features = []
        all_graphs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                # Handle batch format
                if len(batch) == 4:
                    anchor, positive, labels, _ = batch
                else:
                    anchor, positive, labels = batch
                
                anchor = anchor.to(self.device, non_blocking=True)
                positive = positive.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                output = self.model(anchor, positive, return_features=True)
                
                # Get predictions
                _, predicted = torch.max(output['logits'], 1)
                
                # Store results
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store features for visualization
                features = output['features']
                all_features.append(features['anchor_global'].cpu())
                all_moment_features.append(features['moment_features'].cpu())
                all_graphs.append(features['fused_graph'].cpu())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute basic metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(all_labels, all_predictions) * 100
        
        # Top-k accuracy
        if hasattr(self, '_compute_topk_accuracy'):
            metrics['top5_accuracy'] = self._compute_topk_accuracy(all_labels, all_predictions, k=5)
        
        # Per-class accuracy (if sklearn available)
        if HAS_SKLEARN:
            report = classification_report(all_labels, all_predictions, output_dict=True)
            metrics['mean_per_class_accuracy'] = report['macro avg']['recall'] * 100
            
            # Store detailed report
            self.results['classification_report'] = report
        
        # Store predictions and features
        self.results['predictions'] = all_predictions
        self.results['labels'] = all_labels
        self.results['features'] = {
            'cls_features': torch.cat(all_features, dim=0),
            'moment_features': torch.cat(all_moment_features, dim=0),
            'graphs': torch.cat(all_graphs, dim=0)
        }
        
        self.results['metrics'] = metrics
        
        # Log metrics
        self.logger.info("Evaluation Results:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.2f}%")
        
        return metrics
    
    def generate_visualizations(self):
        """Generate evaluation visualizations"""
        self.logger.info("Generating visualizations...")
        
        labels = self.results['labels']
        predictions = self.results['predictions']
        features = self.results['features']
        
        # 1. Confusion Matrix
        if HAS_SKLEARN:
            plot_confusion_matrix(
                labels,
                predictions,
                class_names=self.class_names,
                title="Confusion Matrix",
                save_path=str(self.eval_dir / "confusion_matrix.png")
            )
        
        # 2. Feature Embeddings
        # Sample subset for visualization (if too many samples)
        max_samples = 1000
        if len(labels) > max_samples:
            indices = np.random.choice(len(labels), max_samples, replace=False)
            vis_labels = labels[indices]
            vis_cls_features = features['cls_features'][indices]
            vis_moment_features = features['moment_features'][indices]
        else:
            vis_labels = labels
            vis_cls_features = features['cls_features']
            vis_moment_features = features['moment_features']
        
        # CLS features
        plot_feature_embeddings(
            vis_cls_features,
            vis_labels,
            method='tsne',
            title="CLS Features (t-SNE)",
            save_path=str(self.eval_dir / "cls_features_tsne.png")
        )
        
        # Moment features
        plot_feature_embeddings(
            vis_moment_features,
            vis_labels,
            method='tsne',
            title="Moment Features (t-SNE)",
            save_path=str(self.eval_dir / "moment_features_tsne.png")
        )
        
        # 3. Moment feature analysis
        visualize_moment_features(
            vis_moment_features,
            vis_labels,
            feature_type="moment",
            save_path=str(self.eval_dir / "moment_analysis")
        )
        
        # 4. Graph weights visualization
        sample_graph = features['graphs'][0]  # Take first sample
        plot_graph_weights(
            sample_graph,
            spatial_layout=(14, 14),  # Assume 14x14 ViT patches
            title="Sample Graph Weights",
            save_path=str(self.eval_dir / "graph_weights.png")
        )
        
        # 5. Polynomial coefficients
        if hasattr(self.model, 'gpf'):
            coeffs = self.model.gpf.get_coefficient_matrix()
            plot_polynomial_coefficients(
                coeffs,
                title="Learned Polynomial Coefficients",
                save_path=str(self.eval_dir / "polynomial_coefficients.png")
            )
        
        self.logger.info(f"Visualizations saved to {self.eval_dir}")
    
    def save_results(self):
        """Save evaluation results to file"""
        results_file = self.eval_dir / "results.json"
        
        # Prepare results for JSON serialization
        json_results = {
            'metrics': self.results['metrics'],
            'config': self.config,
            'checkpoint_path': self.checkpoint_path
        }
        
        # Add classification report if available
        if 'classification_report' in self.results:
            json_results['classification_report'] = self.results['classification_report']
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def run_ablation_study(self):
        """Run ablation study if configured"""
        if not self.config.get('ablation', {}).get('enabled', False):
            return
        
        self.logger.info("Running ablation study...")
        
        ablation_config = self.config['ablation']
        components = ablation_config.get('components', [])
        
        ablation_results = {}
        
        for component in components:
            self.logger.info(f"Ablating component: {component}")
            
            # Modify model for ablation
            modified_model = self._create_ablated_model(component)
            
            # Evaluate modified model
            ablated_metrics = self._evaluate_model(modified_model)
            ablation_results[component] = ablated_metrics
            
            self.logger.info(f"Ablation {component} results: {ablated_metrics}")
        
        # Save ablation results
        ablation_file = self.eval_dir / "ablation_results.json"
        with open(ablation_file, 'w') as f:
            json.dump(ablation_results, f, indent=2)
        
        self.logger.info(f"Ablation results saved to {ablation_file}")
    
    def _create_ablated_model(self, component: str):
        """Create model with specific component ablated"""
        # This would create modified versions of the model
        # For now, return the original model (placeholder)
        return self.model
    
    def _evaluate_model(self, model) -> Dict[str, float]:
        """Evaluate a specific model configuration"""
        # This would run evaluation on the modified model
        # For now, return dummy results (placeholder)
        return {"accuracy": 0.0}
    
    def evaluate(self):
        """Run complete evaluation pipeline"""
        self.setup_data()
        self.load_model()
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save results
        self.save_results()
        
        # Run ablation study if enabled
        self.run_ablation_study()
        
        self.logger.info("Evaluation completed!")
        
        return metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Evaluate EGO-Moment-CLE-ViT")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--device', type=str, help='Override device setting')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config['experiment']['output_dir'] = args.output_dir
    if args.device:
        config['experiment']['device'] = args.device
    
    # Create evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
