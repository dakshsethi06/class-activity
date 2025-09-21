"""
Enhanced HGNN Implementation with Hyperparameter Experimentation
Supports multiple datasets, network depths, and comprehensive hyperparameter tuning
"""

import os
import time
import copy
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pprint as pp
from datetime import datetime
from itertools import product
import utils.hypergraph_utils as hgut
from models import HGNN
from config import get_config
from datasets import load_feature_construct_H


class EnhancedHGNN(nn.Module):
    """Enhanced HGNN with configurable depth and architecture"""
    
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5, depth=2, 
                 hidden_dims=None, activation='relu', use_batch_norm=False):
        super(EnhancedHGNN, self).__init__()
        self.depth = depth
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Define activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu
        
        # Build layers dynamically
        if hidden_dims is None:
            hidden_dims = [n_hid] * (depth - 1)
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Input layer
        self.layers.append(HGNN_conv(in_ch, hidden_dims[0]))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # Hidden layers
        for i in range(depth - 2):
            self.layers.append(HGNN_conv(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(HGNN_conv(hidden_dims[-1], n_class))
    
    def forward(self, x, G):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, G)
            if self.batch_norms:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Output layer (no activation)
        x = self.layers[-1](x, G)
        return x


class HGNN_conv(nn.Module):
    """Hypergraph convolution layer"""
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class ExperimentRunner:
    """Comprehensive experiment runner for HGNN hyperparameter tuning"""
    
    def __init__(self, base_config_path='config/config.yaml'):
        self.base_config = get_config(base_config_path)
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def define_hyperparameter_space(self):
        """Define comprehensive hyperparameter search space"""
        return {
            # Network Architecture
            'depth': [2, 3, 4, 5],
            'n_hid': [64, 128, 256, 512],
            'hidden_dims': [
                [128, 128],
                [256, 128],
                [512, 256],
                [256, 256, 128],
                [512, 256, 128]
            ],
            'activation': ['relu', 'leaky_relu', 'gelu'],
            'use_batch_norm': [True, False],
            'dropout': [0.1, 0.3, 0.5, 0.7],
            
            # Training Parameters
            'lr': [0.0001, 0.001, 0.01, 0.1],
            'weight_decay': [0.0001, 0.001, 0.01],
            'optimizer': ['adam', 'sgd', 'adamw'],
            'scheduler': ['step', 'cosine', 'plateau'],
            
            # Hypergraph Parameters
            'K_neigs': [[5], [10], [15], [5, 10], [10, 15]],
            'm_prob': [0.5, 1.0, 1.5, 2.0],
            'is_probH': [True, False],
            
            # Feature Configuration
            'use_mvcnn_feature': [True, False],
            'use_gvcnn_feature': [True, False],
            'use_mvcnn_feature_for_structure': [True, False],
            'use_gvcnn_feature_for_structure': [True, False],
            
            # Training Configuration
            'max_epoch': [100, 300, 600],
            'print_freq': [25, 50, 100]
        }
    
    def generate_experiments(self, max_experiments=50):
        """Generate experiment configurations"""
        param_space = self.define_hyperparameter_space()
        
        # Create a subset of experiments for manageable testing
        experiments = []
        
        # Key parameter combinations
        key_combinations = [
            # Basic configurations
            {'depth': 2, 'n_hid': 128, 'lr': 0.001, 'dropout': 0.5},
            {'depth': 3, 'n_hid': 256, 'lr': 0.001, 'dropout': 0.3},
            {'depth': 4, 'n_hid': 512, 'lr': 0.0001, 'dropout': 0.7},
            
            # Different optimizers
            {'optimizer': 'adam', 'lr': 0.001},
            {'optimizer': 'sgd', 'lr': 0.01},
            {'optimizer': 'adamw', 'lr': 0.001},
            
            # Different hypergraph configurations
            {'K_neigs': [5], 'm_prob': 1.0},
            {'K_neigs': [10], 'm_prob': 1.5},
            {'K_neigs': [5, 10], 'm_prob': 2.0},
            
            # Feature combinations
            {'use_mvcnn_feature': True, 'use_gvcnn_feature': False},
            {'use_mvcnn_feature': False, 'use_gvcnn_feature': True},
            {'use_mvcnn_feature': True, 'use_gvcnn_feature': True},
        ]
        
        for i, base_config in enumerate(key_combinations[:max_experiments]):
            exp_config = self.base_config.copy()
            exp_config.update(base_config)
            exp_config['experiment_id'] = f"exp_{i+1:03d}"
            experiments.append(exp_config)
        
        return experiments
    
    def run_single_experiment(self, config):
        """Run a single experiment with given configuration"""
        print(f"\n{'='*60}")
        print(f"Running Experiment: {config['experiment_id']}")
        print(f"{'='*60}")
        
        try:
            # Set up data
            data_dir = config['modelnet40_ft'] if config['on_dataset'] == 'ModelNet40' else config['ntu2012_ft']
            
            print("Loading data and constructing hypergraph...")
            fts, lbls, idx_train, idx_test, H = load_feature_construct_H(
                data_dir,
                m_prob=config['m_prob'],
                K_neigs=config['K_neigs'],
                is_probH=config['is_probH'],
                use_mvcnn_feature=config['use_mvcnn_feature'],
                use_gvcnn_feature=config['use_gvcnn_feature'],
                use_mvcnn_feature_for_structure=config['use_mvcnn_feature_for_structure'],
                use_gvcnn_feature_for_structure=config['use_gvcnn_feature_for_structure']
            )
            
            G = hgut.generate_G_from_H(H)
            n_class = int(lbls.max()) + 1
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # Move data to device
            fts = torch.Tensor(fts).to(device)
            lbls = torch.Tensor(lbls).squeeze().long().to(device)
            G = torch.Tensor(G).to(device)
            idx_train = torch.Tensor(idx_train).long().to(device)
            idx_test = torch.Tensor(idx_test).long().to(device)
            
            # Create model
            hidden_dims = config.get('hidden_dims', [config['n_hid']] * (config['depth'] - 1))
            model = EnhancedHGNN(
                in_ch=fts.shape[1],
                n_class=n_class,
                n_hid=config['n_hid'],
                dropout=config['dropout'],
                depth=config['depth'],
                hidden_dims=hidden_dims,
                activation=config.get('activation', 'relu'),
                use_batch_norm=config.get('use_batch_norm', False)
            ).to(device)
            
            # Set up optimizer
            if config.get('optimizer', 'adam') == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            elif config['optimizer'] == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], momentum=0.9)
            elif config['optimizer'] == 'adamw':
                optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            
            # Set up scheduler
            if config.get('scheduler', 'step') == 'step':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
            elif config['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'])
            elif config['scheduler'] == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=50)
            
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            print(f"Training with {config['max_epoch']} epochs...")
            best_acc, training_history = self.train_model(
                model, criterion, optimizer, scheduler, 
                fts, lbls, idx_train, idx_test, G,
                num_epochs=config['max_epoch'],
                print_freq=config['print_freq']
            )
            
            # Store results
            result = {
                'experiment_id': config['experiment_id'],
                'config': config,
                'best_accuracy': best_acc,
                'training_history': training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            print(f"Experiment completed. Best accuracy: {best_acc:.4f}")
            
            return result
            
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            return {'experiment_id': config['experiment_id'], 'error': str(e)}
    
    def train_model(self, model, criterion, optimizer, scheduler, fts, lbls, idx_train, idx_test, G, 
                   num_epochs=100, print_freq=50):
        """Enhanced training function with detailed logging"""
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            optimizer.zero_grad()
            
            outputs = model(fts, G)
            loss = criterion(outputs[idx_train], lbls[idx_train])
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            train_acc = torch.sum(preds[idx_train] == lbls[idx_train]).double() / len(idx_train)
            train_loss = loss.item()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                outputs = model(fts, G)
                val_loss = criterion(outputs[idx_test], lbls[idx_test]).item()
                _, preds = torch.max(outputs, 1)
                val_acc = torch.sum(preds[idx_test] == lbls[idx_test]).double() / len(idx_test)
            
            # Update scheduler
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc.item())
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc.item())
            
            # Update best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # Print progress
            if epoch % print_freq == 0:
                print(f'Epoch {epoch:3d}/{num_epochs-1} | '
                      f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                      f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | '
                      f'Best Val Acc: {best_acc:.4f}')
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best validation accuracy: {best_acc:.4f}')
        
        # Load best model weights
        model.load_state_dict(best_model_wts)
        
        return best_acc.item(), training_history
    
    def run_experiments(self, max_experiments=10):
        """Run multiple experiments"""
        experiments = self.generate_experiments(max_experiments)
        
        print(f"Starting {len(experiments)} experiments...")
        
        for i, exp_config in enumerate(experiments):
            print(f"\nProgress: {i+1}/{len(experiments)}")
            self.run_single_experiment(exp_config)
        
        # Save results
        self.save_results()
        self.analyze_results()
    
    def save_results(self):
        """Save experiment results to file"""
        results_file = f"experiment_results_{self.experiment_id}.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in self.results:
            if 'error' not in result:
                # Convert numpy arrays to lists
                history = result['training_history']
                for key in history:
                    if isinstance(history[key], list):
                        history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in history[key]]
                
                # Convert other numpy types
                if 'best_accuracy' in result:
                    result['best_accuracy'] = float(result['best_accuracy'])
            
            serializable_results.append(result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def analyze_results(self):
        """Analyze and display experiment results"""
        if not self.results:
            print("No results to analyze")
            return
        
        successful_results = [r for r in self.results if 'error' not in r]
        
        if not successful_results:
            print("No successful experiments to analyze")
            return
        
        # Sort by best accuracy
        successful_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
        
        print(f"\n{'='*80}")
        print("EXPERIMENT ANALYSIS")
        print(f"{'='*80}")
        print(f"Total experiments: {len(self.results)}")
        print(f"Successful experiments: {len(successful_results)}")
        print(f"Failed experiments: {len(self.results) - len(successful_results)}")
        
        print(f"\nTop 5 Results:")
        print(f"{'Rank':<4} {'Experiment ID':<12} {'Best Accuracy':<15} {'Key Parameters'}")
        print("-" * 80)
        
        for i, result in enumerate(successful_results[:5]):
            config = result['config']
            key_params = f"depth={config.get('depth', 'N/A')}, lr={config.get('lr', 'N/A')}, n_hid={config.get('n_hid', 'N/A')}"
            print(f"{i+1:<4} {result['experiment_id']:<12} {result['best_accuracy']:<15.4f} {key_params}")
        
        # Parameter analysis
        print(f"\nParameter Analysis:")
        self._analyze_parameter_impact(successful_results)
    
    def _analyze_parameter_impact(self, results):
        """Analyze the impact of different parameters"""
        param_analysis = {}
        
        for result in results:
            config = result['config']
            accuracy = result['best_accuracy']
            
            for param, value in config.items():
                if param in ['experiment_id', 'data_root', 'result_root', 'ckpt_folder', 'result_sub_folder']:
                    continue
                
                if param not in param_analysis:
                    param_analysis[param] = {}
                
                value_str = str(value)
                if value_str not in param_analysis[param]:
                    param_analysis[param][value_str] = []
                
                param_analysis[param][value_str].append(accuracy)
        
        # Calculate average accuracy for each parameter value
        for param, values in param_analysis.items():
            print(f"\n{param}:")
            avg_accuracies = {}
            for value, accs in values.items():
                avg_accuracies[value] = np.mean(accs)
            
            # Sort by average accuracy
            sorted_values = sorted(avg_accuracies.items(), key=lambda x: x[1], reverse=True)
            for value, avg_acc in sorted_values[:3]:  # Top 3 values
                print(f"  {value}: {avg_acc:.4f} (n={len(values[value])})")


def main():
    """Main function to run experiments"""
    print("Enhanced HGNN Experimentation Framework")
    print("=" * 50)
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Run experiments
    runner.run_experiments(max_experiments=10)


if __name__ == '__main__':
    main()

