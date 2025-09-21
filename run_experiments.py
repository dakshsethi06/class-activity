"""
Comprehensive Experiment Runner for Enhanced HGNN
Supports hyperparameter tuning, dataset experimentation, and network depth analysis
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from enhanced_hgnn import ExperimentRunner, EnhancedHGNN
from data_augmentation import DataAugmentation, MultiDatasetLoader, DatasetAnalyzer
import utils.hypergraph_utils as hgut
from config import get_config


class ComprehensiveExperimentRunner(ExperimentRunner):
    """Extended experiment runner with advanced features"""
    
    def __init__(self, config_path='config/config.yaml', experiment_config_path='experiment_configs.yaml'):
        super().__init__(config_path)
        self.experiment_config = self._load_experiment_config(experiment_config_path)
        self.dataset_loader = MultiDatasetLoader(self.base_config)
        self.dataset_analyzer = DatasetAnalyzer()
        
    def _load_experiment_config(self, config_path):
        """Load experiment configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Warning: {config_path} not found. Using default configuration.")
            return {}
    
    def run_depth_experiment(self, dataset='ModelNet40', base_config=None):
        """Experiment with different network depths"""
        print(f"\n{'='*60}")
        print(f"NETWORK DEPTH EXPERIMENT - {dataset}")
        print(f"{'='*60}")
        
        if base_config is None:
            base_config = self.base_config.copy()
        
        # Load dataset
        data_config = base_config.copy()
        if dataset == 'ModelNet40':
            data_config['on_dataset'] = 'ModelNet40'
        else:
            data_config['on_dataset'] = 'NTU2012'
        
        # Test different depths
        depths = [2, 3, 4, 5, 6]
        results = []
        
        for depth in depths:
            print(f"\nTesting depth: {depth}")
            
            config = data_config.copy()
            config.update({
                'experiment_id': f'depth_{depth}_{dataset}',
                'depth': depth,
                'n_hid': 256,
                'hidden_dims': [256] * (depth - 1),
                'max_epoch': 200,
                'print_freq': 50
            })
            
            result = self.run_single_experiment(config)
            if 'error' not in result:
                results.append({
                    'depth': depth,
                    'accuracy': result['best_accuracy'],
                    'config': config
                })
        
        # Analyze results
        if results:
            print(f"\nDepth Experiment Results for {dataset}:")
            print(f"{'Depth':<6} {'Accuracy':<10} {'Improvement':<12}")
            print("-" * 30)
            
            best_acc = max(results, key=lambda x: x['accuracy'])['accuracy']
            for result in sorted(results, key=lambda x: x['depth']):
                improvement = result['accuracy'] - results[0]['accuracy']
                print(f"{result['depth']:<6} {result['accuracy']:<10.4f} {improvement:+.4f}")
        
        return results
    
    def run_hyperparameter_sweep(self, dataset='ModelNet40', param_ranges=None):
        """Run comprehensive hyperparameter sweep"""
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER SWEEP - {dataset}")
        print(f"{'='*60}")
        
        if param_ranges is None:
            param_ranges = {
                'lr': [0.0001, 0.001, 0.01],
                'n_hid': [128, 256, 512],
                'dropout': [0.3, 0.5, 0.7],
                'K_neigs': [[5], [10], [5, 10]]
            }
        
        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        results = []
        total_combinations = np.prod([len(v) for v in param_values])
        
        print(f"Testing {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names))):
            config = self.base_config.copy()
            config.update({
                'experiment_id': f'sweep_{i+1:03d}_{dataset}',
                'on_dataset': dataset,
                'max_epoch': 150,
                'print_freq': 50
            })
            
            # Add parameter values
            for j, param_name in enumerate(param_names):
                config[param_name] = combination[j]
            
            print(f"\nProgress: {i+1}/{total_combinations}")
            print(f"Testing: {dict(zip(param_names, combination))}")
            
            result = self.run_single_experiment(config)
            if 'error' not in result:
                results.append({
                    'params': dict(zip(param_names, combination)),
                    'accuracy': result['best_accuracy'],
                    'config': config
                })
        
        # Analyze results
        if results:
            print(f"\nHyperparameter Sweep Results for {dataset}:")
            self._analyze_hyperparameter_results(results, param_names)
        
        return results
    
    def run_dataset_comparison(self, datasets=['ModelNet40', 'NTU2012']):
        """Compare performance across different datasets"""
        print(f"\n{'='*60}")
        print(f"DATASET COMPARISON")
        print(f"{'='*60}")
        
        results = []
        
        for dataset in datasets:
            print(f"\nTesting on {dataset}...")
            
            config = self.base_config.copy()
            config.update({
                'experiment_id': f'dataset_{dataset}',
                'on_dataset': dataset,
                'depth': 3,
                'n_hid': 256,
                'max_epoch': 300,
                'print_freq': 50
            })
            
            result = self.run_single_experiment(config)
            if 'error' not in result:
                results.append({
                    'dataset': dataset,
                    'accuracy': result['best_accuracy'],
                    'config': config
                })
        
        # Compare results
        if results:
            print(f"\nDataset Comparison Results:")
            print(f"{'Dataset':<12} {'Accuracy':<10} {'Parameters'}")
            print("-" * 50)
            
            for result in results:
                config = result['config']
                params = f"depth={config['depth']}, n_hid={config['n_hid']}"
                print(f"{result['dataset']:<12} {result['accuracy']:<10.4f} {params}")
        
        return results
    
    def run_augmentation_experiment(self, dataset='ModelNet40'):
        """Test different data augmentation techniques"""
        print(f"\n{'='*60}")
        print(f"DATA AUGMENTATION EXPERIMENT - {dataset}")
        print(f"{'='*60}")
        
        # Load dataset with augmentation
        augmentation_config = {
            'noise': {'levels': [0.01, 0.05]},
            'dropout': {'rates': [0.1, 0.2]},
            'normalization': {'methods': ['standard']},
            'hypergraph': {'types': ['edge_dropout']}
        }
        
        augmented_data = self.dataset_loader.load_dataset(dataset, augmentation_config)
        
        results = []
        
        for data in augmented_data:
            print(f"\nTesting augmentation: {data['augmentation']}")
            
            # Create config for this augmentation
            config = self.base_config.copy()
            config.update({
                'experiment_id': f'aug_{data["augmentation"]}_{dataset}',
                'on_dataset': dataset,
                'depth': 3,
                'n_hid': 256,
                'max_epoch': 200,
                'print_freq': 50
            })
            
            # Run experiment with augmented data
            result = self._run_experiment_with_data(config, data)
            if 'error' not in result:
                results.append({
                    'augmentation': data['augmentation'],
                    'accuracy': result['best_accuracy'],
                    'config': config
                })
        
        # Analyze results
        if results:
            print(f"\nAugmentation Experiment Results for {dataset}:")
            print(f"{'Augmentation':<20} {'Accuracy':<10} {'Improvement':<12}")
            print("-" * 45)
            
            baseline_acc = next(r['accuracy'] for r in results if r['augmentation'] == 'original')
            for result in results:
                improvement = result['accuracy'] - baseline_acc
                print(f"{result['augmentation']:<20} {result['accuracy']:<10.4f} {improvement:+.4f}")
        
        return results
    
    def _run_experiment_with_data(self, config, data):
        """Run experiment with pre-loaded data"""
        try:
            # Use the provided data instead of loading from file
            fts, lbls, H = data['features'], data['labels'], data['H']
            
            G = hgut.generate_G_from_H(H)
            n_class = int(lbls.max()) + 1
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # Move data to device
            fts = torch.Tensor(fts).to(device)
            lbls = torch.Tensor(lbls).squeeze().long().to(device)
            G = torch.Tensor(G).to(device)
            
            # Create train/test splits (simple 80/20 split)
            n_samples = len(lbls)
            n_train = int(0.8 * n_samples)
            indices = torch.randperm(n_samples)
            idx_train = indices[:n_train]
            idx_test = indices[n_train:]
            
            idx_train = idx_train.to(device)
            idx_test = idx_test.to(device)
            
            # Create model
            model = EnhancedHGNN(
                in_ch=fts.shape[1],
                n_class=n_class,
                n_hid=config['n_hid'],
                dropout=config['dropout'],
                depth=config['depth']
            ).to(device)
            
            # Set up training
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.9)
            criterion = torch.nn.CrossEntropyLoss()
            
            # Train model
            best_acc, training_history = self.train_model(
                model, criterion, optimizer, scheduler,
                fts, lbls, idx_train, idx_test, G,
                num_epochs=config['max_epoch'],
                print_freq=config['print_freq']
            )
            
            return {
                'experiment_id': config['experiment_id'],
                'config': config,
                'best_accuracy': best_acc,
                'training_history': training_history,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'experiment_id': config['experiment_id'], 'error': str(e)}
    
    def _analyze_hyperparameter_results(self, results, param_names):
        """Analyze hyperparameter sweep results"""
        print(f"\nTop 5 Parameter Combinations:")
        print(f"{'Rank':<4} {'Accuracy':<10} {'Parameters'}")
        print("-" * 50)
        
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            params_str = ', '.join([f"{k}={v}" for k, v in result['params'].items()])
            print(f"{i+1:<4} {result['accuracy']:<10.4f} {params_str}")
        
        # Parameter importance analysis
        print(f"\nParameter Importance Analysis:")
        for param_name in param_names:
            param_values = [r['params'][param_name] for r in results]
            accuracies = [r['accuracy'] for r in results]
            
            # Calculate correlation between parameter value and accuracy
            if isinstance(param_values[0], (int, float)):
                correlation = np.corrcoef(param_values, accuracies)[0, 1]
                print(f"{param_name}: correlation = {correlation:.3f}")
            else:
                # For categorical parameters, show average accuracy per value
                unique_values = list(set(param_values))
                print(f"{param_name}:")
                for value in unique_values:
                    value_accs = [r['accuracy'] for r in results if r['params'][param_name] == value]
                    avg_acc = np.mean(value_accs)
                    print(f"  {value}: {avg_acc:.4f} (n={len(value_accs)})")
    
    def run_comprehensive_experiment(self, experiment_type='all'):
        """Run comprehensive experiments based on type"""
        print(f"Starting Comprehensive HGNN Experiments")
        print(f"Experiment Type: {experiment_type}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        all_results = {}
        
        if experiment_type in ['all', 'depth']:
            all_results['depth'] = self.run_depth_experiment()
        
        if experiment_type in ['all', 'hyperparams']:
            all_results['hyperparams'] = self.run_hyperparameter_sweep()
        
        if experiment_type in ['all', 'datasets']:
            all_results['datasets'] = self.run_dataset_comparison()
        
        if experiment_type in ['all', 'augmentation']:
            all_results['augmentation'] = self.run_augmentation_experiment()
        
        # Save all results
        self._save_comprehensive_results(all_results)
        
        return all_results
    
    def _save_comprehensive_results(self, all_results):
        """Save comprehensive experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_results_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = {}
        for exp_type, results in all_results.items():
            serializable_results[exp_type] = []
            for result in results:
                if 'error' not in result:
                    # Convert numpy types
                    if 'accuracy' in result:
                        result['accuracy'] = float(result['accuracy'])
                    if 'training_history' in result:
                        history = result['training_history']
                        for key in history:
                            if isinstance(history[key], list):
                                history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in history[key]]
                serializable_results[exp_type].append(result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nComprehensive results saved to {results_file}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced HGNN Experiment Runner')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'depth', 'hyperparams', 'datasets', 'augmentation'],
                       help='Type of experiment to run')
    parser.add_argument('--dataset', type=str, default='ModelNet40',
                       choices=['ModelNet40', 'NTU2012'],
                       help='Dataset to use')
    parser.add_argument('--max_experiments', type=int, default=20,
                       help='Maximum number of experiments to run')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ComprehensiveExperimentRunner()
    
    # Run experiments
    if args.experiment == 'all':
        runner.run_comprehensive_experiment()
    elif args.experiment == 'depth':
        runner.run_depth_experiment(args.dataset)
    elif args.experiment == 'hyperparams':
        runner.run_hyperparameter_sweep(args.dataset)
    elif args.experiment == 'datasets':
        runner.run_dataset_comparison()
    elif args.experiment == 'augmentation':
        runner.run_augmentation_experiment(args.dataset)


if __name__ == '__main__':
    main()

