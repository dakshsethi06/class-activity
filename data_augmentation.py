"""
Data Augmentation Module for HGNN
Supports various augmentation techniques for hypergraph neural networks
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import utils.hypergraph_utils as hgut


class DataAugmentation:
    """Data augmentation techniques for hypergraph data"""
    
    def __init__(self, augmentation_type='none'):
        self.augmentation_type = augmentation_type
        self.scaler = None
        
    def normalize_features(self, features, method='standard'):
        """Normalize features using different methods"""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_features = self.scaler.fit_transform(features)
        return normalized_features
    
    def add_noise(self, features, noise_level=0.01):
        """Add Gaussian noise to features"""
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    def feature_dropout(self, features, dropout_rate=0.1):
        """Randomly set some features to zero"""
        mask = np.random.random(features.shape) > dropout_rate
        return features * mask
    
    def feature_permutation(self, features, permute_ratio=0.1):
        """Randomly permute a subset of features"""
        n_features = features.shape[1]
        n_permute = int(n_features * permute_ratio)
        
        permuted_features = features.copy()
        for i in range(features.shape[0]):
            perm_indices = np.random.choice(n_features, n_permute, replace=False)
            permuted_features[i, perm_indices] = np.random.permutation(permuted_features[i, perm_indices])
        
        return permuted_features
    
    def pca_augmentation(self, features, n_components=0.8):
        """Augment features using PCA"""
        if n_components < 1:
            n_components = int(features.shape[1] * n_components)
        
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features)
        
        # Add some noise to create variations
        noise = np.random.normal(0, 0.01, pca_features.shape)
        augmented_features = pca_features + noise
        
        return augmented_features
    
    def hypergraph_augmentation(self, H, augmentation_type='edge_dropout'):
        """Augment hypergraph structure"""
        if augmentation_type == 'edge_dropout':
            # Randomly remove some hyperedges
            dropout_rate = 0.1
            mask = np.random.random(H.shape[1]) > dropout_rate
            H_aug = H.copy()
            H_aug[:, ~mask] = 0
            return H_aug
        
        elif augmentation_type == 'node_dropout':
            # Randomly remove some nodes
            dropout_rate = 0.05
            mask = np.random.random(H.shape[0]) > dropout_rate
            H_aug = H.copy()
            H_aug[~mask, :] = 0
            return H_aug
        
        elif augmentation_type == 'edge_noise':
            # Add noise to hyperedge weights
            noise = np.random.normal(1, 0.1, H.shape)
            H_aug = H * noise
            H_aug = np.maximum(H_aug, 0)  # Ensure non-negative
            return H_aug
        
        else:
            return H
    
    def augment_dataset(self, features, labels, H, augmentation_config):
        """Apply multiple augmentation techniques"""
        augmented_data = []
        
        # Original data
        augmented_data.append({
            'features': features,
            'labels': labels,
            'H': H,
            'augmentation': 'original'
        })
        
        # Apply different augmentations
        for aug_type, aug_params in augmentation_config.items():
            if aug_type == 'noise':
                for noise_level in aug_params.get('levels', [0.01, 0.05]):
                    aug_features = self.add_noise(features, noise_level)
                    augmented_data.append({
                        'features': aug_features,
                        'labels': labels,
                        'H': H,
                        'augmentation': f'noise_{noise_level}'
                    })
            
            elif aug_type == 'dropout':
                for dropout_rate in aug_params.get('rates', [0.1, 0.2]):
                    aug_features = self.feature_dropout(features, dropout_rate)
                    augmented_data.append({
                        'features': aug_features,
                        'labels': labels,
                        'H': H,
                        'augmentation': f'dropout_{dropout_rate}'
                    })
            
            elif aug_type == 'normalization':
                for norm_method in aug_params.get('methods', ['standard', 'minmax']):
                    aug_features = self.normalize_features(features, norm_method)
                    augmented_data.append({
                        'features': aug_features,
                        'labels': labels,
                        'H': H,
                        'augmentation': f'norm_{norm_method}'
                    })
            
            elif aug_type == 'hypergraph':
                for hg_aug_type in aug_params.get('types', ['edge_dropout', 'edge_noise']):
                    aug_H = self.hypergraph_augmentation(H, hg_aug_type)
                    augmented_data.append({
                        'features': features,
                        'labels': labels,
                        'H': aug_H,
                        'augmentation': f'hg_{hg_aug_type}'
                    })
        
        return augmented_data


class MultiDatasetLoader:
    """Loader for multiple datasets with augmentation support"""
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        self.augmentation = DataAugmentation()
    
    def load_dataset(self, dataset_name, augmentation_config=None):
        """Load and optionally augment a dataset"""
        if dataset_name == 'ModelNet40':
            data_dir = self.config['modelnet40_ft']
        elif dataset_name == 'NTU2012':
            data_dir = self.config['ntu2012_ft']
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Load original data
        from datasets import load_feature_construct_H
        
        fts, lbls, idx_train, idx_test, H = load_feature_construct_H(
            data_dir,
            m_prob=self.config['m_prob'],
            K_neigs=self.config['K_neigs'],
            is_probH=self.config['is_probH'],
            use_mvcnn_feature=self.config['use_mvcnn_feature'],
            use_gvcnn_feature=self.config['use_gvcnn_feature'],
            use_mvcnn_feature_for_structure=self.config['use_mvcnn_feature_for_structure'],
            use_gvcnn_feature_for_structure=self.config['use_gvcnn_feature_for_structure']
        )
        
        if augmentation_config:
            # Apply augmentations
            augmented_data = self.augmentation.augment_dataset(fts, lbls, H, augmentation_config)
            return augmented_data
        else:
            return [{
                'features': fts,
                'labels': lbls,
                'H': H,
                'idx_train': idx_train,
                'idx_test': idx_test,
                'augmentation': 'original'
            }]
    
    def create_cross_validation_splits(self, features, labels, n_splits=5):
        """Create cross-validation splits"""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []
        
        for train_idx, val_idx in kf.split(features):
            splits.append({
                'train_idx': train_idx,
                'val_idx': val_idx
            })
        
        return splits


class DatasetAnalyzer:
    """Analyze dataset characteristics and suggest optimal hyperparameters"""
    
    def __init__(self):
        pass
    
    def analyze_dataset(self, features, labels, H):
        """Analyze dataset and return recommendations"""
        analysis = {}
        
        # Basic statistics
        analysis['n_samples'] = features.shape[0]
        analysis['n_features'] = features.shape[1]
        analysis['n_classes'] = len(np.unique(labels))
        analysis['n_hyperedges'] = H.shape[1]
        
        # Feature analysis
        analysis['feature_mean'] = np.mean(features)
        analysis['feature_std'] = np.std(features)
        analysis['feature_sparsity'] = np.mean(features == 0)
        
        # Hypergraph analysis
        analysis['hypergraph_density'] = np.mean(H > 0)
        analysis['avg_hyperedge_size'] = np.mean(np.sum(H > 0, axis=0))
        analysis['avg_node_degree'] = np.mean(np.sum(H > 0, axis=1))
        
        # Recommendations
        recommendations = self._generate_recommendations(analysis)
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _generate_recommendations(self, analysis):
        """Generate hyperparameter recommendations based on analysis"""
        recommendations = {}
        
        # Learning rate recommendations
        if analysis['n_samples'] < 1000:
            recommendations['learning_rate'] = 0.01
        elif analysis['n_samples'] < 10000:
            recommendations['learning_rate'] = 0.001
        else:
            recommendations['learning_rate'] = 0.0001
        
        # Hidden size recommendations
        if analysis['n_features'] < 100:
            recommendations['hidden_size'] = 64
        elif analysis['n_features'] < 500:
            recommendations['hidden_size'] = 128
        elif analysis['n_features'] < 1000:
            recommendations['hidden_size'] = 256
        else:
            recommendations['hidden_size'] = 512
        
        # Dropout recommendations
        if analysis['feature_sparsity'] > 0.5:
            recommendations['dropout'] = 0.3
        elif analysis['feature_sparsity'] > 0.2:
            recommendations['dropout'] = 0.5
        else:
            recommendations['dropout'] = 0.7
        
        # Network depth recommendations
        if analysis['n_samples'] < 1000:
            recommendations['depth'] = 2
        elif analysis['n_samples'] < 5000:
            recommendations['depth'] = 3
        else:
            recommendations['depth'] = 4
        
        # Hypergraph parameters
        if analysis['hypergraph_density'] > 0.1:
            recommendations['K_neigs'] = [5, 10]
        else:
            recommendations['K_neigs'] = [10, 15]
        
        return recommendations


def create_augmentation_config():
    """Create a standard augmentation configuration"""
    return {
        'noise': {
            'levels': [0.01, 0.05, 0.1]
        },
        'dropout': {
            'rates': [0.1, 0.2, 0.3]
        },
        'normalization': {
            'methods': ['standard', 'minmax']
        },
        'hypergraph': {
            'types': ['edge_dropout', 'edge_noise']
        }
    }

