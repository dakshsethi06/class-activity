"""
Quick Start Script for Enhanced HGNN Experiments
This script provides easy-to-use functions for running different types of experiments
"""

import os
import sys
from enhanced_hgnn import ExperimentRunner, EnhancedHGNN
from run_experiments import ComprehensiveExperimentRunner
from data_augmentation import DataAugmentation, MultiDatasetLoader, DatasetAnalyzer


def run_quick_validation():
    """Run a quick validation experiment to test the setup"""
    print("Running Quick Validation Experiment...")
    print("=" * 50)
    
    runner = ExperimentRunner()
    
    # Quick test configuration
    config = runner.base_config.copy()
    config.update({
        'experiment_id': 'quick_test',
        'max_epoch': 50,
        'print_freq': 10,
        'depth': 2,
        'n_hid': 128,
        'lr': 0.001,
        'dropout': 0.5
    })
    
    result = runner.run_single_experiment(config)
    
    if 'error' not in result:
        print(f"\n✅ Quick validation successful!")
        print(f"Best accuracy: {result['best_accuracy']:.4f}")
        return True
    else:
        print(f"\n❌ Quick validation failed: {result['error']}")
        return False


def run_depth_comparison():
    """Compare different network depths"""
    print("Running Network Depth Comparison...")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner()
    results = runner.run_depth_experiment('ModelNet40')
    
    if results:
        print(f"\n✅ Depth comparison completed!")
        print(f"Tested depths: {[r['depth'] for r in results]}")
        print(f"Best depth: {max(results, key=lambda x: x['accuracy'])['depth']}")
        return True
    else:
        print(f"\n❌ Depth comparison failed")
        return False


def run_hyperparameter_tuning():
    """Run hyperparameter tuning experiment"""
    print("Running Hyperparameter Tuning...")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner()
    
    # Limited hyperparameter sweep for quick testing
    param_ranges = {
        'lr': [0.001, 0.01],
        'n_hid': [128, 256],
        'dropout': [0.3, 0.5]
    }
    
    results = runner.run_hyperparameter_sweep('ModelNet40', param_ranges)
    
    if results:
        print(f"\n✅ Hyperparameter tuning completed!")
        print(f"Tested {len(results)} combinations")
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"Best accuracy: {best_result['accuracy']:.4f}")
        print(f"Best parameters: {best_result['params']}")
        return True
    else:
        print(f"\n❌ Hyperparameter tuning failed")
        return False


def run_dataset_comparison():
    """Compare performance on different datasets"""
    print("Running Dataset Comparison...")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner()
    results = runner.run_dataset_comparison(['ModelNet40'])
    
    if results:
        print(f"\n✅ Dataset comparison completed!")
        for result in results:
            print(f"{result['dataset']}: {result['accuracy']:.4f}")
        return True
    else:
        print(f"\n❌ Dataset comparison failed")
        return False


def run_augmentation_test():
    """Test data augmentation techniques"""
    print("Running Data Augmentation Test...")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner()
    results = runner.run_augmentation_experiment('ModelNet40')
    
    if results:
        print(f"\n✅ Augmentation test completed!")
        print(f"Tested {len(results)} augmentation techniques")
        return True
    else:
        print(f"\n❌ Augmentation test failed")
        return False


def run_full_experiment():
    """Run comprehensive experiment suite"""
    print("Running Full Experiment Suite...")
    print("=" * 50)
    
    runner = ComprehensiveExperimentRunner()
    results = runner.run_comprehensive_experiment('all')
    
    if results:
        print(f"\n✅ Full experiment suite completed!")
        print(f"Experiment types: {list(results.keys())}")
        return True
    else:
        print(f"\n❌ Full experiment suite failed")
        return False


def main():
    """Main function with interactive menu"""
    print("Enhanced HGNN Experimentation Framework")
    print("=" * 50)
    print("Choose an experiment to run:")
    print("1. Quick Validation (50 epochs)")
    print("2. Network Depth Comparison")
    print("3. Hyperparameter Tuning")
    print("4. Dataset Comparison")
    print("5. Data Augmentation Test")
    print("6. Full Experiment Suite")
    print("7. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                run_quick_validation()
            elif choice == '2':
                run_depth_comparison()
            elif choice == '3':
                run_hyperparameter_tuning()
            elif choice == '4':
                run_dataset_comparison()
            elif choice == '5':
                run_augmentation_test()
            elif choice == '6':
                run_full_experiment()
            elif choice == '7':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nExperiment interrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == '__main__':
    main()

