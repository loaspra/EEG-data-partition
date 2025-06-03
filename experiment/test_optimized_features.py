#!/usr/bin/env python3
"""
Test script to validate the optimized feature extraction (256D vs 288D).
This script actually extracts features and trains models to verify the optimization works.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add experiment directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_utils import load_subject_data, extract_features
from mlp_classifier import MLPClassifier

class OptimizedFeatureValidator:
    """
    Validator to test the optimized feature extraction.
    """
    
    def __init__(self):
        self.results = {
            'feature_extraction': {},
            'model_performance': {},
            'timing_comparison': {}
        }
        
    def test_feature_extraction(self):
        """
        Test feature extraction to verify we get 256 features instead of 288.
        """
        print("="*60)
        print("TESTING OPTIMIZED FEATURE EXTRACTION")
        print("="*60)
        
        # Load test data for one subject
        try:
            X_test, y_test = load_subject_data(1, 'train')
            print(f"Loaded EEG data shape: {X_test.shape}")
            print(f"Labels shape: {y_test.shape}")
            
            # Extract optimized features
            print("\nExtracting optimized features...")
            start_time = time.time()
            features = extract_features(X_test)
            extraction_time = time.time() - start_time
            
            print(f"✅ Feature extraction completed in {extraction_time:.2f} seconds")
            print(f"Feature matrix shape: {features.shape}")
            print(f"Expected: (n_trials, 256)")
            print(f"Actual:   ({features.shape[0]}, {features.shape[1]})")
            
            # Verify feature count
            expected_features = 256
            actual_features = features.shape[1]
            
            if actual_features == expected_features:
                print(f"✅ SUCCESS: Got expected {expected_features} features")
            else:
                print(f"❌ ERROR: Expected {expected_features} features, got {actual_features}")
                
            # Store results
            self.results['feature_extraction'] = {
                'expected_features': expected_features,
                'actual_features': actual_features,
                'extraction_time': extraction_time,
                'success': actual_features == expected_features
            }
            
            return features, y_test
            
        except Exception as e:
            print(f"❌ Error in feature extraction test: {e}")
            self.results['feature_extraction'] = {'error': str(e)}
            return None, None
    
    def test_model_training(self, features, labels):
        """
        Test training MLP with optimized features.
        """
        print("\n" + "="*60)
        print("TESTING MODEL TRAINING WITH OPTIMIZED FEATURES")
        print("="*60)
        
        if features is None:
            print("❌ Cannot test model training - no features available")
            return
            
        try:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create MLP with correct input dimension
            input_dim = features.shape[1]
            print(f"Creating MLP with input_dim={input_dim}")
            
            mlp = MLPClassifier(
                input_shape=input_dim,
                hidden_units=64,
                dropout=0.5
            )
            
            # Perform cross-validation
            print("Performing 5-fold cross-validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            
            start_time = time.time()
            
            # Manual cross-validation to get detailed metrics
            cv_results = {
                'accuracy': [],
                'f1': [],
                'precision': [],
                'recall': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(features_scaled, labels)):
                print(f"Training fold {fold + 1}/5...")
                
                X_train, X_val = features_scaled[train_idx], features_scaled[val_idx]
                y_train, y_val = labels[train_idx], labels[val_idx]
                
                # Create new model for each fold
                fold_mlp = MLPClassifier(
                    input_shape=input_dim,
                    hidden_units=64,
                    dropout=0.5
                )
                
                # Train model (suppress output)
                fold_mlp.train(X_train, y_train, epochs=20, batch_size=32, validation_split=0.0)
                
                # Evaluate
                y_pred = fold_mlp.predict(X_val)
                
                cv_results['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                cv_results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            
            training_time = time.time() - start_time
            
            # Calculate mean and std for each metric
            results_summary = {}
            for metric, scores in cv_results.items():
                results_summary[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
            
            print(f"\n✅ Cross-validation completed in {training_time:.2f} seconds")
            print("Results Summary:")
            print("-" * 40)
            for metric, stats in results_summary.items():
                print(f"{metric.capitalize():>12}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            
            self.results['model_performance'] = {
                'cv_results': results_summary,
                'training_time': training_time,
                'input_dimension': input_dim
            }
            
            return results_summary
            
        except Exception as e:
            print(f"❌ Error in model training test: {e}")
            self.results['model_performance'] = {'error': str(e)}
            return None
    
    def run_timing_comparison(self):
        """
        Compare timing between optimized and non-optimized feature extraction.
        """
        print("\n" + "="*60)
        print("TIMING COMPARISON")
        print("="*60)
        
        try:
            # Load test data
            X_test, _ = load_subject_data(1, 'train')
            
            # Time optimized feature extraction (multiple runs for accuracy)
            n_runs = 5
            optimized_times = []
            
            print(f"Timing optimized feature extraction ({n_runs} runs)...")
            for i in range(n_runs):
                start_time = time.time()
                features = extract_features(X_test)
                extraction_time = time.time() - start_time
                optimized_times.append(extraction_time)
                print(f"  Run {i+1}: {extraction_time:.3f}s")
            
            avg_time = np.mean(optimized_times)
            std_time = np.std(optimized_times)
            
            print(f"\nOptimized extraction: {avg_time:.3f} ± {std_time:.3f} seconds")
            print(f"Features per second: {features.shape[1] / avg_time:.1f}")
            
            self.results['timing_comparison'] = {
                'optimized_times': optimized_times,
                'average_time': avg_time,
                'std_time': std_time,
                'features_per_second': features.shape[1] / avg_time
            }
            
        except Exception as e:
            print(f"❌ Error in timing comparison: {e}")
            self.results['timing_comparison'] = {'error': str(e)}
    
    def generate_report(self):
        """
        Generate a comprehensive report of the validation results.
        """
        print("\n" + "="*60)
        print("OPTIMIZATION VALIDATION REPORT")
        print("="*60)
        
        # Feature extraction results
        if 'feature_extraction' in self.results and 'error' not in self.results['feature_extraction']:
            fe_results = self.results['feature_extraction']
            print(f"\n📊 Feature Extraction:")
            print(f"   Expected features: {fe_results['expected_features']}")
            print(f"   Actual features: {fe_results['actual_features']}")
            print(f"   Extraction time: {fe_results['extraction_time']:.3f}s")
            print(f"   Success: {'✅' if fe_results['success'] else '❌'}")
        
        # Model performance results
        if 'model_performance' in self.results and 'error' not in self.results['model_performance']:
            mp_results = self.results['model_performance']
            print(f"\n🤖 Model Performance (Input dim: {mp_results['input_dimension']}):")
            cv_results = mp_results['cv_results']
            for metric, stats in cv_results.items():
                print(f"   {metric.capitalize():>12}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"   Training time: {mp_results['training_time']:.2f}s")
        
        # Timing comparison
        if 'timing_comparison' in self.results and 'error' not in self.results['timing_comparison']:
            tc_results = self.results['timing_comparison']
            print(f"\n⏱️  Timing Performance:")
            print(f"   Average extraction time: {tc_results['average_time']:.3f} ± {tc_results['std_time']:.3f}s")
            print(f"   Features per second: {tc_results['features_per_second']:.1f}")
        
        # Overall assessment
        print(f"\n🎯 Overall Assessment:")
        feature_success = ('feature_extraction' in self.results and 
                          self.results['feature_extraction'].get('success', False))
        model_success = ('model_performance' in self.results and 
                        'error' not in self.results['model_performance'])
        
        if feature_success and model_success:
            print("   ✅ Optimization successful!")
            print("   ✅ Features correctly reduced from 288 to 256 dimensions")
            print("   ✅ Model training works with optimized features")
            
            if 'model_performance' in self.results:
                f1_score = self.results['model_performance']['cv_results']['f1']['mean']
                if f1_score > 0.5:
                    print(f"   ✅ Reasonable F1 score achieved: {f1_score:.3f}")
                else:
                    print(f"   ⚠️  Low F1 score: {f1_score:.3f} (may need tuning)")
        else:
            print("   ❌ Issues detected in optimization")
            if not feature_success:
                print("   ❌ Feature extraction problems")
            if not model_success:
                print("   ❌ Model training problems")
    
    def save_results(self, filepath):
        """Save results to a file."""
        try:
            import json
            with open(filepath, 'w') as f:
                # Convert numpy types for JSON serialization
                serializable_results = self._make_json_serializable(self.results)
                json.dump(serializable_results, f, indent=2)
            print(f"\n📄 Results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def main():
    """
    Main function to run optimized feature validation.
    """
    print("Starting optimized feature validation...")
    
    # Create validator
    validator = OptimizedFeatureValidator()
    
    # Run tests
    features, labels = validator.test_feature_extraction()
    validator.test_model_training(features, labels)
    validator.run_timing_comparison()
    
    # Generate report
    validator.generate_report()
    
    # Save results
    results_dir = os.path.join(config.RESULTS_DIR, "optimization_validation")
    os.makedirs(results_dir, exist_ok=True)
    validator.save_results(os.path.join(results_dir, "optimization_results.json"))
    
    return validator.results


if __name__ == "__main__":
    main() 