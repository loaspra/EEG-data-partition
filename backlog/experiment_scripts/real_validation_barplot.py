#!/usr/bin/env python3
"""
Real validation script that generates comprehensive barplots comparing the main experiment:
- MLP trained on translated data (S2-S8 translated to S1 space via Min2Net) 
- MLP trained on original data (raw S2-S8 data)
Both tested on Subject 1 held-out data.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Add experiment directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_utils import load_subject_data, extract_features
from mlp_classifier import MLPClassifier
from simple_min2net import SimpleMin2Net

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

class TranslationValidationSuite:
    """
    Real validation suite that tests the main experiment: 
    Translated data MLP vs Original data MLP for P300 classification.
    """
    
    def __init__(self, num_subjects=8):
        self.num_subjects = min(num_subjects, config.NUM_SUBJECTS)
        self.results = {
            'subject_results': [],
            'timing_info': {},
            'translation_info': {}
        }
        self.translator = None
        
    def train_translator(self):
        """
        Train Min2Net translator to map subjects 2-8 to Subject 1 space.
        """
        print(f"\n{'='*50}")
        print("TRAINING MIN2NET TRANSLATOR")
        print(f"{'='*50}")
        
        try:
            # Load reference subject (Subject 1) data
            ref_X_train, ref_y_train = load_subject_data(config.REF_SUBJECT, 'train')
            ref_X_val, ref_y_val = load_subject_data(config.REF_SUBJECT, 'val')
            
            print(f"Reference subject data: Train={ref_X_train.shape}, Val={ref_X_val.shape}")
            
            # Prepare data for Min2Net (reshape from (trials, channels, samples) to (trials, 1, samples, channels))
            from utils.data_utils import prepare_eeg_for_min2net
            ref_X_train = prepare_eeg_for_min2net(ref_X_train)
            ref_X_val = prepare_eeg_for_min2net(ref_X_val)
            
            print(f"Reshaped reference data: Train={ref_X_train.shape}, Val={ref_X_val.shape}")
            
            # Load source subjects (2-8) data
            source_X_list, source_y_list = [], []
            for subject_id in range(2, self.num_subjects + 1):
                try:
                    X_train, y_train = load_subject_data(subject_id, 'train')
                    X_val, y_val = load_subject_data(subject_id, 'val')
                    
                    # Combine train and val for more training data
                    X_combined = np.vstack([X_train, X_val])
                    y_combined = np.concatenate([y_train, y_val])
                    
                    # Prepare data for Min2Net
                    X_combined = prepare_eeg_for_min2net(X_combined)
                    
                    source_X_list.append(X_combined)
                    source_y_list.append(y_combined)
                    print(f"  Subject {subject_id}: {X_combined.shape}")
                except Exception as e:
                    print(f"  Warning: Could not load Subject {subject_id}: {e}")
                    continue
            
            if not source_X_list:
                raise Exception("No source subjects loaded successfully")
            
            # Combine all source data
            source_X = np.vstack(source_X_list)
            source_y = np.concatenate(source_y_list)
            
            # Combine reference data
            ref_X = np.vstack([ref_X_train, ref_X_val])
            ref_y = np.concatenate([ref_y_train, ref_y_val])
            
            print(f"Combined source data: {source_X.shape}")
            print(f"Combined reference data: {ref_X.shape}")
            
            # Initialize and train translator
            print(f"Input shape for SimpleMin2Net: {source_X.shape[1:]}")
            self.translator = SimpleMin2Net(
                input_shape=source_X.shape[1:],
                latent_dim=config.LATENT_DIM,
                batch_size=config.BATCH_SIZE,
                epochs=50,  # Reduced for faster validation
                verbose=1
            )
            
            start_time = time.time()
            
            # Train translator
            history = self.translator.fit(
                source_data=(source_X, source_y),
                target_data=(ref_X, ref_y)
            )
            
            training_time = time.time() - start_time
            print(f"✅ Translator trained in {training_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training translator: {e}")
            return False
    
    def run_subject_validation(self, subject_id, method_name="Translated", use_translation=True, epochs=30):
        """
        Run validation for a single subject using either translated or original data.
        
        Args:
            subject_id (int): Subject ID to validate  
            method_name (str): Name of the method being tested
            use_translation (bool): Whether to use translated data
            epochs (int): Number of training epochs
            
        Returns:
            dict: Validation results for the subject
        """
        print(f"\n{'='*40}")
        print(f"VALIDATING SUBJECT {subject_id} - {method_name}")
        print(f"{'='*40}")
        
        try:
            # Load subject data
            X_train, y_train = load_subject_data(subject_id, 'train')
            X_val, y_val = load_subject_data(subject_id, 'val')
            
            # Combine train and val for more training data
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])
            
            print(f"Loaded data: {X_combined.shape}")
            
            # Apply translation if requested
            if use_translation and self.translator is not None:
                print("  Applying Min2Net translation...")
                start_time = time.time()
                
                # Prepare data for Min2Net
                from utils.data_utils import prepare_eeg_for_min2net
                X_reshaped = prepare_eeg_for_min2net(X_combined)
                X_translated = self.translator.translate(X_reshaped)
                
                # Convert back from (trials, 1, samples, channels) to (trials, channels, samples)
                X_translated_reshaped = np.zeros((X_translated.shape[0], X_translated.shape[3], X_translated.shape[2]))
                for i in range(X_translated.shape[0]):
                    X_translated_reshaped[i] = X_translated[i, 0].T
                
                translation_time = time.time() - start_time
                print(f"  Translation completed in {translation_time:.2f}s")
                X_features_input = X_translated_reshaped
            else:
                X_features_input = X_combined
                translation_time = 0.0
            
            # Extract features
            start_time = time.time()
            features = extract_features(X_features_input)
            extraction_time = time.time() - start_time
            
            print(f"Features extracted: {features.shape} in {extraction_time:.2f}s")
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform 5-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            
            cv_scores = {
                'accuracy': [],
                'f1': [],
                'precision': [],
                'recall': []
            }
            
            fold_times = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(features_scaled, y_combined)):
                print(f"  Fold {fold + 1}/5...")
                
                X_fold_train, X_fold_val = features_scaled[train_idx], features_scaled[val_idx]
                y_fold_train, y_fold_val = y_combined[train_idx], y_combined[val_idx]
                
                # Create and train model
                fold_start = time.time()
                mlp = MLPClassifier(
                    input_shape=features.shape[1],
                    hidden_units=[128, 64],  # Use enhanced architecture for fair comparison
                    dropout=0.4
                )
                
                # Train with reduced output
                mlp.train(X_fold_train, y_fold_train, epochs=epochs, batch_size=32, validation_split=0.0)
                fold_time = time.time() - fold_start
                fold_times.append(fold_time)
                
                # Evaluate
                y_pred = mlp.predict(X_fold_val)
                
                cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
                cv_scores['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
                cv_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
                cv_scores['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            
            # Calculate mean scores
            mean_scores = {metric: np.mean(scores) for metric, scores in cv_scores.items()}
            std_scores = {metric: np.std(scores) for metric, scores in cv_scores.items()}
            
            result = {
                'subject_id': subject_id,
                'method': method_name,
                'f1_mean': mean_scores['f1'],
                'f1_std': std_scores['f1'],
                'accuracy_mean': mean_scores['accuracy'],
                'accuracy_std': std_scores['accuracy'],
                'precision_mean': mean_scores['precision'],
                'recall_mean': mean_scores['recall'],
                'extraction_time': extraction_time,
                'translation_time': translation_time,
                'training_time': np.mean(fold_times),
                'feature_dim': features.shape[1],
                'n_trials': features.shape[0],
                'use_translation': use_translation
            }
            
            print(f"  Results: F1={mean_scores['f1']:.3f}±{std_scores['f1']:.3f}, "
                  f"Acc={mean_scores['accuracy']:.3f}±{std_scores['accuracy']:.3f}")
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error validating subject {subject_id}: {e}")
            return {
                'subject_id': subject_id,
                'method': method_name,
                'error': str(e),
                'f1_mean': 0.0,
                'f1_std': 0.0,
                'accuracy_mean': 0.0,
                'accuracy_std': 0.0,
                'use_translation': use_translation
            }
    
    def run_comprehensive_validation(self):
        """
        Run comprehensive validation comparing translated vs original data.
        """
        print("="*80)
        print("COMPREHENSIVE TRANSLATION EXPERIMENT VALIDATION")
        print("Comparing: MLP on Translated Data vs MLP on Original Data")
        print("="*80)
        
        # First train the translator
        if not self.train_translator():
            print("❌ Failed to train translator - cannot proceed")
            return None
        
        # Run validation for subjects 2-8 (source subjects)
        translated_results = []
        original_results = []
        
        for subject_id in range(2, self.num_subjects + 1):
            try:
                # Test with translated data
                trans_result = self.run_subject_validation(
                    subject_id, "Translated Data", use_translation=True, epochs=25
                )
                if 'error' not in trans_result:
                    translated_results.append(trans_result)
                
                # Test with original data
                orig_result = self.run_subject_validation(
                    subject_id, "Original Data", use_translation=False, epochs=25
                )
                if 'error' not in orig_result:
                    original_results.append(orig_result)
                    
            except Exception as e:
                print(f"Skipping subject {subject_id}: {e}")
                continue
        
        if not translated_results or not original_results:
            print("❌ No successful validations - cannot generate comparison")
            return None
        
        print(f"\n✅ Successfully validated {len(translated_results)} subjects with translation")
        print(f"✅ Successfully validated {len(original_results)} subjects with original data")
        
        # Store results
        self.results = {
            'translated_results': translated_results,
            'original_results': original_results,
            'comparison_data': self._prepare_comparison_data(original_results, translated_results)
        }
        
        return self.results
    
    def _prepare_comparison_data(self, original_results, translated_results):
        """
        Prepare data for comparison plots.
        """
        comparison_data = []
        
        # Match subjects between original and translated results
        orig_dict = {r['subject_id']: r for r in original_results}
        trans_dict = {r['subject_id']: r for r in translated_results}
        
        common_subjects = set(orig_dict.keys()) & set(trans_dict.keys())
        
        for subject_id in common_subjects:
            original = orig_dict[subject_id]
            translated = trans_dict[subject_id]
            
            comparison_data.append({
                'test_subject': subject_id,
                'original_f1': original['f1_mean'],
                'translated_f1': translated['f1_mean'],
                'improvement': translated['f1_mean'] - original['f1_mean'],
                'original_accuracy': original['accuracy_mean'],
                'translated_accuracy': translated['accuracy_mean']
            })
        
        return comparison_data
    
    def create_comprehensive_barplot(self, save_dir):
        """
        Create comprehensive barplot comparing translated vs original data MLPs.
        """
        if not self.results or 'comparison_data' not in self.results:
            print("❌ No results available for plotting")
            return
        
        comparison_data = self.results['comparison_data']
        original_results = self.results['original_results'] 
        translated_results = self.results['translated_results']
        
        # Extract data for plotting
        subjects = [d['test_subject'] for d in comparison_data]
        original_f1 = [d['original_f1'] for d in comparison_data]
        translated_f1 = [d['translated_f1'] for d in comparison_data]
        improvements = [d['improvement'] for d in comparison_data]
        
        # Create comprehensive plot
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Subject-wise comparison
        x_pos = np.arange(len(subjects))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x_pos - width/2, original_f1, width, 
                              label='Original Data MLP', alpha=0.8, color='#ff7f7f')
        bars2 = axes[0, 0].bar(x_pos + width/2, translated_f1, width,
                              label='Translated Data MLP', alpha=0.8, color='#7f7fff')
        
        axes[0, 0].set_xlabel('Subject ID')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].set_title('F1 Score by Subject (Translation Experiment)')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(subjects)
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Improvement distribution
        axes[0, 1].hist(improvements, bins=max(3, len(improvements)//2), alpha=0.7, 
                       edgecolor='black', color='green')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', 
                          label='No improvement', linewidth=2)
        axes[0, 1].set_xlabel('F1 Score Improvement')
        axes[0, 1].set_ylabel('Number of Subjects')
        axes[0, 1].set_title('Distribution of Translation Improvements')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Methods comparison (main barplot)
        methods = ['Original Data\nMLP', 'Translated Data\nMLP']
        means = [np.mean(original_f1), np.mean(translated_f1)]
        stds = [np.std(original_f1), np.std(translated_f1)]
        
        x_pos_methods = np.arange(len(methods))
        bars = axes[0, 2].bar(x_pos_methods, means, yerr=stds, capsize=10,
                             color=['#ff7f7f', '#7f7fff'], alpha=0.8, 
                             edgecolor='black', linewidth=1.5)
        
        axes[0, 2].set_xlabel('Method')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Translation Experiment Comparison')
        axes[0, 2].set_xticks(x_pos_methods)
        axes[0, 2].set_xticklabels(methods)
        axes[0, 2].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + std/2,
                           f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=12)
        
        # Statistical testing
        t_stat, p_value = stats.ttest_rel(translated_f1, original_f1)
        
        # Add significance indicator
        if p_value < 0.05:
            max_height = max(means) + max(stds)
            axes[0, 2].plot([0, 1], [max_height + 0.02, max_height + 0.02], 'k-', linewidth=2)
            axes[0, 2].plot([0, 0], [max_height + 0.015, max_height + 0.02], 'k-', linewidth=2)
            axes[0, 2].plot([1, 1], [max_height + 0.015, max_height + 0.02], 'k-', linewidth=2)
            
            sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'
            axes[0, 2].text(0.5, max_height + 0.025, sig_text, ha='center', va='bottom', 
                           fontsize=16, fontweight='bold')
            axes[0, 2].text(0.5, max_height + 0.035, f'p = {p_value:.4f}', ha='center', va='bottom', 
                           fontsize=10)
        
        # 4. Individual subject improvements
        improvement_colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = axes[1, 0].bar(subjects, improvements, color=improvement_colors, 
                             alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_xlabel('Subject ID')
        axes[1, 0].set_ylabel('F1 Score Improvement')
        axes[1, 0].set_title('Individual Subject Translation Benefits')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., 
                           height + 0.005 if height >= 0 else height - 0.01,
                           f'{imp:.3f}', ha='center', 
                           va='bottom' if height >= 0 else 'top', 
                           fontweight='bold', fontsize=9)
        
        # 5. Box plots comparison
        data_to_plot = [original_f1, translated_f1]
        box_plot = axes[1, 1].boxplot(data_to_plot, labels=['Original Data\nMLP', 'Translated Data\nMLP'], 
                                      patch_artist=True)
        box_plot['boxes'][0].set_facecolor('#ff7f7f')
        box_plot['boxes'][1].set_facecolor('#7f7fff')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Distribution Comparison (Box Plots)')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 6. Statistical summary
        cohens_d = np.mean(improvements) / np.std(improvements) if np.std(improvements) > 0 else 0
        
        stats_text = f"Translation Experiment Results\n"
        stats_text += f"{'='*35}\n\n"
        stats_text += f"Subjects tested: {len(subjects)}\n"
        stats_text += f"Translation: S2-S8 → S1 space\n"
        stats_text += f"Architecture: Min2Net + MLP\n\n"
        stats_text += f"Mean improvement: {np.mean(improvements):.4f}\n"
        stats_text += f"Std improvement: {np.std(improvements):.4f}\n"
        stats_text += f"Cohen's d: {cohens_d:.4f}\n"
        stats_text += f"t-statistic: {t_stat:.4f}\n"
        stats_text += f"p-value: {p_value:.4f}\n\n"
        
        if p_value < 0.05:
            stats_text += "✅ Translation shows significant improvement\n"
        else:
            stats_text += "❌ No significant translation benefit\n"
            
        improved_subjects = sum(1 for i in improvements if i > 0)
        stats_text += f"\nSubjects improved: {improved_subjects}/{len(improvements)}\n"
        
        if translated_results:
            avg_translation_time = np.mean([r['translation_time'] for r in translated_results])
            stats_text += f"Avg translation time: {avg_translation_time:.3f}s\n"
        
        axes[1, 2].text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('Translation Experiment Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save plots
        save_path = os.path.join(save_dir, 'translation_experiment_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        save_path_pdf = os.path.join(save_dir, 'translation_experiment_results.pdf')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        
        print(f"\n📊 Translation experiment plots saved to:")
        print(f"   PNG: {save_path}")
        print(f"   PDF: {save_path_pdf}")
        
        plt.close()
        
        return save_path
    
    def save_results(self, save_dir):
        """
        Save validation results to CSV and JSON.
        """
        if not self.results:
            return
        
        # Save detailed results
        all_results = self.results['original_results'] + self.results['translated_results']
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(save_dir, 'translation_experiment_detailed_results.csv')
        results_df.to_csv(csv_path, index=False)
        
        # Save comparison summary
        comparison_df = pd.DataFrame(self.results['comparison_data'])
        comparison_path = os.path.join(save_dir, 'translation_experiment_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n📄 Results saved to:")
        print(f"   Detailed: {csv_path}")
        print(f"   Comparison: {comparison_path}")


def main():
    """
    Main function to run translation experiment validation and generate comprehensive barplots.
    """
    print("Starting Translation Experiment Validation...")
    print("Comparing: MLP trained on Translated Data vs MLP trained on Original Data")
    
    # Setup results directory
    results_dir = os.path.join(config.RESULTS_DIR, "translation_experiment")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create validator
    validator = TranslationValidationSuite(num_subjects=min(8, config.NUM_SUBJECTS))
    
    # Run validation
    results = validator.run_comprehensive_validation()
    
    if results:
        # Generate comprehensive barplot
        validator.create_comprehensive_barplot(results_dir)
        
        # Save results
        validator.save_results(results_dir)
        
        print(f"\n✅ Translation experiment validation completed successfully!")
        print(f"📁 All results saved to: {results_dir}")
        
    else:
        print("❌ Translation experiment validation failed - no results to plot")
    
    return results


if __name__ == "__main__":
    main() 