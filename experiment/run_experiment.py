"""
Main script for running the P300 Translation Experiment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time
from scipy import stats

# Import config first to ensure global seed is set
import config
from utils import (
    preprocess_and_partition_data,
    train_min2net_translator,
    translate_source_subjects,
    extract_features_from_translated_data,
    train_and_evaluate_classifiers,
    compare_original_vs_translated,
    load_original_features,
    visualize_latent_space
)

def create_comprehensive_comparison_plots(translated_metrics, original_metrics, save_dir):
    """
    Create comprehensive plots comparing translated vs original data MLPs.
    
    Args:
        translated_metrics: Metrics from MLP trained on translated data
        original_metrics: Metrics from MLP trained on original data
        save_dir: Directory to save plots
    """
    print("\n=== Creating Comprehensive Comparison Plots ===")
    
    # Create results directory
    plots_dir = os.path.join(save_dir, "comparison_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract metrics
    trans_acc = translated_metrics["accuracy"]
    trans_f1 = translated_metrics["f1"]
    trans_prec = translated_metrics["precision"]
    trans_rec = translated_metrics["recall"]
    trans_cm = translated_metrics["confusion_matrix"]
    
    orig_acc = original_metrics["accuracy"]
    orig_f1 = original_metrics["f1"]
    orig_prec = original_metrics["precision"]
    orig_rec = original_metrics["recall"]
    orig_cm = original_metrics["confusion_matrix"]
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # 1. Main Comparison Bar Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Metrics comparison
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    original_values = [orig_acc, orig_f1, orig_prec, orig_rec]
    translated_values = [trans_acc, trans_f1, trans_prec, trans_rec]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x_pos - width/2, original_values, width, 
                          label='Original Data MLP', alpha=0.8, color='#ff7f7f')
    bars2 = axes[0, 0].bar(x_pos + width/2, translated_values, width,
                          label='Translated Data MLP', alpha=0.8, color='#7f7fff')
    
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('MLP Performance Comparison\n(Both tested on Subject 1 held-out data)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Improvement Analysis
    improvements = np.array(translated_values) - np.array(original_values)
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = axes[0, 1].bar(metrics, improvements, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('Metrics')
    axes[0, 1].set_ylabel('Improvement (Translated - Original)')
    axes[0, 1].set_title('Translation Benefits by Metric')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., 
                       height + 0.005 if height >= 0 else height - 0.01,
                       f'{imp:.3f}', ha='center', 
                       va='bottom' if height >= 0 else 'top', 
                       fontweight='bold', fontsize=10)
    
    # 3. Confusion Matrices Comparison
    # Original model confusion matrix
    sns.heatmap(orig_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-P300', 'P300'], 
                yticklabels=['Non-P300', 'P300'], 
                ax=axes[0, 2])
    axes[0, 2].set_title('Original Data MLP\nConfusion Matrix')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('True')
    
    # Translated model confusion matrix
    sns.heatmap(trans_cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Non-P300', 'P300'], 
                yticklabels=['Non-P300', 'P300'], 
                ax=axes[1, 0])
    axes[1, 0].set_title('Translated Data MLP\nConfusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # 4. Radar Chart for Comprehensive Comparison
    categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Add values for each model
    original_values_radar = original_values + [original_values[0]]  # Complete the circle
    translated_values_radar = translated_values + [translated_values[0]]  # Complete the circle
    
    axes[1, 1].plot(angles, original_values_radar, 'o-', linewidth=2, label='Original Data MLP', color='#ff7f7f')
    axes[1, 1].fill(angles, original_values_radar, alpha=0.25, color='#ff7f7f')
    axes[1, 1].plot(angles, translated_values_radar, 'o-', linewidth=2, label='Translated Data MLP', color='#7f7fff')
    axes[1, 1].fill(angles, translated_values_radar, alpha=0.25, color='#7f7fff')
    
    # Add category labels
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Performance Radar Chart')
    axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    axes[1, 1].grid(True)
    
    # 5. Statistical Summary and Experiment Info
    experiment_summary = f"""
Translation Experiment Results
{'='*35}

METHODOLOGY:
• Training Data: Subjects 2-8 (Source → Subject 1 space)
• Test Data: Subject 1 held-out data
• Architecture: Min2Net + MLP Classifier
• Features: Time-domain statistical features

PERFORMANCE COMPARISON:
Original Data MLP:
  Accuracy:  {orig_acc:.4f}
  F1 Score:  {orig_f1:.4f}
  Precision: {orig_prec:.4f}
  Recall:    {orig_rec:.4f}

Translated Data MLP:
  Accuracy:  {trans_acc:.4f}
  F1 Score:  {trans_f1:.4f}
  Precision: {trans_prec:.4f}
  Recall:    {trans_rec:.4f}

IMPROVEMENTS:
  Accuracy:  {trans_acc - orig_acc:+.4f}
  F1 Score:  {trans_f1 - orig_f1:+.4f}
  Precision: {trans_prec - orig_prec:+.4f}
  Recall:    {trans_rec - orig_rec:+.4f}

CONCLUSION:
"""
    
    if trans_f1 > orig_f1:
        if trans_f1 - orig_f1 > 0.05:
            experiment_summary += "✅ Translation shows SIGNIFICANT improvement"
        else:
            experiment_summary += "✅ Translation shows modest improvement"
    else:
        experiment_summary += "❌ Translation did not improve performance"
    
    axes[1, 2].text(0.05, 0.95, experiment_summary, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                   transform=axes[1, 2].transAxes, family='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Experiment Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save main comparison plot
    main_plot_path = os.path.join(plots_dir, 'main_comparison.png')
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"Main comparison plot saved to: {main_plot_path}")
    
    # Also save as PDF
    main_plot_pdf = os.path.join(plots_dir, 'main_comparison.pdf')
    plt.savefig(main_plot_pdf, bbox_inches='tight')
    
    plt.close()
    
    # 6. Create additional focused plots
    create_focused_comparison_plots(translated_metrics, original_metrics, plots_dir)
    
    return plots_dir

def create_focused_comparison_plots(translated_metrics, original_metrics, plots_dir):
    """
    Create additional focused comparison plots.
    """
    # Extract probabilities for ROC analysis if available
    if 'probabilities' in translated_metrics and 'probabilities' in original_metrics:
        create_roc_comparison(translated_metrics, original_metrics, plots_dir)
    
    # Create simple bar chart for key metrics
    create_simple_bar_chart(translated_metrics, original_metrics, plots_dir)
    
    # Create difference plot
    create_difference_plot(translated_metrics, original_metrics, plots_dir)

def create_roc_comparison(translated_metrics, original_metrics, plots_dir):
    """Create ROC curve comparison if probabilities are available."""
    # This would require access to test labels and predicted probabilities
    # For now, create a placeholder
    pass

def create_simple_bar_chart(translated_metrics, original_metrics, plots_dir):
    """Create a simple, clean bar chart for the main metrics."""
    plt.figure(figsize=(10, 6))
    
    metrics = ['Accuracy', 'F1 Score']
    original_values = [original_metrics["accuracy"], original_metrics["f1"]]
    translated_values = [translated_metrics["accuracy"], translated_metrics["f1"]]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = plt.bar(x_pos - width/2, original_values, width, 
                   label='Original Data MLP', alpha=0.8, color='#2E86AB')
    bars2 = plt.bar(x_pos + width/2, translated_values, width,
                   label='Translated Data MLP', alpha=0.8, color='#A23B72')
    
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('P300 Classification Performance Comparison\n(MLPs trained on Original vs Translated Data)', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, metrics)
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    simple_plot_path = os.path.join(plots_dir, 'simple_comparison.png')
    plt.savefig(simple_plot_path, dpi=300, bbox_inches='tight')
    print(f"Simple comparison plot saved to: {simple_plot_path}")
    
    plt.close()

def create_difference_plot(translated_metrics, original_metrics, plots_dir):
    """Create a plot showing the differences between models."""
    plt.figure(figsize=(8, 6))
    
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    original_values = [original_metrics["accuracy"], original_metrics["f1"], 
                      original_metrics["precision"], original_metrics["recall"]]
    translated_values = [translated_metrics["accuracy"], translated_metrics["f1"], 
                        translated_metrics["precision"], translated_metrics["recall"]]
    
    differences = np.array(translated_values) - np.array(original_values)
    colors = ['green' if diff > 0 else 'red' for diff in differences]
    
    bars = plt.bar(metrics, differences, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Performance Difference\n(Translated - Original)', fontsize=12)
    plt.title('Translation Benefit Analysis', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.005 if height >= 0 else height - 0.01,
                f'{diff:+.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    diff_plot_path = os.path.join(plots_dir, 'difference_analysis.png')
    plt.savefig(diff_plot_path, dpi=300, bbox_inches='tight')
    print(f"Difference analysis plot saved to: {diff_plot_path}")
    
    plt.close()

def run_experiment(preprocess=False, translate=True, evaluate=True):
    """
    Run the P300 translation experiment.
    
    Args:
        preprocess: Whether to preprocess the data
        translate: Whether to translate source subjects' data
        evaluate: Whether to evaluate the models
    """
    print("\n=== P300 Translation Experiment ===\n")
    start_time = time.time()
    
    if preprocess:
        # Preprocess and partition data
        preprocess_and_partition_data()
    
    translated_data = None
    translator = None
    
    if translate:
        # Train Min2Net translator
        translator = train_min2net_translator()
        
        # Translate source subjects' data using translator
        translated_data = translate_source_subjects(translator)
        
        # Visualize the latent space to understand translation quality
        # visualize_latent_space(translator, ref_subject_id=config.REF_SUBJECT)
        
        # Extract features from translated data
        translated_features = extract_features_from_translated_data(translated_data)
        
        # Compare original vs translated waveforms for a random subject
        if translator is not None:
            # Compare for a few subjects to get better understanding
            for subject_id in range(2, min(5, config.NUM_SUBJECTS + 1)):
                compare_original_vs_translated(translator, subject_id=subject_id)
    
    if evaluate and translated_data is not None:
        # Load original features from all subjects for comparison
        original_features = load_original_features()
        
        # Train and evaluate MLP classifiers
        translated_metrics, original_metrics = train_and_evaluate_classifiers(translated_features, original_features)
        
        if translated_metrics is not None and original_metrics is not None:
            # Create comprehensive comparison plots
            plots_dir = create_comprehensive_comparison_plots(translated_metrics, original_metrics, config.RESULTS_DIR)
            
            # Print final comparison summary
            print("\n" + "="*60)
            print("FINAL EXPERIMENT SUMMARY")
            print("="*60)
            print(f"Original Data MLP    - F1: {original_metrics['f1']:.4f}, Acc: {original_metrics['accuracy']:.4f}")
            print(f"Translated Data MLP  - F1: {translated_metrics['f1']:.4f}, Acc: {translated_metrics['accuracy']:.4f}")
            print(f"F1 Score Improvement: {translated_metrics['f1'] - original_metrics['f1']:+.4f}")
            print(f"Accuracy Improvement: {translated_metrics['accuracy'] - original_metrics['accuracy']:+.4f}")
            
            if translated_metrics['f1'] > original_metrics['f1']:
                print("✅ Translation IMPROVED P300 classification performance")
            else:
                print("❌ Translation did NOT improve P300 classification performance")
            
            print(f"\n📊 Comprehensive plots saved to: {plots_dir}")
            print("="*60)
    
    end_time = time.time()
    print(f"\nExperiment completed in {end_time - start_time:.2f} seconds\n")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='P300 Translation Experiment')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess raw data')
    parser.add_argument('--translate', action='store_true', help='Translate data')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate classifiers')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # If no arguments are provided or --all is specified, run all steps
    if not (args.preprocess or args.translate or args.evaluate) or args.all:
        args.preprocess = True
        args.translate = True
        args.evaluate = True
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set up debug if requested
    if args.debug:
        import tensorflow as tf
        tf.get_logger().setLevel('INFO')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Create results directory if it doesn't exist
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    
    run_experiment(
        preprocess=args.preprocess,
        translate=args.translate,
        evaluate=args.evaluate
    )