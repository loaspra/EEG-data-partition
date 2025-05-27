"""
Main script for running the P300 Translation Experiment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

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
            # Print comparison summary
            print("\nFinal Experiment Summary:")
            print(f"Translation {'improved' if translated_metrics['f1'] > original_metrics['f1'] else 'did not improve'} P300 classification")
            print(f"F1 Score Difference: {translated_metrics['f1'] - original_metrics['f1']:.4f}")
    
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