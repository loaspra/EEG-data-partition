"""
Main script for running the P300 Translation Experiment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

# Import config and utils
import config
from utils import (
    preprocess_and_partition_data,
    train_min2net_translator,
    translate_source_subjects,
    extract_features_from_translated_data,
    train_and_evaluate_classifiers,
    compare_original_vs_translated,
    load_original_features
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
        
        # Extract features from translated data
        translated_features = extract_features_from_translated_data(translated_data)
        
        # Compare original vs translated waveforms for a random subject
        if translator is not None:
            compare_original_vs_translated(translator)
    
    if evaluate and translated_data is not None:
        # Load original features from all subjects for comparison
        original_features = load_original_features()
        
        # Train and evaluate MLP classifiers
        translated_metrics, original_metrics = train_and_evaluate_classifiers(translated_features, original_features)
    
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
    
    args = parser.parse_args()
    
    # If no arguments are provided or --all is specified, run all steps
    if not (args.preprocess or args.translate or args.evaluate) or args.all:
        args.preprocess = True
        args.translate = True
        args.evaluate = True
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(
        preprocess=args.preprocess,
        translate=args.translate,
        evaluate=args.evaluate
    ) 