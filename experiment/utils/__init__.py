"""
Utilities package for P300 Translation Experiment.
This package contains utility functions for data processing, model training, and visualization.
"""

from utils.data_utils import (
    prepare_eeg_for_min2net,
    load_subject_data,
    load_features,
    extract_features,
    load_original_features,
    preprocess_and_partition_data
)

from utils.model_utils import (
    train_min2net_translator,
    translate_source_subjects,
    extract_features_from_translated_data,
    train_and_evaluate_classifiers
)

from utils.visualization import (
    compare_original_vs_translated,
    plot_training_curves,
    visualize_feature_importance
)

__all__ = [
    # Data utilities
    'prepare_eeg_for_min2net',
    'load_subject_data',
    'load_features',
    'extract_features',
    'load_original_features',
    'preprocess_and_partition_data',
    
    # Model utilities
    'train_min2net_translator',
    'translate_source_subjects',
    'extract_features_from_translated_data',
    'train_and_evaluate_classifiers',
    
    # Visualization utilities
    'compare_original_vs_translated',
    'plot_training_curves',
    'visualize_feature_importance'
] 