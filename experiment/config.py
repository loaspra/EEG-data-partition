"""
Configuration file for P300 Translation Experiment.
"""

import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PARTITIONED_DATA_DIR = os.path.join(DATA_DIR, "partitioned")
RESULTS_DIR = os.path.join(ROOT_DIR, "experiment", "results")
MODEL_DIR = os.path.join(ROOT_DIR, "experiment", "models")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Data parameters
NUM_SUBJECTS = 8
REF_SUBJECT = 1  # Reference subject for translation
EEG_CHANNELS = ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
NUM_CHANNELS = len(EEG_CHANNELS)
SAMPLING_RATE = 256
TIME_WINDOW = [0, 800]  # ms post-stimulus
FILTER_RANGE = [0.1, 30]  # Hz bandpass filter

# SimpleMin2Net translation parameters
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 64
LEARNING_RATE = 0.001

# SimpleMin2Net parameters
MIN2NET_INPUT_SHAPE = (1, 204, 8)  # (trials, samples, channels)
MIN2NET_BATCH_SIZE = 64
MIN2NET_EPOCHS = 100
MIN2NET_PATIENCE = 10
MIN2NET_LEARNING_RATE = 0.001
MIN2NET_ENCODER_UNITS = 64

# MLP classifier parameters
MLP_HIDDEN_UNITS = 64
MLP_BATCH_SIZE = 32
MLP_EPOCHS = 100
MLP_DROPOUT = 0.5
MLP_LEARNING_RATE = 0.0005

# Feature extraction parameters
FEATURE_EXTRACTION_METHODS = [
    "statistical",
    "temporal",
    "hjorth",
    "zcr",
    "line_length"
]

# Evaluation parameters
CROSS_VAL_FOLDS = 5
RANDOM_SEED = 42

# Reproducibility setup
def set_global_seed(seed=None):
    """
    Set global random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed to use. If None, uses RANDOM_SEED from config.
    """
    if seed is None:
        seed = RANDOM_SEED
    
    import numpy as np
    import random
    import os
    
    # Set seeds for basic random number generators
    random.seed(seed)
    np.random.seed(seed)
    
    # Set additional environment variables for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # Try to import and configure TensorFlow if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # Configure TensorFlow for reproducibility if available
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            # Older TensorFlow versions may not have this function
            pass
        
        print(f"Global random seed set to: {seed} (TensorFlow configured)")
    except ImportError:
        print(f"Global random seed set to: {seed} (TensorFlow not available)")

# Call the function to set seeds immediately when config is imported
set_global_seed()