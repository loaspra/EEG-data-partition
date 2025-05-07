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