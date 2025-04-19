#!/bin/bash

# P300 Translation Experiment Setup Script

echo "Setting up environment for P300 Translation Experiment..."

# Create conda environment
echo "Creating conda environment..."
conda create -n p300_translation python=3.6.9 -y

# Activate environment
echo "Activating environment..."
source activate p300_translation

# Install TensorFlow and dependencies
echo "Installing TensorFlow and dependencies..."
pip install tensorflow-gpu==2.2.0
pip install tensorflow-addons==0.9.1

# Install EEG processing libraries
echo "Installing EEG processing libraries..."
pip install mne
pip install scikit-learn>=0.24.1
pip install numpy scipy pandas

# Install visualization libraries
echo "Installing visualization libraries..."
pip install matplotlib
pip install seaborn

# Install additional dependencies
echo "Installing additional dependencies..."
pip install wget>=3.2

# Create necessary directories
echo "Creating directories..."
mkdir -p ../data/raw
mkdir -p ../data/processed
mkdir -p ../data/partitioned
mkdir -p models
mkdir -p results

echo "Setup complete!"
echo "To activate the environment, run: conda activate p300_translation"
echo "To run the experiment, run: python run_experiment.py --all" 