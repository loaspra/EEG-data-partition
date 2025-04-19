#!/usr/bin/env python
"""
Test script for SimpleMin2Net class
"""

import os
import numpy as np
import tensorflow as tf
from simple_min2net import SimpleMin2Net

# Print versions to verify environment
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

# Create a small random dataset for testing
def create_test_data(n_trials=100, n_samples=204, n_channels=8):
    """Create random test data"""
    # Source domain data (subjects 2-8)
    X_source = np.random.randn(n_trials, 1, n_samples, n_channels)
    y_source = np.random.randint(0, 2, size=(n_trials,))
    
    # Target domain data (subject 1)
    X_target = np.random.randn(n_trials // 2, 1, n_samples, n_channels)
    y_target = np.random.randint(0, 2, size=(n_trials // 2,))
    
    return (X_source, y_source), (X_target, y_target)

def main():
    """Test SimpleMin2Net class"""
    print("Creating test data...")
    source_data, target_data = create_test_data()
    
    print("Source data shape:", source_data[0].shape)
    print("Target data shape:", target_data[0].shape)
    
    # Create model
    print("Creating SimpleMin2Net model...")
    model = SimpleMin2Net(
        input_shape=(1, 204, 8),
        latent_dim=32,
        epochs=2,  # Just for testing
        batch_size=16,
        log_path='./test_models'
    )
    
    # Build model to check architecture
    print("Building model...")
    model.build()
    
    # Attempt to train for a small number of epochs
    print("Testing model fitting (2 epochs)...")
    try:
        model.fit(source_data, target_data)
        print("Model training successful!")
    except Exception as e:
        print(f"Error during model training: {e}")

if __name__ == "__main__":
    main() 