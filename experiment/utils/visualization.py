"""
Visualization utilities for P300 Translation Experiment.
Functions for visualizing results and comparing models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import config
from utils.data_utils import load_subject_data, prepare_eeg_for_min2net

def compare_original_vs_translated(translator, subject_id=None):
    """
    Compare waveforms from original and translated data.
    
    Args:
        translator (SimpleMin2Net): Trained translator model
        subject_id (int): Subject ID to compare, if None, selects a random source subject
        
    Returns:
        None
    """
    if subject_id is None:
        subject_id = np.random.randint(2, config.NUM_SUBJECTS + 1)
    
    print(f"=== Comparing Original vs. Translated Waveforms for Subject {subject_id} ===")
    
    # Load test data for the selected subject
    X_test, y_test = load_subject_data(subject_id, 'test')
    
    # Prepare for SimpleMin2Net and translate
    X_test_min2net = prepare_eeg_for_min2net(X_test)
    X_translated_min2net = translator.translate(X_test_min2net)
    
    # Convert back from Min2Net format
    n_trials = X_translated_min2net.shape[0]
    n_channels = X_translated_min2net.shape[3]
    n_samples = X_translated_min2net.shape[2]
    
    X_translated = np.zeros((n_trials, n_channels, n_samples))
    for i in range(n_trials):
        X_translated[i] = X_translated_min2net[i, 0].T
    
    # Load reference subject data for comparison
    ref_X_test, ref_y_test = load_subject_data(config.REF_SUBJECT, 'test')
    
    # Find P300 trials
    p300_indices_source = np.where(y_test == 1)[0]
    p300_indices_ref = np.where(ref_y_test == 1)[0]
    
    if len(p300_indices_source) == 0 or len(p300_indices_ref) == 0:
        print("No P300 trials found for comparison.")
        return
    
    # Select a random P300 trial
    p300_idx_source = np.random.choice(p300_indices_source)
    p300_idx_ref = np.random.choice(p300_indices_ref)
    
    # Plot comparison
    time_points = np.arange(X_test.shape[2]) / config.SAMPLING_RATE * 1000  # Convert to milliseconds
    
    plt.figure(figsize=(15, 10))
    
    # Plot for each channel
    for ch in range(n_channels):
        plt.subplot(2, 4, ch+1)
        
        # Original source subject data
        plt.plot(time_points, X_test[p300_idx_source, ch], 'b-', label=f'Original (Subject {subject_id})')
        
        # Translated data
        plt.plot(time_points, X_translated[p300_idx_source, ch], 'r-', label='Translated')
        
        # Reference subject data
        plt.plot(time_points, ref_X_test[p300_idx_ref, ch], 'g-', label=f'Reference (Subject {config.REF_SUBJECT})')
        
        plt.title(f'Channel {config.EEG_CHANNELS[ch]}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        if ch == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f'waveform_comparison_subject_{subject_id}.png'))
    plt.close()
    
    print(f"Waveform comparison saved to {os.path.join(config.RESULTS_DIR, f'waveform_comparison_subject_{subject_id}.png')}")

def plot_training_curves(history):
    """
    Plot training and validation curves for a model.
    
    Args:
        history: Training history dictionary with keys 'loss', 'val_loss', etc.
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
    elif 'acc' in history:
        plt.plot(history['acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'training_curves.png'))
    plt.close()
    
    print(f"Training curves saved to {os.path.join(config.RESULTS_DIR, 'training_curves.png')}")
    
def visualize_feature_importance(model, feature_names=None):
    """
    Visualize feature importance if the model supports it.
    
    Args:
        model: Trained classifier model
        feature_names: List of feature names
        
    Returns:
        None
    """
    # This function would need to be adapted to the specific model type
    # Here's a placeholder for visualization
    
    # Example for models with feature_importances_ attribute
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices], align='center')
        
        if feature_names is not None:
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        else:
            plt.xticks(range(len(importances)), indices)
            
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_DIR, 'feature_importance.png'))
        plt.close()
        
        print(f"Feature importance visualization saved to {os.path.join(config.RESULTS_DIR, 'feature_importance.png')}")
    else:
        print("This model doesn't support feature importance visualization.") 