"""
Visualization utilities for P300 Translation Experiment.
Functions for visualizing results and comparing models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    
    # Create a figure with two subplots - one for all channels, one for P300 averages
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Plot individual channels
    for ch in range(n_channels):
        plt.subplot(3, 3, ch+1)
        
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
    
    # 2. Plot average of all P300 trials to see translation quality
    plt.subplot(3, 3, 9)
    
    # Calculate averages for P300 trials
    original_p300_avg = np.mean(X_test[p300_indices_source], axis=(0, 1))
    translated_p300_avg = np.mean(X_translated[p300_indices_source], axis=(0, 1))
    reference_p300_avg = np.mean(ref_X_test[p300_indices_ref], axis=(0, 1))
    
    plt.plot(time_points, original_p300_avg, 'b-', label=f'Original Avg (Subject {subject_id})')
    plt.plot(time_points, translated_p300_avg, 'r-', label='Translated Avg')
    plt.plot(time_points, reference_p300_avg, 'g-', label=f'Reference Avg (Subject {config.REF_SUBJECT})')
    
    plt.title('Average P300 Waveform Across All Channels')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f'waveform_comparison_subject_{subject_id}.png'))
    plt.close()
    
    print(f"Waveform comparison saved to {os.path.join(config.RESULTS_DIR, f'waveform_comparison_subject_{subject_id}.png')}")
    
    # 3. Create heatmap of the difference between translated and reference signals
    fig = plt.figure(figsize=(12, 8))
    
    # Average across trials
    diff_map = np.mean(np.abs(X_translated[p300_indices_source] - ref_X_test[p300_indices_ref].mean(axis=0)), axis=0)
    
    # Create a heatmap of channel x time differences
    ax = plt.subplot(111)
    im = ax.imshow(diff_map, aspect='auto', cmap='viridis')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # Set labels
    ax.set_title('Absolute Difference: Translated vs Reference P300 Signals')
    ax.set_xlabel('Time Samples')
    ax.set_ylabel('Channels')
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(config.EEG_CHANNELS)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f'translation_difference_map_subject_{subject_id}.png'))
    plt.close()
    
    print(f"Translation difference map saved to {os.path.join(config.RESULTS_DIR, f'translation_difference_map_subject_{subject_id}.png')}")

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

def visualize_latent_space(translator, ref_subject_id=1):
    """
    Visualize the latent space of the translator to see if subjects are properly aligned.
    
    Args:
        translator: Trained translator model
        ref_subject_id: Reference subject ID (default: 1)
        
    Returns:
        None
    """
    print("Visualizing latent space representation...")
    
    # Collect data from all subjects
    all_data = []
    all_labels = []  # For subject identification
    all_p300_labels = []  # For P300 vs non-P300 distinction
    
    # Reference subject
    ref_X, ref_y = load_subject_data(ref_subject_id, 'test')
    ref_X_min2net = prepare_eeg_for_min2net(ref_X)
    
    # Get latent representations
    ref_latent = translator.encoder.predict(ref_X_min2net)
    
    all_data.append(ref_latent)
    all_labels.extend([f"Ref-{ref_subject_id}"] * len(ref_latent))
    all_p300_labels.extend(ref_y)
    
    # Source subjects
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        src_X, src_y = load_subject_data(subject_id, 'test')
        src_X_min2net = prepare_eeg_for_min2net(src_X)
        
        # Original latent representation
        src_latent_orig = translator.encoder.predict(src_X_min2net)
        all_data.append(src_latent_orig)
        all_labels.extend([f"Src-{subject_id}"] * len(src_latent_orig))
        all_p300_labels.extend(src_y)
        
        # Translated latent representation
        src_latent_trans = translator.encoder.predict(translator.translate(src_X_min2net))
        all_data.append(src_latent_trans)
        all_labels.extend([f"Trans-{subject_id}"] * len(src_latent_trans))
        all_p300_labels.extend(src_y)
    
    # Combine all data
    all_data = np.vstack(all_data)
    
    # Reduce dimensionality for visualization
    # First PCA to handle high dimensionality, then t-SNE for better visualization
    pca = PCA(n_components=min(50, all_data.shape[1]))
    data_pca = pca.fit_transform(all_data)
    
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot by subject
    plt.subplot(1, 2, 1)
    for subject in set(all_labels):
        idx = np.array(all_labels) == subject
        color = 'g' if 'Ref' in subject else ('r' if 'Trans' in subject else 'b')
        marker = 'o' if 'Ref' in subject else ('x' if 'Trans' in subject else '+')
        plt.scatter(data_tsne[idx, 0], data_tsne[idx, 1], c=color, marker=marker, label=subject, alpha=0.7)
    
    plt.title('Latent Space by Subject')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot by P300 label
    plt.subplot(1, 2, 2)
    for label, name, color in zip([0, 1], ['Non-P300', 'P300'], ['blue', 'red']):
        idx = np.array(all_p300_labels) == label
        plt.scatter(data_tsne[idx, 0], data_tsne[idx, 1], c=color, label=name, alpha=0.7)
    
    plt.title('Latent Space by P300 Label')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, 'latent_space_visualization.png'))
    plt.close()
    
    print(f"Latent space visualization saved to {os.path.join(config.RESULTS_DIR, 'latent_space_visualization.png')}") 