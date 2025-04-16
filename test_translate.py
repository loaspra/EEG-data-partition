# python test_translate.py --data_path data/raw --log_path logs/translator --output_dir target_signals --visualize

import sys
sys.path.append("/home/loaspra/Code/PFC1/PFC1-Tema6-MIN2NET")

import tensorflow as tf
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import scipy.io as sio
from min2net.model import MIN2Net
import h5py

# Import from the training script
from train_p300_translator import (
    TranslatorMIN2Net, load_matlab_file, transform_data, 
    SAMPLE_RATE, SAMPLE_DURATION, INTENSIFIED_N_TIMES, MATRIX_DIMENSIONS, N_CHANNELS
)

def load_target_templates(log_path):
    """Load the target templates saved during training.
    
    Args:
        log_path (str): Path to the log directory containing target templates
        
    Returns:
        tuple: Target non-P300 and P300 templates
    """
    target_non_p300_path = os.path.join(log_path, "target_non_p300.npy")
    target_p300_path = os.path.join(log_path, "target_p300.npy")
    
    if not os.path.exists(target_non_p300_path) or not os.path.exists(target_p300_path):
        raise FileNotFoundError(f"Target templates not found in {log_path}. Run training first.")
    
    target_non_p300 = np.load(target_non_p300_path)
    target_p300 = np.load(target_p300_path)
    
    print(f"Loaded target templates - Non-P300: {target_non_p300.shape}, P300: {target_p300.shape}")
    return target_non_p300, target_p300

def prepare_test_data(subjects_data):
    """Prepare test data from subjects 2-8.
    
    Args:
        subjects_data (dict): Dictionary with data from all subjects
        
    Returns:
        tuple: X_test (input features) and labels (class labels)
    """
    # Count total number of samples from subjects 2-8
    total_non_p300_samples = 0
    total_p300_samples = 0
    
    for subject_id in range(2, 9):  # Subjects 2-8
        subject_key = f"A{subject_id:02d}"
        non_p300_data, p300_data = subjects_data[subject_key]
        total_non_p300_samples += non_p300_data.shape[1]
        total_p300_samples += p300_data.shape[1]
    
    # Get sample dimensions from the first subject's data
    non_p300_data, p300_data = subjects_data[f"A{2:02d}"]
    sample_length = non_p300_data.shape[0]
    n_channels = non_p300_data.shape[2]
    
    # Initialize arrays for input - adding D=1 dimension for MIN2Net
    X_non_p300 = np.zeros((total_non_p300_samples, 1, sample_length, n_channels))
    X_p300 = np.zeros((total_p300_samples, 1, sample_length, n_channels))
    
    # Create arrays to track which subject each sample came from
    subject_indices_non_p300 = np.zeros(total_non_p300_samples, dtype=np.int32)
    subject_indices_p300 = np.zeros(total_p300_samples, dtype=np.int32)
    
    # Fill the input arrays with data from subjects 2-8
    non_p300_idx = 0
    p300_idx = 0
    
    for subject_id in range(2, 9):  # Subjects 2-8
        subject_key = f"A{subject_id:02d}"
        non_p300_data, p300_data = subjects_data[subject_key]
        
        # Add non-P300 samples
        for i in range(non_p300_data.shape[1]):
            X_non_p300[non_p300_idx, 0] = non_p300_data[:, i, :]
            subject_indices_non_p300[non_p300_idx] = subject_id
            non_p300_idx += 1
        
        # Add P300 samples
        for i in range(p300_data.shape[1]):
            X_p300[p300_idx, 0] = p300_data[:, i, :]
            subject_indices_p300[p300_idx] = subject_id
            p300_idx += 1
    
    # Combine non-P300 and P300 data
    X_test = np.concatenate([X_non_p300, X_p300], axis=0)
    labels = np.zeros(X_test.shape[0])
    labels[total_non_p300_samples:] = 1  # 0 for non-P300, 1 for P300
    
    # Combine subject indices
    subject_indices = np.concatenate([subject_indices_non_p300, subject_indices_p300])
    
    # Keep track of original indices within each class
    original_indices = np.arange(X_test.shape[0])
    
    print(f"Test data prepared - X shape: {X_test.shape}")
    return X_test, labels, subject_indices, original_indices

def visualize_translation(original, translated, target, subject_id, is_p300, save_path=None):
    """Visualize the original signal, translated signal, and target template.
    
    Args:
        original (numpy.ndarray): Original signal
        translated (numpy.ndarray): Translated signal
        target (numpy.ndarray): Target template
        subject_id (int): Subject ID
        is_p300 (bool): Whether this is a P300 signal
        save_path (str, optional): Path to save the figure
    """
    # Create time vector (ms)
    time = np.arange(original.shape[0]) / SAMPLE_RATE * 1000
    
    # Plot for channel Pz (index 2), which typically shows the clearest P300
    channel_idx = 2  # Pz
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Original signal
    ax.plot(time, original[:, channel_idx], 'b-', alpha=0.5, label='Original')
    
    # Translated signal
    ax.plot(time, translated[:, channel_idx], 'g-', label='Translated')
    
    # Target template
    ax.plot(time, target[:, channel_idx], 'r--', label='Target Template')
    
    signal_type = "P300" if is_p300 else "Non-P300"
    ax.set_title(f"Subject {subject_id} - {signal_type} Signal Translation")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (μV)")
    ax.axvline(x=300, color='gray', linestyle='--')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()

def visualize_subject_translations(subject_id, originals, translations, target_templates, labels, output_dir):
    """Visualize translations for a specific subject.
    
    Args:
        subject_id (int): Subject ID
        originals (numpy.ndarray): Original signals from this subject
        translations (numpy.ndarray): Translated signals
        target_templates (tuple): Target templates (non-P300, P300)
        labels (numpy.ndarray): Class labels
        output_dir (str): Directory to save visualizations
    """
    # Create output directory for this subject
    subject_dir = os.path.join(output_dir, f"A{subject_id:02d}")
    os.makedirs(subject_dir, exist_ok=True)
    
    # Separate non-P300 and P300 samples
    non_p300_indices = np.where(labels == 0)[0]
    p300_indices = np.where(labels == 1)[0]
    
    # Get target templates
    target_non_p300, target_p300 = target_templates
    
    # Visualize a few non-P300 samples
    num_visualize = min(5, len(non_p300_indices))
    for i in range(num_visualize):
        idx = non_p300_indices[i]
        original = originals[idx, 0]  # Remove D dimension
        translated = translations[idx, 0]  # Remove D dimension
        
        save_path = os.path.join(subject_dir, f"non_p300_sample_{i}.png")
        visualize_translation(original, translated, target_non_p300, subject_id, False, save_path)
    
    # Visualize a few P300 samples
    num_visualize = min(5, len(p300_indices))
    for i in range(num_visualize):
        idx = p300_indices[i]
        original = originals[idx, 0]  # Remove D dimension
        translated = translations[idx, 0]  # Remove D dimension
        
        save_path = os.path.join(subject_dir, f"p300_sample_{i}.png")
        visualize_translation(original, translated, target_p300, subject_id, True, save_path)

def save_translations(X_test, translations, labels, subject_indices, output_dir):
    """Save translated signals to HDF5 format.
    
    Args:
        X_test (numpy.ndarray): Original test data
        translations (numpy.ndarray): Translated signals
        labels (numpy.ndarray): Class labels
        subject_indices (numpy.ndarray): Subject indices for each sample
        output_dir (str): Directory to save translations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create HDF5 file
    output_file = os.path.join(output_dir, "translated_signals.h5")
    with h5py.File(output_file, 'w') as f:
        # Save original signals
        f.create_dataset('original_signals', data=X_test)
        
        # Save translated signals
        f.create_dataset('translated_signals', data=translations)
        
        # Save metadata
        f.create_dataset('labels', data=labels)
        f.create_dataset('subject_indices', data=subject_indices)
    
    print(f"Saved translations to {output_file}")
    
    # Save individual MAT files for each subject
    for subject_id in range(2, 9):
        # Get indices for this subject
        subject_mask = subject_indices == subject_id
        
        if np.sum(subject_mask) > 0:
            subject_file = os.path.join(output_dir, f"A{subject_id:02d}_translated.mat")
            
            # Extract data for this subject
            subject_originals = X_test[subject_mask]
            subject_translations = translations[subject_mask]
            subject_labels = labels[subject_mask]
            
            # Save to MAT file
            sio.savemat(subject_file, {
                'original_signals': subject_originals,
                'translated_signals': subject_translations,
                'labels': subject_labels
            })
            
            print(f"Saved translations for subject A{subject_id:02d} to {subject_file}")

def main():
    parser = argparse.ArgumentParser(description='Test P300 signal translator model')
    parser.add_argument('--data_path', type=str, default='data/raw', help='Path to raw MATLAB data files')
    parser.add_argument('--log_path', type=str, default='logs/translator', help='Path to model logs and weights')
    parser.add_argument('--output_dir', type=str, default='target_signals', help='Directory to save translated signals')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension (must match training)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of translations')
    parser.add_argument('--model_path', type=str, default=None, help='Specific path to the model file or weights')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {tf.config.experimental.list_physical_devices('GPU')}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load target templates
    target_templates = load_target_templates(args.log_path)
    
    # Load data for all subjects
    subjects_data = {}
    print("Loading and processing data for all subjects...")
    
    subject_files = [f"A{i:02d}.mat" for i in range(1, 9)]  # A01.mat to A08.mat
    
    for file in subject_files:
        subject_id = file.replace(".mat", "")
        data_path = os.path.join(args.data_path, file)
        
        print(f"Processing {subject_id}...")
        matlab_data = load_matlab_file(data_path)['data'][0]
        non_p300_data, p300_data = transform_data(matlab_data)
        
        subjects_data[subject_id] = (non_p300_data, p300_data)
    
    # Prepare test data
    print("\nPreparing test data...")
    X_test, labels, subject_indices, original_indices = prepare_test_data(subjects_data)
    
    # Create model for inference
    print("\nLoading model for inference...")
    input_shape = X_test.shape[1:]  # (1, T, C)
    
    # Try to load the full model first
    full_model_path = args.model_path if args.model_path else os.path.join(args.log_path, "P300_translator_full_model.h5")
    
    if os.path.exists(full_model_path):
        print(f"Loading full model from {full_model_path}")
        # Load the full model directly
        try:
            from tensorflow.keras.models import load_model
            model = load_model(full_model_path, compile=False)
            
            # Run inference
            print("\nRunning inference to translate signals...")
            translations, _ = model.predict(X_test, batch_size=args.batch_size)
        except Exception as e:
            print(f"Error loading full model: {e}")
            print("Falling back to loading weights instead...")
            # Fall back to weights loading
            use_full_model = False
    else:
        print(f"Full model not found at {full_model_path}")
        print("Falling back to loading weights instead...")
        use_full_model = False
    
    # If full model loading failed, try with weights
    if not os.path.exists(full_model_path) or 'use_full_model' in locals() and not use_full_model:
        # Create model with same architecture as in training
        model = TranslatorMIN2Net(
            input_shape=input_shape,
            class_balancing=False,
            num_class=2,
            loss=None,  # Not used for inference
            loss_weights=None,  # Not used for inference
            epochs=1,  # Not used for inference
            batch_size=args.batch_size,
            optimizer=Adam(),
            latent_dim=args.latent_dim,
            log_path=args.log_path,
            model_name="P300_translator"
        )
        
        # Run inference
        print("\nRunning inference to translate signals...")
        translations = model.predict_translation(X_test)
    
    # Save the translations
    print("\nSaving translated signals...")
    save_translations(X_test, translations, labels, subject_indices, args.output_dir)
    
    # Generate visualizations
    if args.visualize:
        print("\nGenerating visualizations...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        for subject_id in range(2, 9):
            # Get indices for this subject
            subject_mask = subject_indices == subject_id
            
            if np.sum(subject_mask) > 0:
                subject_X = X_test[subject_mask]
                subject_translations = translations[subject_mask]
                subject_labels = labels[subject_mask]
                
                visualize_subject_translations(
                    subject_id,
                    subject_X,
                    subject_translations,
                    target_templates,
                    subject_labels,
                    vis_dir
                )
    
    print("\nTranslation completed!")

if __name__ == "__main__":
    main() 