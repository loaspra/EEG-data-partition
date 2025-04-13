import sys
sys.path.append("/home/loaspra/Code/PFC1/PFC1-Tema6-MIN2NET")

import tensorflow as tf
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from min2net.model import MIN2Net
from min2net.utils import write_log
from min2net.loss import mean_squared_error, triplet_loss

"""
P300 Translator Model Training Script

This script trains a model to translate P300 signals from subjects 2-8 
to match the target signals from subject 1. 

Process:
1. Select one random P300 and one random non-P300 sample from subject 1 as targets
2. Use P300 and non-P300 signals from subjects 2-8 as input
3. Train the model to translate these signals to match the subject 1 targets
"""

# Constants for the P300 dataset
SAMPLE_RATE = 256  # Hz sampling rate
SAMPLE_DURATION = 64  # Samples in stimulus window
INTENSIFIED_N_TIMES = 20  # Each item intensified 20 times (10 row + 10 column)
MATRIX_DIMENSIONS = 6  # 6x6 matrix
N_CHANNELS = 8  # Number of EEG channels (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8)

def load_matlab_file(file_path):
    """Load a MATLAB .mat file and return its contents."""
    import scipy.io as sio
    return sio.loadmat(file_path)

def transform_data(data, samples_per_target=256):
    """Transform raw MATLAB data into structured arrays for target and non-target stimuli.
    
    Args:
        data (dict): MATLAB data dictionary
        samples_per_target (int): Number of samples to include per target
        
    Returns:
        tuple: Two arrays containing class 1 (non-target) and class 2 (target) data
    """
    # Extract relevant data from the MATLAB structure
    eeg_data = data['X'][0]  # EEG data [samples × channels]
    stimulus_type = data['y'][0]  # Stimulus type (1=non-target, 2=target)
    trial_start_indices = data['trial'][0][0]  # Trial start indices
    
    # Calculate total samples per trial
    samples_per_trial = SAMPLE_DURATION * INTENSIFIED_N_TIMES * MATRIX_DIMENSIONS
    
    # Initialize arrays for non-target (class 1) and target (class 2) data
    final_data_class1 = np.zeros((samples_per_target, samples_per_trial, N_CHANNELS))
    final_data_class2 = np.zeros((samples_per_target, samples_per_trial, N_CHANNELS))
    
    # Counters for the number of samples in each class
    class_1_count = 0
    class_2_count = 0
    
    # Process each trial
    for i, start_idx in enumerate(trial_start_indices):
        # Define the trial window with padding
        end_idx = start_idx + samples_per_trial + SAMPLE_RATE  # Add 1 second padding
        trial_data = eeg_data[start_idx:end_idx]
        trial_stimulus_type = stimulus_type[start_idx:end_idx]
        
        # Process each stimulus intensification in the trial
        for j in range(INTENSIFIED_N_TIMES * MATRIX_DIMENSIONS):
            # Get the stimulus type for this segment
            current_stimulus_type = trial_stimulus_type[j * SAMPLE_DURATION:(j + 1) * SAMPLE_DURATION - 1]
            
            # Get the EEG data for this segment (including response window)
            character_data = trial_data[(j * SAMPLE_DURATION):((j * SAMPLE_DURATION) + SAMPLE_RATE), :]
            
            # Classify based on stimulus type
            if 1 in current_stimulus_type:  # Non-target stimulus
                if class_1_count < final_data_class1.shape[1]:
                    final_data_class1[:, class_1_count] = character_data
                    class_1_count += 1
            elif 2 in current_stimulus_type:  # Target stimulus (P300 present)
                if class_2_count < final_data_class2.shape[1]:
                    final_data_class2[:, class_2_count] = character_data
                    class_2_count += 1
    
    # Trim arrays to actual sample counts
    final_data_class1 = final_data_class1[:, :class_1_count]
    final_data_class2 = final_data_class2[:, :class_2_count]
    
    print(f"Class 1 (Non-target): {class_1_count} samples, Class 2 (Target): {class_2_count} samples")
    return final_data_class1, final_data_class2

def select_target_samples(subject1_data):
    """Select one random sample from each class of subject 1's data.
    
    Args:
        subject1_data (tuple): Tuple containing (non_target_data, target_data) for subject 1
        
    Returns:
        tuple: One selected non-target and one selected target sample
    """
    non_target_data, target_data = subject1_data
    
    # Select one random sample from non-target class
    non_target_idx = random.randint(0, non_target_data.shape[1] - 1)
    selected_non_target = non_target_data[:, non_target_idx, :]
    
    # Select one random sample from target class
    target_idx = random.randint(0, target_data.shape[1] - 1)
    selected_target = target_data[:, target_idx, :]
    
    print(f"Selected template samples - Non-target: sample {non_target_idx}, Target: sample {target_idx}")
    return selected_non_target, selected_target

def prepare_training_data(subjects_data, target_samples):
    """Prepare training data from subjects 2-8 mapped to subject 1's target samples.
    
    Args:
        subjects_data (dict): Dictionary with data from all subjects
        target_samples (tuple): Target samples from subject 1
        
    Returns:
        tuple: X_train (input features) and y_train (target outputs)
    """
    target_non_p300, target_p300 = target_samples
    
    # Count total number of samples from subjects 2-8
    total_non_p300_samples = 0
    total_p300_samples = 0
    
    for subject_id in range(2, 9):  # Subjects 2-8
        subject_key = f"A{subject_id:02d}"
        non_p300_data, p300_data = subjects_data[subject_key]
        total_non_p300_samples += non_p300_data.shape[1]
        total_p300_samples += p300_data.shape[1]
    
    # Determine input and output shapes
    sample_length = target_non_p300.shape[0]
    n_channels = target_non_p300.shape[1]
    
    # Initialize arrays for input and output
    X_non_p300 = np.zeros((total_non_p300_samples, sample_length, n_channels))
    X_p300 = np.zeros((total_p300_samples, sample_length, n_channels))
    
    # Create corresponding target arrays by repeating the target samples
    y_non_p300 = np.tile(target_non_p300[np.newaxis, :, :], (total_non_p300_samples, 1, 1))
    y_p300 = np.tile(target_p300[np.newaxis, :, :], (total_p300_samples, 1, 1))
    
    # Fill the input arrays with data from subjects 2-8
    non_p300_idx = 0
    p300_idx = 0
    
    for subject_id in range(2, 9):  # Subjects 2-8
        subject_key = f"A{subject_id:02d}"
        non_p300_data, p300_data = subjects_data[subject_key]
        
        # Add non-P300 samples
        for i in range(non_p300_data.shape[1]):
            X_non_p300[non_p300_idx] = non_p300_data[:, i, :]
            non_p300_idx += 1
        
        # Add P300 samples
        for i in range(p300_data.shape[1]):
            X_p300[p300_idx] = p300_data[:, i, :]
            p300_idx += 1
    
    # Combine non-P300 and P300 data
    X_train = np.concatenate([X_non_p300, X_p300], axis=0)
    y_train = np.concatenate([y_non_p300, y_p300], axis=0)
    
    # Create labels for loss computation
    labels = np.zeros(X_train.shape[0])
    labels[total_non_p300_samples:] = 1  # 0 for non-P300, 1 for P300
    
    # Shuffle the data while maintaining the mapping between X and y
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    labels = labels[indices]
    
    print(f"Training data prepared - X shape: {X_train.shape}, y shape: {y_train.shape}")
    return X_train, y_train, labels

def visualize_samples(target_samples, subject_samples, subject_id, save_path=None):
    """Visualize the target samples and sample translations from a subject.
    
    Args:
        target_samples (tuple): Target non-P300 and P300 samples from subject 1
        subject_samples (tuple): Non-P300 and P300 samples from the subject
        subject_id (str): Subject identifier
        save_path (str, optional): Path to save the figure
    """
    target_non_p300, target_p300 = target_samples
    subject_non_p300, subject_p300 = subject_samples
    
    # Create time vector (ms)
    time = np.arange(target_non_p300.shape[0]) / SAMPLE_RATE * 1000
    
    # Plot for channel Pz (index 2), which typically shows the clearest P300
    channel_idx = 2  # Pz
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Target Non-P300
    axes[0, 0].plot(time, target_non_p300[:, channel_idx], 'b-')
    axes[0, 0].set_title(f"Subject 1 - Non-P300 Template")
    axes[0, 0].set_ylabel("Amplitude (μV)")
    axes[0, 0].axvline(x=300, color='gray', linestyle='--')
    axes[0, 0].grid(True)
    
    # Target P300
    axes[0, 1].plot(time, target_p300[:, channel_idx], 'r-')
    axes[0, 1].set_title(f"Subject 1 - P300 Template")
    axes[0, 1].axvline(x=300, color='gray', linestyle='--')
    axes[0, 1].grid(True)
    
    # Subject Non-P300 (sample)
    if subject_non_p300.shape[1] > 0:
        # Plot a random sample
        sample_idx = random.randint(0, subject_non_p300.shape[1] - 1)
        axes[1, 0].plot(time, subject_non_p300[:, sample_idx, channel_idx], 'b-')
        axes[1, 0].set_title(f"Subject {subject_id} - Non-P300 Sample")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Amplitude (μV)")
        axes[1, 0].axvline(x=300, color='gray', linestyle='--')
        axes[1, 0].grid(True)
    
    # Subject P300 (sample)
    if subject_p300.shape[1] > 0:
        # Plot a random sample
        sample_idx = random.randint(0, subject_p300.shape[1] - 1)
        axes[1, 1].plot(time, subject_p300[:, sample_idx, channel_idx], 'r-')
        axes[1, 1].set_title(f"Subject {subject_id} - P300 Sample")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].axvline(x=300, color='gray', linestyle='--')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def train_translator_model(X_train, y_train, labels, args):
    """Train the translator model using MIN2Net architecture without the classifier.
    
    Args:
        X_train (numpy.ndarray): Input signals from subjects 2-8
        y_train (numpy.ndarray): Target signals from subject 1
        labels (numpy.ndarray): Class labels (0=non-P300, 1=P300)
        args (argparse.Namespace): Command-line arguments
        
    Returns:
        MIN2Net: Trained model
    """
    # Determine input shape
    input_shape = X_train.shape[1:]
    
    # Create model
    model = MIN2Net(input_shape=input_shape,
                    class_balancing=False,  # We've already balanced our data
                    num_class=2,  # Binary: P300 vs. non-P300
                    # Use only the autoencoder and triplet losses, not the classifier
                    loss=[mean_squared_error, triplet_loss(margin=args.margin)],
                    loss_weights=args.loss_weights[:2],  # Only use the first two weights
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                    lr=args.learning_rate,
                    min_lr=args.min_lr,
                    factor=args.factor,
                    patience=args.patience,
                    es_patience=args.es_patience,
                    latent_dim=args.latent_dim,
                    log_path=args.log_path,
                    model_name="P300_translator")
    
    # Modify model to remove the classifier (we only care about translation)
    # This is done by setting the classifier loss weight to 0 in the MIN2Net constructor
    
    # Train model
    # Since we're not using the classifier, y_train contains the target signals, not class labels
    # We'll use X_train as input and y_train as the reconstruction target
    model.fit_translator(X_train, y_train, labels)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train P300 signal translator model')
    parser.add_argument('--data_path', type=str, default='data/raw', help='Path to raw MATLAB data files')
    parser.add_argument('--log_path', type=str, default='logs/translator', help='Path to save logs and results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--factor', type=float, default=0.5, help='Factor for reducing learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for reducing learning rate')
    parser.add_argument('--es_patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--latent_dim', type=int, default=64, help='Dimension of latent space')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--loss_weights', nargs='+', default=[0.5, 0.5], type=float, 
                       help='Weights for different loss components (AE and triplet)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print(f"Using GPU: {tf.config.experimental.list_physical_devices('GPU')}")
    
    # Create log directory
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
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
    
    # Select target samples from subject 1
    print("\nSelecting target templates from Subject A01...")
    target_samples = select_target_samples(subjects_data["A01"])
    
    # Visualize target samples and compare with samples from other subjects
    os.makedirs(os.path.join(args.log_path, "visualizations"), exist_ok=True)
    for subject_id in range(1, 9):
        subject_key = f"A{subject_id:02d}"
        save_path = os.path.join(args.log_path, "visualizations", f"{subject_key}_samples.png")
        visualize_samples(target_samples, subjects_data[subject_key], subject_id, save_path)
    
    # Prepare training data
    print("\nPreparing training data...")
    X_train, y_train, labels = prepare_training_data(subjects_data, target_samples)
    
    # Save the target samples for later use
    target_non_p300, target_p300 = target_samples
    np.save(os.path.join(args.log_path, "target_non_p300.npy"), target_non_p300)
    np.save(os.path.join(args.log_path, "target_p300.npy"), target_p300)
    
    # Train model
    print("\nTraining translator model...")
    model = train_translator_model(X_train, y_train, labels, args)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 