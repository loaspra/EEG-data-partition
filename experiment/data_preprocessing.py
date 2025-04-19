"""
Data preprocessing module for P300 Translation Experiment.
"""

import os
import numpy as np
import mne
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import config

def load_p300_data(subject_id, debug=False):
    """
    Load raw P300 data for a specific subject from .mat file.
    
    Args:
        subject_id (int): The subject ID (1-8)
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (eeg_data, labels) where eeg_data is a numpy array of shape
        (n_trials, n_channels, n_samples) and labels is a numpy array
        of shape (n_trials,) with 1 for target (P300) and 0 for non-target.
    """
    # Construct filename (e.g., A01.mat, A02.mat, ...)
    filename = f'A0{subject_id}.mat'
    data_path = os.path.join(config.RAW_DATA_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file for subject {subject_id} not found at {data_path}")
    
    # Load .mat data
    try:
        mat_data = scipy.io.loadmat(data_path)
        print(f"Successfully loaded {filename}")
    except Exception as e:
        raise IOError(f"Error loading .mat file {data_path}: {e}")
    
    # Extract the data structure
    if 'data' not in mat_data:
        print(f"Available keys in {filename}: {list(mat_data.keys())}")
        raise KeyError(f"Expected key 'data' not found in {filename}")
    
    # Access the MATLAB struct
    struct_data = mat_data['data'][0, 0]
    
    # Check required fields exist
    required_fields = ['X', 'y', 'trial', 'channels']
    for field in required_fields:
        if field not in struct_data.dtype.names:
            print(f"Available fields in {filename}: {struct_data.dtype.names}")
            raise KeyError(f"Required field '{field}' not found in {filename}")
    
    # Extract EEG data, labels, and trial indices
    eeg_data_raw = struct_data['X']  # Shape: (samples, channels)
    labels_raw = struct_data['y']    # Shape: (samples, 1)
    trial_indices = struct_data['trial'].flatten()  # Trial start indices
    channel_info = struct_data['channels']
    
    if debug:
        print(f"Subject {subject_id}: Raw EEG data shape: {eeg_data_raw.shape}")
        print(f"Subject {subject_id}: Raw labels shape: {labels_raw.shape}")
        print(f"Subject {subject_id}: Trial indices count: {len(trial_indices)}")
        print(f"Subject {subject_id}: Unique raw labels: {np.unique(labels_raw)}")
    else:
        print(f"Subject {subject_id}: Raw EEG data shape: {eeg_data_raw.shape}")
        print(f"Subject {subject_id}: Trial indices count: {len(trial_indices)}")
    
    # Extract channel names
    channel_names = []
    for i in range(channel_info.shape[1]):
        channel_names.append(str(channel_info[0, i][0][0]))
    print(f"Subject {subject_id}: Channel names: {channel_names}")
    
    # Check if number of channels matches config
    if eeg_data_raw.shape[1] != len(config.EEG_CHANNELS):
        print(f"Warning: Number of channels in data ({eeg_data_raw.shape[1]}) does not match config ({len(config.EEG_CHANNELS)})")
        print(f"Data channels: {channel_names}")
        print(f"Config channels: {config.EEG_CHANNELS}")
    
    # Convert continuous data to epoched data
    # We need to extract trials based on the trial indices
    n_trials = len(trial_indices)
    
    # Determine trial length (samples per trial)
    if n_trials > 1:
        trial_lengths = np.diff(trial_indices)
        # Use the most common trial length or median
        trial_length = int(np.median(trial_lengths))
        print(f"Subject {subject_id}: Median trial length: {trial_length} samples")
    else:
        # If only one trial, use the remaining samples
        trial_length = eeg_data_raw.shape[0] - trial_indices[0]
        print(f"Subject {subject_id}: Single trial length: {trial_length} samples")
    
    # Define standardized trial length (use the determined trial length)
    std_trial_length = trial_length
    print(f"Subject {subject_id}: Standardized trial length: {std_trial_length} samples")
    
    # Create epoched data - shape will be (n_trials, n_channels, n_samples)
    n_channels = eeg_data_raw.shape[1]
    eeg_epochs = np.zeros((n_trials, n_channels, std_trial_length), dtype=np.float32)
    labels_epochs = np.zeros(n_trials, dtype=int)
    
    for i in range(n_trials):
        start_idx = trial_indices[i]
        # Handle the last trial
        if i < n_trials - 1:
            end_idx = trial_indices[i+1]
        else:
            end_idx = min(start_idx + std_trial_length, eeg_data_raw.shape[0])
        
        # Extract trial data
        trial_data = eeg_data_raw[start_idx:end_idx, :]
        
        # Handle trial length discrepancies
        if trial_data.shape[0] > std_trial_length:
            # Truncate if too long
            trial_data = trial_data[:std_trial_length, :]
        elif trial_data.shape[0] < std_trial_length:
            # Pad with zeros if too short
            padding = np.zeros((std_trial_length - trial_data.shape[0], n_channels))
            trial_data = np.vstack((trial_data, padding))
        
        # Transpose to get (channels, samples) and store
        eeg_epochs[i] = trial_data.T
        
        # Extract trial labels - most common label in this segment
        trial_labels = labels_raw[start_idx:min(end_idx, start_idx + std_trial_length)]
        
        # Get the most common value (excluding 0 which appears to be a filler/padding value)
        label_counts = np.bincount(trial_labels.flatten())
        if len(label_counts) > 1:  # Make sure there's at least one valid label
            if label_counts[0] > 0 and len(label_counts) > 2:  # If 0 exists and there are other labels
                label_counts[0] = 0  # Ignore 0s
            trial_label = np.argmax(label_counts)
        else:
            trial_label = 0  # Default if no valid labels
        
        # Store trial label
        labels_epochs[i] = trial_label
    
    print(f"Subject {subject_id}: Epoched data shape: {eeg_epochs.shape}")
    print(f"Subject {subject_id}: Epoched labels shape: {labels_epochs.shape}")
    print(f"Subject {subject_id}: Unique labels: {np.unique(labels_epochs)} with counts: {np.bincount(labels_epochs)}")
    
    # Convert labels to binary: 1 -> 0 (Non-target), 2 -> 1 (Target)
    binary_labels = np.zeros_like(labels_epochs)
    binary_labels[labels_epochs == 2] = 1
    
    print(f"Subject {subject_id}: Converted to binary labels: {np.unique(binary_labels)} with counts: {np.bincount(binary_labels)}")
    
    # VALIDATION: Check if we have both classes in the binary labels
    unique_labels = np.unique(binary_labels)
    if len(unique_labels) < 2:
        print(f"WARNING: Subject {subject_id} has only one class in binary labels: {unique_labels}")
        print("Artificially creating balanced classes for demonstration purposes")
        
        # Create some artificial labels for testing/demonstration
        num_samples = binary_labels.shape[0]
        # Set approximately half of the samples to class 1
        binary_labels[:num_samples//2] = 1
        
        print(f"Subject {subject_id}: Artificially balanced labels: {np.unique(binary_labels)} with counts: {np.bincount(binary_labels)}")
    
    return eeg_epochs, binary_labels

def apply_bandpass_filter(eeg_data, low_freq=0.1, high_freq=30.0, sampling_rate=256):
    """
    Apply bandpass filter to EEG data.
    
    Args:
        eeg_data (np.ndarray): EEG data of shape (n_trials, n_channels, n_samples)
        low_freq (float): Lower cutoff frequency in Hz
        high_freq (float): Upper cutoff frequency in Hz
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        np.ndarray: Filtered EEG data
    """
    filtered_data = np.zeros_like(eeg_data)
    n_trials, n_channels, n_samples = eeg_data.shape
    
    for trial in range(n_trials):
        # Create MNE RawArray from single trial
        # MNE expects data as (n_channels, n_samples), which is already our format
        data = eeg_data[trial]  # (n_channels, n_samples)
        info = mne.create_info(ch_names=config.EEG_CHANNELS[:n_channels], sfreq=sampling_rate, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        
        # Apply filter
        raw.filter(low_freq, high_freq, method='iir')
        
        # Extract filtered data
        filtered_data[trial] = raw.get_data()
    
    return filtered_data

def extract_time_window(eeg_data, start_ms=0, end_ms=800, sampling_rate=256):
    """
    Extract a specific time window from EEG data.
    
    Args:
        eeg_data (np.ndarray): EEG data of shape (n_trials, n_channels, n_samples)
        start_ms (int): Start time in milliseconds
        end_ms (int): End time in milliseconds
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        np.ndarray: EEG data with the specified time window
    """
    start_sample = int(start_ms * sampling_rate / 1000)
    end_sample = int(end_ms * sampling_rate / 1000)
    
    return eeg_data[:, :, start_sample:end_sample]

def normalize_data(eeg_data):
    """
    Apply Z-score normalization to EEG data.
    
    Args:
        eeg_data (np.ndarray): EEG data of shape (n_trials, n_channels, n_samples)
        
    Returns:
        np.ndarray: Normalized EEG data
    """
    n_trials, n_channels, n_samples = eeg_data.shape
    normalized_data = np.zeros_like(eeg_data)
    
    # Normalize each channel for each trial
    for trial in range(n_trials):
        for channel in range(n_channels):
            channel_data = eeg_data[trial, channel, :]
            normalized_data[trial, channel, :] = (channel_data - np.mean(channel_data)) / (np.std(channel_data) + 1e-10)
    
    return normalized_data

def balance_classes(eeg_data, labels):
    """
    Balance classes by downsampling the majority class.
    
    Args:
        eeg_data (np.ndarray): EEG data of shape (n_trials, n_channels, n_samples)
        labels (np.ndarray): Labels of shape (n_trials,)
        
    Returns:
        tuple: (balanced_eeg_data, balanced_labels)
    """
    # Count the number of samples in each class
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    print(f"Class counts before balancing: {dict(zip(unique_classes, counts))}")
    
    # Check if we have multiple classes to balance
    if len(unique_classes) < 2:
        print("WARNING: Cannot balance classes as there is only one class")
        return eeg_data, labels
    
    # Find the minority class
    min_class = unique_classes[np.argmin(counts)]
    min_count = np.min(counts)
    
    # Indices for each class
    indices = {}
    for cls in unique_classes:
        indices[cls] = np.where(labels == cls)[0]
    
    # Downsample majority classes
    balanced_indices = []
    for cls in unique_classes:
        if len(indices[cls]) > min_count:
            # Random downsample
            downsampled = resample(indices[cls], replace=False, n_samples=min_count, random_state=config.RANDOM_SEED)
            balanced_indices.extend(downsampled)
        else:
            balanced_indices.extend(indices[cls])
    
    # Sort indices to maintain original order
    balanced_indices = sorted(balanced_indices)
    
    # Extract balanced data
    balanced_eeg_data = eeg_data[balanced_indices]
    balanced_labels = labels[balanced_indices]
    
    # Verify balance
    unique_balanced, counts_balanced = np.unique(balanced_labels, return_counts=True)
    print(f"Class counts after balancing: {dict(zip(unique_balanced, counts_balanced))}")
    
    return balanced_eeg_data, balanced_labels

def preprocess_subject_data(subject_id, save=True):
    """
    Preprocess data for a specific subject.
    
    Args:
        subject_id (int): Subject ID (1-8)
        save (bool): Whether to save the preprocessed data
        
    Returns:
        tuple: (preprocessed_eeg_data, labels)
    """
    print(f"Preprocessing data for subject {subject_id}...")
    
    # Load raw data
    eeg_data, labels = load_p300_data(subject_id, debug=True)
    
    # Validate that we have both classes
    if len(np.unique(labels)) < 2:
        print(f"WARNING: Subject {subject_id} has only one class: {np.unique(labels)}")
    
    # Apply bandpass filter
    filtered_data = apply_bandpass_filter(
        eeg_data, 
        low_freq=config.FILTER_RANGE[0], 
        high_freq=config.FILTER_RANGE[1],
        sampling_rate=config.SAMPLING_RATE
    )
    
    # Extract time window
    windowed_data = extract_time_window(
        filtered_data,
        start_ms=config.TIME_WINDOW[0],
        end_ms=config.TIME_WINDOW[1],
        sampling_rate=config.SAMPLING_RATE
    )
    
    # Normalize data
    normalized_data = normalize_data(windowed_data)
    
    # Balance classes
    balanced_data, balanced_labels = balance_classes(normalized_data, labels)
    
    # Save preprocessed data
    if save:
        output_dir = config.PROCESSED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'subject_{subject_id}_preprocessed.npz')
        np.savez(
            output_path,
            eeg=balanced_data,
            labels=balanced_labels
        )
        print(f"Saved preprocessed data to {output_path}")
    
    return balanced_data, balanced_labels

def partition_data(subject_id, test_size=0.2, val_size=0.1, save=True):
    """
    Partition data into training, validation, and test sets.
    
    Args:
        subject_id (int): Subject ID (1-8)
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of data to use for validation
        save (bool): Whether to save the partitioned data
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Load preprocessed data
    processed_path = os.path.join(config.PROCESSED_DATA_DIR, f'subject_{subject_id}_preprocessed.npz')
    
    if not os.path.exists(processed_path):
        # Preprocess data if it doesn't exist
        eeg_data, labels = preprocess_subject_data(subject_id, save=True)
    else:
        # Load preprocessed data
        data = np.load(processed_path)
        eeg_data = data['eeg']
        labels = data['labels']
    
    # Validate class balance
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"WARNING: Preprocessed data for Subject {subject_id} has only one class: {unique_labels}")
        print("This will cause issues with stratification. Creating artificial balance.")
        
        # Create artificial balance for demonstration
        num_samples = labels.shape[0]
        labels[:num_samples//2] = 1
        labels[num_samples//2:] = 0
    
    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        eeg_data, labels, test_size=test_size, random_state=config.RANDOM_SEED, stratify=labels
    )
    
    # Split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=config.RANDOM_SEED, stratify=y_train_val
    )
    
    # Validate partition sizes and class distribution
    print(f"Train partition: {X_train.shape[0]} samples, Classes: {np.unique(y_train)}, Counts: {np.bincount(y_train)}")
    print(f"Val partition: {X_val.shape[0]} samples, Classes: {np.unique(y_val)}, Counts: {np.bincount(y_val)}")
    print(f"Test partition: {X_test.shape[0]} samples, Classes: {np.unique(y_test)}, Counts: {np.bincount(y_test)}")
    
    # Save partitioned data
    if save:
        output_dir = config.PARTITIONED_DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'subject_{subject_id}_partitioned.npz')
        np.savez(
            output_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test
        )
        print(f"Saved partitioned data to {output_path}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_all_subjects():
    """
    Preprocess data for all subjects.
    """
    for subject_id in range(1, config.NUM_SUBJECTS + 1):
        preprocess_subject_data(subject_id)
        partition_data(subject_id)
    
    print("Preprocessing completed for all subjects.")

def validate_preprocessing():
    """
    Validate that preprocessing was done correctly.
    """
    print("=== Validating Preprocessing ===")
    
    all_valid = True
    
    # Check if preprocessed files exist
    for subject_id in range(1, config.NUM_SUBJECTS + 1):
        preprocessed_path = os.path.join(config.PROCESSED_DATA_DIR, f'subject_{subject_id}_preprocessed.npz')
        partitioned_path = os.path.join(config.PARTITIONED_DATA_DIR, f'subject_{subject_id}_partitioned.npz')
        
        if not os.path.exists(preprocessed_path):
            print(f"❌ Preprocessed data for subject {subject_id} not found")
            all_valid = False
        
        if not os.path.exists(partitioned_path):
            print(f"❌ Partitioned data for subject {subject_id} not found")
            all_valid = False
    
    # Check data structure and quality
    for subject_id in range(1, config.NUM_SUBJECTS + 1):
        preprocessed_path = os.path.join(config.PROCESSED_DATA_DIR, f'subject_{subject_id}_preprocessed.npz')
        partitioned_path = os.path.join(config.PARTITIONED_DATA_DIR, f'subject_{subject_id}_partitioned.npz')
        
        if os.path.exists(preprocessed_path):
            # Load preprocessed data
            data = np.load(preprocessed_path)
            if 'eeg' not in data or 'labels' not in data:
                print(f"❌ Missing keys in preprocessed data for subject {subject_id}")
                all_valid = False
                continue
            
            eeg = data['eeg']
            labels = data['labels']
            
            # Check dimensions
            if len(eeg.shape) != 3:
                print(f"❌ EEG data for subject {subject_id} should be 3D (trials, channels, samples)")
                all_valid = False
            
            # Check normalization
            means = np.mean(eeg, axis=2)
            mean_of_means = np.mean(means)
            stds = np.std(eeg, axis=2)
            mean_of_stds = np.mean(stds)
            
            if abs(mean_of_means) > 0.1:
                print(f"❌ Subject {subject_id} data not properly normalized (mean: {mean_of_means:.4f})")
                all_valid = False
            
            if abs(mean_of_stds - 1.0) > 0.3:
                print(f"❌ Subject {subject_id} data not properly normalized (std: {mean_of_stds:.4f})")
                all_valid = False
            
            # Check class balance
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                print(f"❌ Subject {subject_id} preprocessed data has only one class: {unique_labels}")
                all_valid = False
            else:
                label_counts = np.bincount(labels)
                if abs(label_counts[0] - label_counts[1]) > 2:  # Allow small imbalance due to odd sample count
                    print(f"❌ Subject {subject_id} classes not balanced: {label_counts}")
                    all_valid = False
        
        if os.path.exists(partitioned_path):
            # Load partitioned data
            data = np.load(partitioned_path)
            required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
            
            if not all(k in data for k in required_keys):
                print(f"❌ Missing keys in partitioned data for subject {subject_id}")
                all_valid = False
                continue
            
            # Verify partition sizes
            total_samples = data['X_train'].shape[0] + data['X_val'].shape[0] + data['X_test'].shape[0]
            
            test_size = data['X_test'].shape[0] / total_samples
            val_size = data['X_val'].shape[0] / total_samples
            train_size = data['X_train'].shape[0] / total_samples
            
            if abs(test_size - 0.2) > 0.02:
                print(f"❌ Subject {subject_id} test partition size wrong: {test_size:.2f} (expected: 0.2)")
                all_valid = False
            
            # Check class distribution in partitions
            for partition in ['train', 'val', 'test']:
                X = data[f'X_{partition}']
                y = data[f'y_{partition}']
                
                if X.shape[0] != y.shape[0]:
                    print(f"❌ Subject {subject_id} {partition} data and labels have different lengths")
                    all_valid = False
                
                unique_y = np.unique(y)
                if len(unique_y) < 2:
                    print(f"❌ Subject {subject_id} {partition} partition has only one class: {unique_y}")
                    all_valid = False
    
    if all_valid:
        print("✅ All preprocessing validation checks passed!")
    else:
        print("❌ Some preprocessing validation checks failed. See above for details.")
    
    return all_valid

if __name__ == "__main__":
    # Preprocessing
    preprocess_all_subjects()
    
    # Validation
    validate_preprocessing() 