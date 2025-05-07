"""
Data utilities for P300 Translation Experiment.
Functions for loading, preprocessing, and transforming EEG data.
"""

import os
import numpy as np
import config
from data_preprocessing import preprocess_subject_data, partition_data
from feature_extraction import extract_features_all_subjects, extract_and_save_features, extract_features_from_dataset
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

def prepare_eeg_for_min2net(eeg_data):
    """
    Prepare EEG data for SimpleMin2Net model.
    Converts data from shape (trials, channels, samples) to (trials, 1, samples, channels)
    
    Args:
        eeg_data (numpy.ndarray): EEG data with shape (trials, channels, samples)
        
    Returns:
        numpy.ndarray: Reshaped data with shape (trials, 1, samples, channels)
    """
    # Convert from (trials, channels, samples) to (trials, 1, samples, channels)
    n_trials, n_channels, n_samples = eeg_data.shape
    reshaped_data = np.zeros((n_trials, 1, n_samples, n_channels))
    
    for i in range(n_trials):
        # Transpose to get shape (samples, channels) and add dimension for samples
        reshaped_data[i, 0] = eeg_data[i].T
    
    return reshaped_data

def load_subject_data(subject_id, partition='train', process_if_not_exists=True):
    """
    Load partitioned data for a given subject.
    
    Args:
        subject_id (int): Subject ID (1-8)
        partition (str): Data partition ('train', 'val', or 'test')
        process_if_not_exists (bool): Whether to process data if it doesn't exist
        
    Returns:
        tuple: (X, y) data and labels
    """
    data_path = os.path.join(config.PARTITIONED_DATA_DIR, f'subject_{subject_id}_partitioned.npz')
    
    if not os.path.exists(data_path) and process_if_not_exists:
        print(f"Partitioned data for subject {subject_id} not found. Processing raw data...")
        _ = partition_data(subject_id)
    
    data = np.load(data_path)
    
    if partition == 'train':
        return data['X_train'], data['y_train']
    elif partition == 'val':
        return data['X_val'], data['y_val']
    elif partition == 'test':
        return data['X_test'], data['y_test']
    else:
        raise ValueError(f"Invalid partition: {partition}")

def load_features(subject_id, partition='train', extract_if_not_exists=True):
    """
    Load extracted features for a given subject.
    
    Args:
        subject_id (int): Subject ID (1-8)
        partition (str): Data partition ('train', 'val', or 'test')
        extract_if_not_exists (bool): Whether to extract features if they don't exist
        
    Returns:
        tuple: (features, labels)
    """
    features_dir = os.path.join(config.PROCESSED_DATA_DIR, 'features')
    features_path = os.path.join(features_dir, f'subject_{subject_id}_{partition}_features.npz')
    
    if not os.path.exists(features_path) and extract_if_not_exists:
        print(f"Features for subject {subject_id}, {partition} partition not found. Extracting features...")
        features, labels = extract_and_save_features(subject_id, partition)
        return features, labels
    
    data = np.load(features_path)
    return data['features'], data['labels']

def extract_p300_time_window_features(eeg_data, sampling_rate=256):
    """
    Extract features specifically designed for P300 detection, focusing on time windows
    where P300 components typically appear.
    
    Args:
        eeg_data (numpy.ndarray): EEG data with shape (trials, channels, samples)
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        numpy.ndarray: P300-specific features
    """
    n_trials, n_channels, n_samples = eeg_data.shape
    features_list = []
    
    # Define time windows relevant for P300 (pre, during, post)
    # P300 typically appears between 250-500ms post-stimulus
    p300_windows = [(0, 200), (200, 400), (400, 600)]  # ms
    window_samples = [(int(w[0]*sampling_rate/1000), int(w[1]*sampling_rate/1000)) for w in p300_windows]
    
    for trial in range(n_trials):
        trial_features = []
        
        # Extract features from each channel
        for ch in range(n_channels):
            # Basic features from the entire signal
            signal = eeg_data[trial, ch, :]
            
            # Global features
            trial_features.extend([
                np.mean(signal),
                np.std(signal),
                np.max(signal),
                np.min(signal),
                skew(signal),
                kurtosis(signal)
            ])
            
            # Window-based features
            for start, end in window_samples:
                window_data = eeg_data[trial, ch, start:end]
                trial_features.extend([
                    np.mean(window_data),  # Mean amplitude
                    np.std(window_data),   # Variability
                    np.max(window_data),   # Peak amplitude
                    np.argmax(window_data) + start,  # Peak latency
                    np.sum(np.abs(np.diff(window_data)))  # Signal complexity
                ])
            
            # Peak detection
            peaks, _ = find_peaks(signal, height=0)
            if len(peaks) > 0:
                # Features based on the highest peak
                max_peak_idx = peaks[np.argmax(signal[peaks])]
                trial_features.extend([
                    max_peak_idx,  # Position of maximum peak
                    signal[max_peak_idx],  # Value of maximum peak
                    len(peaks)     # Number of peaks
                ])
            else:
                # Default values if no peaks found
                trial_features.extend([0, 0, 0])
        
        features_list.append(trial_features)
    
    return np.array(features_list)

def extract_features(eeg_data):
    """
    Extract features from EEG data using enhanced P300 features.
    
    Args:
        eeg_data (numpy.ndarray): EEG data with shape (trials, channels, samples)
        
    Returns:
        numpy.ndarray: Extracted features
    """
    # Extract standard features from feature_extraction.py
    standard_features = extract_features_from_dataset(eeg_data, config.SAMPLING_RATE)
    
    # Extract enhanced P300-specific features
    p300_features = extract_p300_time_window_features(eeg_data, config.SAMPLING_RATE)
    
    # Combine both feature sets
    combined_features = np.hstack((standard_features, p300_features))
    
    return combined_features

def load_original_features():
    """
    Load original features for all subjects.
    
    Returns:
        Dict of original features by subject_id
    """
    print("=== Loading Original Features ===")
    
    original_features = {}
    
    # Load reference subject data (Subject 1)
    ref_subject_id = config.REF_SUBJECT
    original_features[ref_subject_id] = {}
    
    for partition in ['train', 'val', 'test']:
        print(f"Loading original features for reference subject {ref_subject_id}, {partition} partition...")
        X, y = load_subject_data(ref_subject_id, partition)
        
        # Extract features
        features = extract_features(X)
        
        original_features[ref_subject_id][f'{partition}_features'] = features
        original_features[ref_subject_id][f'{partition}_labels'] = y
    
    # Load source subjects data
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        original_features[subject_id] = {}
        
        for partition in ['train', 'val', 'test']:
            print(f"Loading original features for subject {subject_id}, {partition} partition...")
            X, y = load_subject_data(subject_id, partition)
            
            # Extract features
            features = extract_features(X)
            
            original_features[subject_id][f'{partition}_features'] = features
            original_features[subject_id][f'{partition}_labels'] = y
    
    return original_features

def preprocess_and_partition_data():
    """
    Preprocess and partition data for all subjects.
    This function ensures all raw data is preprocessed and partitioned.
    """
    print("=== Preprocessing and Partitioning Data ===")
    
    for subject_id in range(1, config.NUM_SUBJECTS + 1):
        print(f"Processing subject {subject_id}...")
        
        # Step 1: Preprocess raw data
        preprocess_subject_data(subject_id, save=True)
        
        # Step 2: Partition preprocessed data
        partition_data(subject_id, save=True)
        
    print("Preprocessing and partitioning completed.") 