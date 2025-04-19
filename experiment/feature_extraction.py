"""
Feature extraction module for P300 Translation Experiment.
Implements various time-domain features for EEG signals.
"""

import os
import numpy as np
import scipy.stats as stats
from scipy.signal import find_peaks
import config

def extract_statistical_features(eeg_trial):
    """
    Extract statistical features from EEG data.
    
    Args:
        eeg_trial (np.ndarray): EEG data for a single trial, shape (n_channels, n_samples)
        
    Returns:
        np.ndarray: Statistical features (mean, variance, skewness, kurtosis) for each channel
    """
    n_channels = eeg_trial.shape[0]
    features = np.zeros((n_channels, 4))
    
    for ch in range(n_channels):
        # Mean
        features[ch, 0] = np.mean(eeg_trial[ch])
        # Variance
        features[ch, 1] = np.var(eeg_trial[ch])
        # Skewness
        features[ch, 2] = stats.skew(eeg_trial[ch])
        # Kurtosis
        features[ch, 3] = stats.kurtosis(eeg_trial[ch])
    
    return features.flatten()

def extract_temporal_parameters(eeg_trial, sampling_rate=256):
    """
    Extract temporal parameters from EEG data.
    
    Args:
        eeg_trial (np.ndarray): EEG data for a single trial, shape (n_channels, n_samples)
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        np.ndarray: Temporal parameters (peak amplitude, peak latency, area under curve)
    """
    n_channels = eeg_trial.shape[0]
    features = np.zeros((n_channels, 3))
    
    for ch in range(n_channels):
        # Peak amplitude (maximum absolute value)
        peak_amplitude = np.max(np.abs(eeg_trial[ch]))
        features[ch, 0] = peak_amplitude
        
        # Peak latency (time of maximum absolute value in ms)
        peak_index = np.argmax(np.abs(eeg_trial[ch]))
        peak_latency = peak_index * (1000 / sampling_rate)  # Convert to ms
        features[ch, 1] = peak_latency
        
        # Area under the curve (simple integration)
        area = np.trapz(np.abs(eeg_trial[ch]))
        features[ch, 2] = area
    
    return features.flatten()

def extract_hjorth_parameters(eeg_trial):
    """
    Extract Hjorth parameters from EEG data.
    
    Args:
        eeg_trial (np.ndarray): EEG data for a single trial, shape (n_channels, n_samples)
        
    Returns:
        np.ndarray: Hjorth parameters (activity, mobility, complexity) for each channel
    """
    n_channels = eeg_trial.shape[0]
    features = np.zeros((n_channels, 3))
    
    for ch in range(n_channels):
        # First derivative
        diff1 = np.diff(eeg_trial[ch], n=1)
        # Second derivative
        diff2 = np.diff(eeg_trial[ch], n=2)
        
        # Activity (variance of the signal)
        activity = np.var(eeg_trial[ch])
        features[ch, 0] = activity
        
        # Mobility (square root of variance of the first derivative divided by variance of the signal)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
        features[ch, 1] = mobility
        
        # Complexity (mobility of the first derivative divided by mobility of the signal)
        mobility_diff = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
        complexity = mobility_diff / (mobility + 1e-10)
        features[ch, 2] = complexity
    
    return features.flatten()

def extract_zero_crossing_rate(eeg_trial):
    """
    Extract zero-crossing rate from EEG data.
    
    Args:
        eeg_trial (np.ndarray): EEG data for a single trial, shape (n_channels, n_samples)
        
    Returns:
        np.ndarray: Zero-crossing rate for each channel
    """
    n_channels = eeg_trial.shape[0]
    n_samples = eeg_trial.shape[1]
    features = np.zeros(n_channels)
    
    for ch in range(n_channels):
        # Count sign changes
        sign_changes = np.sum(np.diff(np.signbit(eeg_trial[ch])))
        # Normalize by signal length
        zero_crossing_rate = sign_changes / (n_samples - 1)
        features[ch] = zero_crossing_rate
    
    return features

def extract_line_length(eeg_trial):
    """
    Extract line length from EEG data.
    
    Args:
        eeg_trial (np.ndarray): EEG data for a single trial, shape (n_channels, n_samples)
        
    Returns:
        np.ndarray: Line length for each channel
    """
    n_channels = eeg_trial.shape[0]
    features = np.zeros(n_channels)
    
    for ch in range(n_channels):
        # Sum of absolute differences between consecutive samples
        line_length = np.sum(np.abs(np.diff(eeg_trial[ch])))
        features[ch] = line_length
    
    return features

def extract_all_features(eeg_trial, sampling_rate=256):
    """
    Extract all time-domain features from EEG data.
    
    Args:
        eeg_trial (np.ndarray): EEG data for a single trial, shape (n_channels, n_samples)
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        np.ndarray: Combined feature vector
    """
    features = []
    
    # Extract features based on configuration
    if "statistical" in config.FEATURE_EXTRACTION_METHODS:
        features.append(extract_statistical_features(eeg_trial))
    
    if "temporal" in config.FEATURE_EXTRACTION_METHODS:
        features.append(extract_temporal_parameters(eeg_trial, sampling_rate))
    
    if "hjorth" in config.FEATURE_EXTRACTION_METHODS:
        features.append(extract_hjorth_parameters(eeg_trial))
    
    if "zcr" in config.FEATURE_EXTRACTION_METHODS:
        features.append(extract_zero_crossing_rate(eeg_trial))
    
    if "line_length" in config.FEATURE_EXTRACTION_METHODS:
        features.append(extract_line_length(eeg_trial))
    
    return np.concatenate(features)

def extract_features_from_dataset(eeg_data, sampling_rate=256):
    """
    Extract features from a dataset of EEG trials.
    
    Args:
        eeg_data (np.ndarray): EEG data, shape (n_trials, n_channels, n_samples)
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        np.ndarray: Feature matrix, shape (n_trials, n_features)
    """
    n_trials = eeg_data.shape[0]
    features_list = []
    
    for trial in range(n_trials):
        trial_features = extract_all_features(eeg_data[trial], sampling_rate)
        features_list.append(trial_features)
    
    return np.array(features_list)

def extract_and_save_features(subject_id, partition='train'):
    """
    Extract features from partitioned data and save them.
    
    Args:
        subject_id (int): Subject ID (1-8)
        partition (str): Data partition ('train', 'val', or 'test')
        
    Returns:
        tuple: (features, labels)
    """
    # Load partitioned data
    data_path = os.path.join(config.PARTITIONED_DATA_DIR, f'subject_{subject_id}_partitioned.npz')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Partitioned data for subject {subject_id} not found at {data_path}")
    
    data = np.load(data_path)
    
    if partition == 'train':
        eeg_data = data['X_train']
        labels = data['y_train']
    elif partition == 'val':
        eeg_data = data['X_val']
        labels = data['y_val']
    elif partition == 'test':
        eeg_data = data['X_test']
        labels = data['y_test']
    else:
        raise ValueError(f"Invalid partition: {partition}")
    
    # Extract features
    print(f"Extracting features for subject {subject_id}, {partition} partition...")
    features = extract_features_from_dataset(eeg_data, config.SAMPLING_RATE)
    
    # Save features
    output_dir = os.path.join(config.PROCESSED_DATA_DIR, 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'subject_{subject_id}_{partition}_features.npz')
    np.savez(
        output_path,
        features=features,
        labels=labels
    )
    print(f"Saved features to {output_path}")
    
    return features, labels

def extract_features_all_subjects():
    """
    Extract features for all subjects and all partitions.
    """
    for subject_id in range(1, config.NUM_SUBJECTS + 1):
        for partition in ['train', 'val', 'test']:
            extract_and_save_features(subject_id, partition)
    
    print("Feature extraction completed for all subjects.")

if __name__ == "__main__":
    extract_features_all_subjects() 