#!/usr/bin/env python3
"""
Script to inspect raw data files and understand label distributions.
"""

import os
import numpy as np
import scipy.io
import config

def inspect_raw_file(subject_id):
    """
    Inspect raw .mat file for a specific subject.
    
    Args:
        subject_id (int): Subject ID (1-8)
    """
    print(f"\n=== Inspecting Raw Data for Subject {subject_id} ===")
    
    # Construct filename (e.g., A01.mat, A02.mat, ...)
    filename = f'A0{subject_id}.mat'
    data_path = os.path.join(config.RAW_DATA_DIR, filename)
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        return
    
    # Load .mat data
    try:
        mat_data = scipy.io.loadmat(data_path)
        print(f"Successfully loaded {filename}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Extract the data structure
    if 'data' not in mat_data:
        print(f"Key 'data' not found. Available keys: {list(mat_data.keys())}")
        return
    
    # Access the MATLAB struct
    struct_data = mat_data['data'][0, 0]
    print(f"Available fields in struct: {struct_data.dtype.names}")
    
    # Check for labels
    if 'y' not in struct_data.dtype.names:
        print("No 'y' field found for labels")
        return
    
    # Examine labels
    labels_raw = struct_data['y']
    unique_labels = np.unique(labels_raw)
    label_counts = {int(l): np.sum(labels_raw == l) for l in unique_labels}
    
    print(f"Raw label shape: {labels_raw.shape}")
    print(f"Unique raw labels: {unique_labels}")
    print(f"Label counts: {label_counts}")
    
    # Check trial indices
    if 'trial' in struct_data.dtype.names:
        trial_indices = struct_data['trial'].flatten()
        print(f"Number of trials: {len(trial_indices)}")
        
        # Count labels per trial
        trial_labels = []
        for i in range(len(trial_indices)):
            start_idx = trial_indices[i]
            # Handle the last trial
            if i < len(trial_indices) - 1:
                end_idx = trial_indices[i+1]
            else:
                end_idx = labels_raw.shape[0]
            
            # Get labels for this trial
            trial_label_segment = labels_raw[start_idx:end_idx]
            unique_trial_labels = np.unique(trial_label_segment)
            
            # Skip 0 which appears to be a filler value
            if len(unique_trial_labels) > 1 and 0 in unique_trial_labels:
                unique_trial_labels = unique_trial_labels[unique_trial_labels != 0]
            
            if len(unique_trial_labels) > 0:
                most_common = np.argmax(np.bincount(trial_label_segment.flatten(), minlength=max(unique_labels)+1)[1:]) + 1
                trial_labels.append(most_common)
            else:
                trial_labels.append(0)
        
        trial_labels = np.array(trial_labels)
        unique_trial_labels = np.unique(trial_labels)
        trial_label_counts = {int(l): np.sum(trial_labels == l) for l in unique_trial_labels}
        
        print(f"Processed trial label counts: {trial_label_counts}")
        
        # Convert to binary (assuming 1=non-target, 2=target)
        binary_labels = np.zeros_like(trial_labels)
        if 2 in unique_trial_labels:
            binary_labels[trial_labels == 2] = 1
            
        binary_counts = {int(l): np.sum(binary_labels == l) for l in np.unique(binary_labels)}
        print(f"Binary label counts (0=non-target, 1=target): {binary_counts}")
    
    # Examine EEG data
    if 'X' in struct_data.dtype.names:
        eeg_data = struct_data['X']
        print(f"EEG data shape: {eeg_data.shape}")

def inspect_processed_file(subject_id):
    """
    Inspect processed data for a specific subject.
    
    Args:
        subject_id (int): Subject ID (1-8)
    """
    print(f"\n=== Inspecting Processed Data for Subject {subject_id} ===")
    
    preprocessed_path = os.path.join(config.PROCESSED_DATA_DIR, f'subject_{subject_id}_preprocessed.npz')
    
    if not os.path.exists(preprocessed_path):
        print(f"Preprocessed data not found at {preprocessed_path}")
        return
    
    # Load preprocessed data
    data = np.load(preprocessed_path)
    print(f"Available keys: {list(data.keys())}")
    
    if 'eeg' in data and 'labels' in data:
        eeg = data['eeg']
        labels = data['labels']
        
        print(f"EEG data shape: {eeg.shape}")
        print(f"Labels shape: {labels.shape}")
        
        unique_labels = np.unique(labels)
        label_counts = {int(l): np.sum(labels == l) for l in unique_labels}
        
        print(f"Unique labels: {unique_labels}")
        print(f"Label counts: {label_counts}")
        
        # Check for potential normalization issues
        means = np.mean(eeg, axis=2)  # Per channel, per trial
        stds = np.std(eeg, axis=2)
        
        print(f"Mean of means: {np.mean(means):.6f}")
        print(f"Mean of stds: {np.mean(stds):.6f}")

def inspect_partitioned_file(subject_id):
    """
    Inspect partitioned data for a specific subject.
    
    Args:
        subject_id (int): Subject ID (1-8)
    """
    print(f"\n=== Inspecting Partitioned Data for Subject {subject_id} ===")
    
    partitioned_path = os.path.join(config.PARTITIONED_DATA_DIR, f'subject_{subject_id}_partitioned.npz')
    
    if not os.path.exists(partitioned_path):
        print(f"Partitioned data not found at {partitioned_path}")
        return
    
    # Load partitioned data
    data = np.load(partitioned_path)
    print(f"Available keys: {list(data.keys())}")
    
    # Check partition sizes
    if all(k in data for k in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']):
        train_size = data['X_train'].shape[0]
        val_size = data['X_val'].shape[0]
        test_size = data['X_test'].shape[0]
        total_size = train_size + val_size + test_size
        
        print(f"Train set: {train_size} samples ({train_size/total_size:.1%})")
        print(f"Val set: {val_size} samples ({val_size/total_size:.1%})")
        print(f"Test set: {test_size} samples ({test_size/total_size:.1%})")
        
        # Check label distributions
        for partition in ['train', 'val', 'test']:
            labels = data[f'y_{partition}']
            unique_labels = np.unique(labels)
            label_counts = {int(l): np.sum(labels == l) for l in unique_labels}
            
            print(f"{partition.capitalize()} set labels: {label_counts}")

def main():
    """Main function to inspect data for all subjects."""
    # Check if output directories exist
    print(f"Raw data directory: {config.RAW_DATA_DIR}")
    print(f"Processed data directory: {config.PROCESSED_DATA_DIR}")
    print(f"Partitioned data directory: {config.PARTITIONED_DATA_DIR}")
    
    for directory in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.PARTITIONED_DATA_DIR]:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} does not exist")
    
    # Inspect data for all subjects
    for subject_id in range(1, config.NUM_SUBJECTS + 1):
        inspect_raw_file(subject_id)
        inspect_processed_file(subject_id)
        inspect_partitioned_file(subject_id)

if __name__ == "__main__":
    main() 