"""
Main script for running the P300 Translation Experiment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from sklearn.utils import class_weight

# Import local modules
import config
from data_preprocessing import preprocess_subject_data, partition_data
from feature_extraction import extract_features_all_subjects, extract_and_save_features
from simple_min2net import SimpleMin2Net
from mlp_classifier import MLPClassifier, compare_classifiers

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

def train_min2net_translator():
    """
    Train SimpleMin2Net translator to map subjects 2-8 to Subject 1's neural space.
    
    Returns:
        SimpleMin2Net: Trained translator model
    """
    print("=== Training SimpleMin2Net P300 Translator ===")
    
    # Load data for reference subject (Subject 1)
    ref_X_train, ref_y_train = load_subject_data(config.REF_SUBJECT, 'train')
    ref_X_val, ref_y_val = load_subject_data(config.REF_SUBJECT, 'val')
    
    # Load and combine data from source subjects (2-8)
    source_X_train_list = []
    source_y_train_list = []
    source_X_val_list = []
    source_y_val_list = []
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        X_train, y_train = load_subject_data(subject_id, 'train')
        X_val, y_val = load_subject_data(subject_id, 'val')
        
        source_X_train_list.append(X_train)
        source_y_train_list.append(y_train)
        source_X_val_list.append(X_val)
        source_y_val_list.append(y_val)
    
    # Combine source subjects' data
    source_X_train = np.vstack(source_X_train_list)
    source_y_train = np.concatenate(source_y_train_list)
    source_X_val = np.vstack(source_X_val_list)
    source_y_val = np.concatenate(source_y_val_list)
    
    # Prepare data for SimpleMin2Net
    source_X_train_min2net = prepare_eeg_for_min2net(source_X_train)
    source_y_train_min2net = source_y_train
    ref_X_train_min2net = prepare_eeg_for_min2net(ref_X_train)
    ref_y_train_min2net = ref_y_train
    
    source_X_val_min2net = prepare_eeg_for_min2net(source_X_val)
    source_y_val_min2net = source_y_val
    ref_X_val_min2net = prepare_eeg_for_min2net(ref_X_val)
    ref_y_val_min2net = ref_y_val
    
    # Initialize SimpleMin2Net model
    D, T, C = 1, ref_X_train.shape[2], ref_X_train.shape[1]
    input_shape = (D, T, C)
    
    translator = SimpleMin2Net(
        input_shape=input_shape,
        latent_dim=config.LATENT_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        lr=config.LEARNING_RATE,
        log_path=config.MODEL_DIR,
        model_name='P300_Translator'
    )
    
    # Train translator
    print(f"Training on {source_X_train_min2net.shape[0]} source samples and {ref_X_train_min2net.shape[0]} reference samples")
    translator.fit(
        source_data=(source_X_train_min2net, source_y_train_min2net),
        target_data=(ref_X_train_min2net, ref_y_train_min2net),
        val_data=((source_X_val_min2net, source_y_val_min2net), (ref_X_val_min2net, ref_y_val_min2net))
    )
    
    return translator

def translate_source_subjects(translator):
    """
    Translate EEG data from source subjects (2-8) to reference subject space.
    
    Args:
        translator (SimpleMin2Net): Trained translator model
        
    Returns:
        dict: Dictionary of translated data for each subject and partition
    """
    print("=== Translating Source Subject Data ===")
    translated_data = {}
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        translated_data[subject_id] = {}
        
        for partition in ['train', 'val', 'test']:
            print(f"Translating subject {subject_id}, {partition} partition...")
            X, y = load_subject_data(subject_id, partition)
            X_min2net = prepare_eeg_for_min2net(X)
            
            # Translate data
            X_translated_min2net = translator.translate(X_min2net)
            
            # Convert back from Min2Net format to original format
            n_trials = X_translated_min2net.shape[0]
            n_channels = X_translated_min2net.shape[3]
            n_samples = X_translated_min2net.shape[2]
            
            X_translated = np.zeros((n_trials, n_channels, n_samples))
            for i in range(n_trials):
                X_translated[i] = X_translated_min2net[i, 0].T
            
            translated_data[subject_id][partition] = (X_translated, y)
    
    return translated_data

def extract_features_from_translated_data(translated_data):
    """
    Extract features from translated data.
    
    Args:
        translated_data (dict): Dictionary of translated data
        
    Returns:
        dict: Dictionary of extracted features from translated data
    """
    print("=== Extracting Features from Translated Data ===")
    translated_features = {}
    
    for subject_id in translated_data:
        translated_features[subject_id] = {}
        
        for partition in translated_data[subject_id]:
            X_translated, y = translated_data[subject_id][partition]
            print(f"Extracting features for translated subject {subject_id}, {partition} partition...")
            
            # Extract features from translated data
            n_trials = X_translated.shape[0]
            features_list = []
            
            from feature_extraction import extract_all_features
            
            for trial in range(n_trials):
                trial_features = extract_all_features(X_translated[trial], config.SAMPLING_RATE)
                features_list.append(trial_features)
            
            features = np.array(features_list)
            translated_features[subject_id][partition] = (features, y)
    
    return translated_features

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

def train_and_evaluate_classifiers(translated_features):
    """
    Train and evaluate MLP classifiers on original and translated data.
    
    Args:
        translated_features (dict): Dictionary of extracted features from translated data
        
    Returns:
        tuple: (original_metrics, translated_metrics)
    """
    print("=== Training and Evaluating Classifiers ===")
    
    # Load reference subject (Subject 1) data for testing
    ref_X_test, ref_y_test = load_features(config.REF_SUBJECT, 'test')
    
    # Combine original source subjects' data (subjects 2-8)
    original_X_train_list = []
    original_y_train_list = []
    original_X_val_list = []
    original_y_val_list = []
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        X_train, y_train = load_features(subject_id, 'train')
        X_val, y_val = load_features(subject_id, 'val')
        
        original_X_train_list.append(X_train)
        original_y_train_list.append(y_train)
        original_X_val_list.append(X_val)
        original_y_val_list.append(y_val)
    
    original_X_train = np.vstack(original_X_train_list)
    original_y_train = np.concatenate(original_y_train_list)
    original_X_val = np.vstack(original_X_val_list)
    original_y_val = np.concatenate(original_y_val_list)
    
    # Combine translated source subjects' data
    translated_X_train_list = []
    translated_y_train_list = []
    translated_X_val_list = []
    translated_y_val_list = []
    
    for subject_id in translated_features:
        X_train, y_train = translated_features[subject_id]['train']
        X_val, y_val = translated_features[subject_id]['val']
        
        translated_X_train_list.append(X_train)
        translated_y_train_list.append(y_train)
        translated_X_val_list.append(X_val)
        translated_y_val_list.append(y_val)
    
    translated_X_train = np.vstack(translated_X_train_list)
    translated_y_train = np.concatenate(translated_y_train_list)
    translated_X_val = np.vstack(translated_X_val_list)
    translated_y_val = np.concatenate(translated_y_val_list)
    
    # Calculate class weights
    classes = np.unique(original_y_train)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=original_y_train)
    class_weights_dict = {i: weight for i, weight in zip(classes, class_weights)}
    
    # Train MLP on original data
    print("\nTraining MLP on original data...")
    mlp_original = MLPClassifier(
        hidden_layers=config.MLP_HIDDEN_LAYERS,
        dropout_rate=config.MLP_DROPOUT,
        batch_size=config.MLP_BATCH_SIZE,
        epochs=config.MLP_EPOCHS,
        learning_rate=config.MLP_LEARNING_RATE,
        log_path=config.MODEL_DIR,
        model_name='P300_MLP_Original'
    )
    
    mlp_original.fit(
        original_X_train, original_y_train,
        X_val=original_X_val, y_val=original_y_val,
        class_weights=class_weights_dict
    )
    
    # Train MLP on translated data
    print("\nTraining MLP on translated data...")
    mlp_translated = MLPClassifier(
        hidden_layers=config.MLP_HIDDEN_LAYERS,
        dropout_rate=config.MLP_DROPOUT,
        batch_size=config.MLP_BATCH_SIZE,
        epochs=config.MLP_EPOCHS,
        learning_rate=config.MLP_LEARNING_RATE,
        log_path=config.MODEL_DIR,
        model_name='P300_MLP_Translated'
    )
    
    mlp_translated.fit(
        translated_X_train, translated_y_train,
        X_val=translated_X_val, y_val=translated_y_val,
        class_weights=class_weights_dict
    )
    
    # Evaluate on reference subject test data
    print("\nEvaluating classifiers on reference subject test data...")
    print("\nResults for Original Data Classifier:")
    original_metrics = mlp_original.evaluate(ref_X_test, ref_y_test)
    
    print("\nResults for Translated Data Classifier:")
    translated_metrics = mlp_translated.evaluate(ref_X_test, ref_y_test)
    
    # Compare results
    compare_path = os.path.join(config.RESULTS_DIR, 'classifier_comparison.png')
    compare_classifiers(original_metrics, translated_metrics, compare_path)
    
    return original_metrics, translated_metrics

def run_experiment(preprocess=False, translate=True, evaluate=True):
    """
    Run the P300 translation experiment.
    
    Args:
        preprocess (bool): Whether to preprocess raw data
        translate (bool): Whether to translate data from source subjects to target domain
        evaluate (bool): Whether to evaluate classifiers
        
    Returns:
        None
    """
    start_time = time.time()
    
    # Create output directories
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(config.PARTITIONED_DATA_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Preprocess data if requested
    if preprocess:
        print("=== Preprocessing Raw Data ===")
        from data_preprocessing import preprocess_all_subjects
        preprocess_all_subjects()
        
        print("=== Extracting Features ===")
        extract_features_all_subjects()
    
    translator = None
    translated_data = None
    
    # Translate data if requested
    if translate:
        # Train SimpleMin2Net translator
        translator = train_min2net_translator()
        
        # Translate source subjects' data to reference subject space
        translated_data = translate_source_subjects(translator)
        
        # Compare original vs. translated waveforms
        compare_original_vs_translated(translator)
        
        # Extract features from translated data
        translated_features = extract_features_from_translated_data(translated_data)
    
    # Evaluate classifiers if requested
    if evaluate and translated_data is not None:
        # Train and evaluate MLP classifiers
        original_metrics, translated_metrics = train_and_evaluate_classifiers(translated_features)
    
    end_time = time.time()
    print(f"\nExperiment completed in {(end_time - start_time) / 60:.2f} minutes.")

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='P300 Translation Experiment')
    parser.add_argument('--preprocess', action='store_true', help='Preprocess raw data')
    parser.add_argument('--translate', action='store_true', help='Translate data')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate classifiers')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # If no arguments are provided or --all is specified, run all steps
    if not (args.preprocess or args.translate or args.evaluate) or args.all:
        args.preprocess = True
        args.translate = True
        args.evaluate = True
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    run_experiment(
        preprocess=args.preprocess,
        translate=args.translate,
        evaluate=args.evaluate
    ) 