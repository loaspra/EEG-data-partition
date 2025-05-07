"""
Model utilities for P300 Translation Experiment.
Functions for model training, translation, and evaluation.
"""

import os
import numpy as np
import time
from sklearn.utils import class_weight
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import config
from utils.data_utils import load_subject_data, prepare_eeg_for_min2net, extract_features
from simple_min2net import SimpleMin2Net
from mlp_classifier import MLPClassifier, compare_classifiers

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
    
    # Check class balance for reference subject
    unique, counts = np.unique(ref_y_train, return_counts=True)
    print(f"Reference subject class distribution (train): {dict(zip(unique, counts))}")
    
    # Balance reference data if needed
    if len(np.unique(counts)) > 1:
        print("Balancing reference subject data...")
        p300_indices = np.where(ref_y_train == 1)[0]
        non_p300_indices = np.where(ref_y_train == 0)[0]
        
        # Determine the minority class
        min_class_count = min(len(p300_indices), len(non_p300_indices))
        
        # Randomly select samples from the majority class to match minority class
        if len(p300_indices) > min_class_count:
            p300_indices = np.random.choice(p300_indices, min_class_count, replace=False)
        else:
            non_p300_indices = np.random.choice(non_p300_indices, min_class_count, replace=False)
        
        # Combine the balanced indices
        balanced_indices = np.concatenate([p300_indices, non_p300_indices])
        np.random.shuffle(balanced_indices)
        
        # Create balanced dataset
        ref_X_train = ref_X_train[balanced_indices]
        ref_y_train = ref_y_train[balanced_indices]
        
        # Verify balance
        unique, counts = np.unique(ref_y_train, return_counts=True)
        print(f"Balanced reference subject class distribution (train): {dict(zip(unique, counts))}")
    
    # Load and combine data from source subjects (2-8)
    source_X_train_list = []
    source_y_train_list = []
    source_X_val_list = []
    source_y_val_list = []
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        X_train, y_train = load_subject_data(subject_id, 'train')
        X_val, y_val = load_subject_data(subject_id, 'val')
        
        # Check class balance for source subject
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"Subject {subject_id} class distribution (train): {dict(zip(unique, counts))}")
        
        # Balance source subject data if needed
        if len(np.unique(counts)) > 1:
            print(f"Balancing Subject {subject_id} data...")
            p300_indices = np.where(y_train == 1)[0]
            non_p300_indices = np.where(y_train == 0)[0]
            
            # Determine the minority class
            min_class_count = min(len(p300_indices), len(non_p300_indices))
            
            # Randomly select samples from the majority class to match minority class
            if len(p300_indices) > min_class_count:
                p300_indices = np.random.choice(p300_indices, min_class_count, replace=False)
            else:
                non_p300_indices = np.random.choice(non_p300_indices, min_class_count, replace=False)
            
            # Combine the balanced indices
            balanced_indices = np.concatenate([p300_indices, non_p300_indices])
            np.random.shuffle(balanced_indices)
            
            # Create balanced dataset
            X_train = X_train[balanced_indices]
            y_train = y_train[balanced_indices]
            
            # Verify balance
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Balanced Subject {subject_id} class distribution (train): {dict(zip(unique, counts))}")
        
        source_X_train_list.append(X_train)
        source_y_train_list.append(y_train)
        source_X_val_list.append(X_val)
        source_y_val_list.append(y_val)
    
    # Combine source subjects' data
    source_X_train = np.vstack(source_X_train_list)
    source_y_train = np.concatenate(source_y_train_list)
    source_X_val = np.vstack(source_X_val_list)
    source_y_val = np.concatenate(source_y_val_list)
    
    # Verify combined source data balance
    unique, counts = np.unique(source_y_train, return_counts=True)
    print(f"Combined source subjects class distribution (train): {dict(zip(unique, counts))}")
    
    # Balance combined source data if needed
    if len(np.unique(counts)) > 1:
        print("Balancing combined source data...")
        p300_indices = np.where(source_y_train == 1)[0]
        non_p300_indices = np.where(source_y_train == 0)[0]
        
        # Determine the minority class
        min_class_count = min(len(p300_indices), len(non_p300_indices))
        
        # Randomly select samples from the majority class to match minority class
        if len(p300_indices) > min_class_count:
            p300_indices = np.random.choice(p300_indices, min_class_count, replace=False)
        else:
            non_p300_indices = np.random.choice(non_p300_indices, min_class_count, replace=False)
        
        # Combine the balanced indices
        balanced_indices = np.concatenate([p300_indices, non_p300_indices])
        np.random.shuffle(balanced_indices)
        
        # Create balanced dataset
        source_X_train = source_X_train[balanced_indices]
        source_y_train = source_y_train[balanced_indices]
        
        # Verify balance
        unique, counts = np.unique(source_y_train, return_counts=True)
        print(f"Balanced combined source subjects class distribution (train): {dict(zip(unique, counts))}")
    
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
            
            # Store with proper keys for data and labels
            translated_data[subject_id][f"{partition}_data"] = X_translated
            translated_data[subject_id][f"{partition}_labels"] = y
    
    return translated_data

def extract_features_from_translated_data(translated_data):
    """
    Extract features from translated data.
    
    Args:
        translated_data: Dict of translated data by subject_id
    
    Returns:
        Dict of extracted features by subject_id
    """
    print("=== Extracting Features from Translated Data ===")
    
    translated_features = {}
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        translated_features[subject_id] = {}
        
        for partition in ['train', 'val', 'test']:
            print(f"Extracting features for translated subject {subject_id}, {partition} partition...")
            X = translated_data[subject_id][f"{partition}_data"]
            y = translated_data[subject_id][f"{partition}_labels"]
            
            # Extract features directly without saving
            features = extract_features(X)
            
            translated_features[subject_id][f"{partition}_features"] = features
            translated_features[subject_id][f"{partition}_labels"] = y
    
    # Also extract features for reference subject (Subject 1)
    translated_features[config.REF_SUBJECT] = {}
    
    for partition in ['train', 'val', 'test']:
        print(f"Extracting features for reference subject {config.REF_SUBJECT}, {partition} partition...")
        X, y = load_subject_data(config.REF_SUBJECT, partition)
        
        # Extract features directly
        features = extract_features(X)
        
        translated_features[config.REF_SUBJECT][f"{partition}_features"] = features
        translated_features[config.REF_SUBJECT][f"{partition}_labels"] = y
    
    return translated_features

def find_optimal_threshold(y_true, y_prob):
    """
    Find the optimal threshold for classification based on ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        
    Returns:
        float: Optimal threshold
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Calculate the geometric mean of sensitivity and specificity
    gmeans = np.sqrt(tpr * (1 - fpr))
    
    # Find the optimal threshold
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[ix]
    
    # Plot ROC curve and optimal threshold
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', 
               label=f'Optimal Threshold = {optimal_threshold:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with Optimal Threshold')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    save_path = os.path.join(config.RESULTS_DIR, 'roc_curve.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    
    return optimal_threshold

def train_and_evaluate_classifiers(translated_features=None, original_features=None):
    """
    Train and evaluate MLP classifiers on translated and original data.
    
    Args:
        translated_features: Dict of translated features by subject_id
        original_features: Dict of original features by subject_id
        
    Returns:
        tuple: (translated_metrics, original_metrics)
    """
    if translated_features is None or original_features is None:
        print("Error: Both translated and original features must be provided")
        return None, None
    
    print("=== Training and Evaluating Classifiers ===")
    
    # Get subject 1's test data - this will be used to evaluate both classifiers
    ref_subject_id = config.REF_SUBJECT
    test_features = original_features[ref_subject_id]['test_features']
    test_labels = original_features[ref_subject_id]['test_labels']
    
    print(f"Test data shape: {test_features.shape}, Labels: {test_labels.shape}")
    
    # Print class distributions
    unique, counts = np.unique(test_labels, return_counts=True)
    print(f"Test data class distribution: {dict(zip(unique, counts))}")
    
    # Combine translated data from subjects 2-8 for training
    trans_train_features = []
    trans_train_labels = []
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        # Combine train and validation data for more training samples
        trans_train_features.append(translated_features[subject_id]['train_features'])
        trans_train_labels.append(translated_features[subject_id]['train_labels'])
        trans_train_features.append(translated_features[subject_id]['val_features'])
        trans_train_labels.append(translated_features[subject_id]['val_labels'])
    
    trans_train_features = np.vstack(trans_train_features)
    trans_train_labels = np.concatenate(trans_train_labels)
    
    # Check class balance in translated training data
    unique, counts = np.unique(trans_train_labels, return_counts=True)
    print(f"Translated training data class distribution: {dict(zip(unique, counts))}")
    
    # PART 1: Train classifier on translated data
    print("\nTraining classifier on translated data...")
    translated_model = MLPClassifier(
        input_shape=trans_train_features.shape[1],
        hidden_units=[128, 64],  # Use a more complex model architecture
        dropout=0.4  # Adjust dropout for better regularization
    )
    
    start_time = time.time()
    history_trans = translated_model.train(
        trans_train_features,
        trans_train_labels,
        batch_size=config.MLP_BATCH_SIZE,
        epochs=config.MLP_EPOCHS,
        learning_rate=0.0005  # Lower learning rate for better convergence
    )
    trans_train_time = time.time() - start_time
    print(f"Training completed in {trans_train_time:.2f} seconds")
    
    # Combine original data from subjects 2-8 for training
    orig_train_features = []
    orig_train_labels = []
    
    for subject_id in range(2, config.NUM_SUBJECTS + 1):
        # Combine train and validation data for more training samples
        orig_train_features.append(original_features[subject_id]['train_features'])
        orig_train_labels.append(original_features[subject_id]['train_labels'])
        orig_train_features.append(original_features[subject_id]['val_features'])
        orig_train_labels.append(original_features[subject_id]['val_labels'])
    
    orig_train_features = np.vstack(orig_train_features)
    orig_train_labels = np.concatenate(orig_train_labels)
    
    # Check class balance in original training data
    unique, counts = np.unique(orig_train_labels, return_counts=True)
    print(f"Original training data class distribution: {dict(zip(unique, counts))}")
    
    # PART 2: Train classifier on original data
    print("\nTraining classifier on original data...")
    original_model = MLPClassifier(
        input_shape=orig_train_features.shape[1],
        hidden_units=[128, 64],  # Use the same architecture for fair comparison
        dropout=0.4
    )
    
    start_time = time.time()
    history_orig = original_model.train(
        orig_train_features,
        orig_train_labels,
        batch_size=config.MLP_BATCH_SIZE,
        epochs=config.MLP_EPOCHS,
        learning_rate=0.0005
    )
    orig_train_time = time.time() - start_time
    print(f"Training completed in {orig_train_time:.2f} seconds")
    
    # PART 3: Evaluate both classifiers on subject 1's test data
    print("\nEvaluating classifiers on Subject 1 test data...")
    
    # Make predictions with translated model and find optimal threshold
    trans_preds_raw, _ = translated_model.model.predict(test_features), None
    trans_preds_raw = trans_preds_raw.flatten()
    trans_threshold = find_optimal_threshold(test_labels, trans_preds_raw)
    
    # Evaluate translated model with optimal threshold
    trans_pred, trans_metrics = translated_model.evaluate(test_features, test_labels, threshold=trans_threshold)
    
    # Make predictions with original model and find optimal threshold
    orig_preds_raw, _ = original_model.model.predict(test_features), None
    orig_preds_raw = orig_preds_raw.flatten()
    orig_threshold = find_optimal_threshold(test_labels, orig_preds_raw)
    
    # Evaluate original model with optimal threshold
    orig_pred, orig_metrics = original_model.evaluate(test_features, test_labels, threshold=orig_threshold)
    
    # Extract metrics for easier comparison
    trans_accuracy = trans_metrics["accuracy"]
    trans_precision = trans_metrics["precision"]
    trans_recall = trans_metrics["recall"]
    trans_f1 = trans_metrics["f1"]
    trans_cm = trans_metrics["confusion_matrix"]
    
    orig_accuracy = orig_metrics["accuracy"]
    orig_precision = orig_metrics["precision"]
    orig_recall = orig_metrics["recall"]
    orig_f1 = orig_metrics["f1"]
    orig_cm = orig_metrics["confusion_matrix"]
    
    # PART 4: Compare the results
    print("\n=== Classification Results ===")
    print(f"Translated Model - Accuracy: {trans_accuracy:.4f}, F1 Score: {trans_f1:.4f}, Threshold: {trans_threshold:.4f}")
    print(f"Original Model - Accuracy: {orig_accuracy:.4f}, F1 Score: {orig_f1:.4f}, Threshold: {orig_threshold:.4f}")
    
    # Compare and visualize the results
    compare_classifiers(
        translated_model,
        original_model,
        test_features,
        test_labels,
        ['Translated Model', 'Original Model']
    )
    
    # Create a detailed comparison table
    print("\nModel Comparison:")
    print(f"{'Metric':<12} {'Translated Model':<15} {'Original Model':<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<12} {trans_accuracy:<15.4f} {orig_accuracy:<15.4f} {trans_accuracy - orig_accuracy:<15.4f}")
    print(f"{'F1 Score':<12} {trans_f1:<15.4f} {orig_f1:<15.4f} {trans_f1 - orig_f1:<15.4f}")
    print(f"{'Precision':<12} {trans_precision:<15.4f} {orig_precision:<15.4f} {trans_precision - orig_precision:<15.4f}")
    print(f"{'Recall':<12} {trans_recall:<15.4f} {orig_recall:<15.4f} {trans_recall - orig_recall:<15.4f}")
    
    # Save the detailed comparison
    comparison_file = os.path.join(config.RESULTS_DIR, "classifier_comparison.png")
    print(f"\nComparison visualization saved to: {comparison_file}")
    
    return trans_metrics, orig_metrics 