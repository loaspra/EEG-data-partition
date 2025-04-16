import sys
sys.path.append("/home/loaspra/Code/PFC1/PFC1-Tema6-MIN2NET")

import tensorflow as tf
import numpy as np
import os
import argparse
import random
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from train_p300_translator import load_matlab_file, transform_data

def load_translated_data(data_path, subject_specific=False, subject_id=None):
    """Load translated EEG signals from h5 file or individual MAT files.
    
    Args:
        data_path (str): Path to directory containing translated signals
        subject_specific (bool): Whether to load only a specific subject's data
        subject_id (int): Subject ID to load if subject_specific is True
        
    Returns:
        tuple: X_data (signals), y_data (labels)
    """
    if subject_specific and subject_id is not None:
        # Load data for specific subject
        mat_file = os.path.join(data_path, f"A{subject_id:02d}_translated.mat")
        if not os.path.exists(mat_file):
            raise FileNotFoundError(f"No data found for subject A{subject_id:02d} at {mat_file}")
        
        data = sio.loadmat(mat_file)
        X_data = data['translated_signals']
        y_data = data['labels'].ravel()
        
        print(f"Loaded subject A{subject_id:02d} data - X shape: {X_data.shape}, y shape: {y_data.shape}")
    else:
        # Load combined data from h5 file
        h5_file = os.path.join(data_path, "translated_signals.h5")
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"No translated signals found at {h5_file}")
        
        with h5py.File(h5_file, 'r') as f:
            X_data = f['translated_signals'][:]
            y_data = f['labels'][:]
            subject_indices = f['subject_indices'][:]
        
        print(f"Loaded combined data - X shape: {X_data.shape}, y shape: {y_data.shape}")
    
    return X_data, y_data

def preprocess_data(X_data, y_data, flatten=True, normalize=True):
    """Preprocess the data for MLP classification.
    
    Args:
        X_data (numpy.ndarray): Input signals (D, T, C) or (N, D, T, C)
        y_data (numpy.ndarray): Class labels
        flatten (bool): Whether to flatten the signals for MLP input
        normalize (bool): Whether to normalize the features
        
    Returns:
        tuple: X_processed, y_processed
    """
    # Make sure X is 4D (N, D, T, C)
    if X_data.ndim == 3:
        X_data = X_data[:, np.newaxis, :, :]
    
    # Flatten the signals for MLP input
    if flatten:
        # Flatten the D, T, C dimensions to a single feature vector
        N, D, T, C = X_data.shape
        X_flat = X_data.reshape(N, D * T * C)
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            X_flat = scaler.fit_transform(X_flat)
        
        print(f"Flattened data shape: {X_flat.shape}")
        return X_flat, y_data
    
    # Return without flattening (mainly for CNN/RNN models)
    return X_data, y_data

def build_mlp_classifier(input_dim, num_classes=2, hidden_layers=[128, 64], dropout_rate=0.3):
    """Build a simple MLP classifier for EEG signal classification.
    
    Args:
        input_dim (int): Input feature dimension
        num_classes (int): Number of classes (default: 2 for binary classification)
        hidden_layers (list): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: MLP model
    """
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    for layer_size in hidden_layers[1:]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    if num_classes == 2:
        # Binary classification
        model.add(Dense(1, activation='sigmoid'))
    else:
        # Multi-class classification
        model.add(Dense(num_classes, activation='softmax'))
    
    return model

def train_classifier(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50, 
                     learning_rate=0.001, log_dir="logs/classifier"):
    """Train the MLP classifier.
    
    Args:
        model (tf.keras.Model): MLP model
        X_train (numpy.ndarray): Training data
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation data
        y_val (numpy.ndarray): Validation labels
        batch_size (int): Batch size
        epochs (int): Maximum number of epochs
        learning_rate (float): Learning rate
        log_dir (str): Directory to save logs and checkpoints
        
    Returns:
        tf.keras.Model: Trained model
        dict: Training history
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Compile the model
    if model.output_shape[-1] == 1:
        # Binary classification
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # Multi-class classification
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(log_dir, 'best_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    csv_logger = CSVLogger(os.path.join(log_dir, 'training_log.csv'))
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint, csv_logger],
        verbose=1
    )
    
    return model, history.history

def evaluate_classifier(model, X_test, y_test, log_dir="logs/classifier"):
    """Evaluate the classifier on test data.
    
    Args:
        model (tf.keras.Model): Trained MLP model
        X_test (numpy.ndarray): Test data
        y_test (numpy.ndarray): Test labels
        log_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    # Predict probabilities
    y_pred_prob = model.predict(X_test)
    
    # For binary classification
    if model.output_shape[-1] == 1:
        y_pred = (y_pred_prob > 0.5).astype(int).ravel()
        y_pred_prob = y_pred_prob.ravel()
    else:
        # For multi-class
        y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary')
    }
    
    # Print metrics
    print("\nClassification Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve and AUC (for binary classification)
    if model.output_shape[-1] == 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(log_dir, 'roc_curve.png'), dpi=300)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)  # Binary classification
    plt.xticks(tick_marks, ['Non-P300', 'P300'])
    plt.yticks(tick_marks, ['Non-P300', 'P300'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'), dpi=300)
    
    # Save metrics to a text file
    with open(os.path.join(log_dir, 'evaluation_metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return metrics

def load_original_data(data_path, subject_id=1):
    """Load original EEG data from the specified subject for testing.
    
    Args:
        data_path (str): Path to the raw data files
        subject_id (int): Subject ID to load (default: 1)
        
    Returns:
        tuple: X_data (signals), y_data (labels)
    """
    from train_p300_translator import load_matlab_file, transform_data
    
    # Load the subject data
    subject_key = f"A{subject_id:02d}"
    file_path = os.path.join(data_path, f"{subject_key}.mat")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data found for subject {subject_key} at {file_path}")
    
    print(f"Loading original data for subject {subject_key}...")
    matlab_data = load_matlab_file(file_path)['data'][0]
    non_p300_data, p300_data = transform_data(matlab_data)
    
    # Create balanced dataset
    # Find the minimum number of samples between the two classes
    n_samples = min(non_p300_data.shape[1], p300_data.shape[1])
    
    # Select equal numbers of samples from each class
    non_p300_samples = non_p300_data[:, :n_samples, :]
    p300_samples = p300_data[:, :n_samples, :]
    
    # Reshape to match MIN2Net format with D=1 dimension
    X_non_p300 = np.zeros((n_samples, 1, non_p300_samples.shape[0], non_p300_samples.shape[2]))
    X_p300 = np.zeros((n_samples, 1, p300_samples.shape[0], p300_samples.shape[2]))
    
    for i in range(n_samples):
        X_non_p300[i, 0] = non_p300_samples[:, i, :]
        X_p300[i, 0] = p300_samples[:, i, :]
    
    # Combine data and create labels
    X_data = np.concatenate([X_non_p300, X_p300], axis=0)
    y_data = np.zeros(X_data.shape[0])
    y_data[n_samples:] = 1  # 0 for non-P300, 1 for P300
    
    # Shuffle the data
    indices = np.arange(X_data.shape[0])
    np.random.shuffle(indices)
    X_data = X_data[indices]
    y_data = y_data[indices]
    
    print(f"Loaded original data - X shape: {X_data.shape}, y shape: {y_data.shape}")
    print(f"Class distribution: Non-P300: {np.sum(y_data == 0)}, P300: {np.sum(y_data == 1)}")
    
    return X_data, y_data

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate a P300 signal classifier')
    parser.add_argument('--train_data_path', type=str, default='target_signals', help='Path to translated signals for training')
    parser.add_argument('--test_data_path', type=str, default='data/raw', help='Path to original raw data for testing')
    parser.add_argument('--log_path', type=str, default='logs/classifier', help='Path to save logs and results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_layers', nargs='+', type=int, default=[128, 64], help='Sizes of hidden layers')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--subject_specific', action='store_true', help='Train on a specific subject')
    parser.add_argument('--subject_id', type=int, default=None, help='Subject ID to train on (if subject_specific is True)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Proportion of training data to use for validation')
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
    os.makedirs(args.log_path, exist_ok=True)
    
    # ======= TRAINING DATA: Translated Signals =======
    # Load translated signals for training
    X_train_full, y_train_full = load_translated_data(
        args.train_data_path, 
        subject_specific=args.subject_specific,
        subject_id=args.subject_id
    )
    
    # Preprocess training data
    X_train_proc, y_train_proc = preprocess_data(X_train_full, y_train_full)
    
    # Split into training and validation sets only
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_proc, y_train_proc,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_proc
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # ======= TESTING DATA: Original Subject 1 Data =======
    # Load original Subject 1 data for testing
    X_test_orig, y_test_orig = load_original_data(args.test_data_path, subject_id=1)
    
    # Preprocess testing data (same preprocessing as training data)
    X_test, y_test = preprocess_data(X_test_orig, y_test_orig)
    
    print(f"Test set (Subject 1): {X_test.shape[0]} samples")
    
    # Build model
    input_dim = X_train.shape[1]
    model = build_mlp_classifier(
        input_dim=input_dim, 
        num_classes=2,  # Binary classification: P300 vs. non-P300
        hidden_layers=args.hidden_layers,
        dropout_rate=args.dropout_rate
    )
    
    model.summary()
    
    # Train model
    print("\nTraining classifier...")
    model, history = train_classifier(
        model, X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_dir=args.log_path
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_path, 'training_history.png'), dpi=300)
    
    # Evaluate model on original Subject 1 data
    print("\nEvaluating classifier on original Subject 1 data...")
    metrics = evaluate_classifier(model, X_test, y_test, log_dir=args.log_path)
    
    print("\nClassification completed!")

if __name__ == "__main__":
    main() 