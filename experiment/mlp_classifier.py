"""
MLP classifier module for P300 Classification.
Implements a Multi-Layer Perceptron for P300 detection.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
import config

class MLPClassifier:
    """
    Multi-Layer Perceptron classifier for P300 detection.
    """
    def __init__(self,
                hidden_layers=[128, 64],
                dropout_rate=0.5,
                batch_size=32,
                epochs=50,
                learning_rate=0.0005,
                min_lr=0.00001,
                factor=0.5,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='models',
                model_name='P300_MLP',
                random_seed=42,
                **kwargs):
        """
        Initialize MLP classifier.
        
        Args:
            hidden_layers (list): List of neurons in hidden layers
            dropout_rate (float): Dropout rate for regularization
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            learning_rate (float): Initial learning rate
            min_lr (float): Minimum learning rate for reduction
            factor (float): Factor for learning rate reduction
            patience (int): Patience for learning rate reduction
            es_patience (int): Patience for early stopping
            verbose (int): Verbosity level
            log_path (str): Path for saving logs and model
            model_name (str): Name of the model
            random_seed (int): Random seed for reproducibility
            **kwargs: Additional parameters
        """
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.model_name = model_name
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        
        # Paths for saving model and logs
        self.model_path = os.path.join(log_path, f'{model_name}.h5')
        self.csv_path = os.path.join(log_path, f'{model_name}_log.log')
        
        # Update attributes with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def build(self, input_dim):
        """
        Build MLP model.
        
        Args:
            input_dim (int): Dimension of input features
            
        Returns:
            Sequential: Compiled MLP model
        """
        model = Sequential(name=self.model_name)
        
        # Add input layer
        model.add(Dense(self.hidden_layers[0], input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Add hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Add output layer (binary classification)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        """
        Train MLP model.
        
        Args:
            X_train (np.ndarray): Training features, shape (n_samples, n_features)
            y_train (np.ndarray): Training labels, shape (n_samples,)
            X_val (np.ndarray): Validation features, shape (n_samples, n_features)
            y_val (np.ndarray): Validation labels, shape (n_samples,)
            class_weights (dict): Class weights for imbalanced data
            
        Returns:
            Sequential: Trained MLP model
        """
        # Ensure binary labels are in correct format (0 or 1)
        y_train = y_train.astype(int)
        if y_val is not None:
            y_val = y_val.astype(int)
        
        # Build model
        input_dim = X_train.shape[1]
        model = self.build(input_dim)
        
        if self.verbose > 0:
            model.summary()
        
        # Prepare callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint = ModelCheckpoint(
            filepath=self.model_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            mode='min',
            verbose=self.verbose
        )
        callbacks.append(checkpoint)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=self.es_patience,
            mode='min',
            restore_best_weights=True,
            verbose=self.verbose
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
            mode='min',
            verbose=self.verbose
        )
        callbacks.append(reduce_lr)
        
        # CSV logger callback
        csv_logger = CSVLogger(self.csv_path)
        callbacks.append(csv_logger)
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=self.verbose
        )
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        self.model = model
        
        return history
    
    def predict(self, X_test, threshold=0.5):
        """
        Make predictions with trained model.
        
        Args:
            X_test (np.ndarray): Test features, shape (n_samples, n_features)
            threshold (float): Classification threshold for binary prediction
            
        Returns:
            tuple: (y_pred, y_prob) where y_pred is binary predictions and y_prob is probabilities
        """
        # Load model if not already loaded
        if not hasattr(self, 'model'):
            self.model = load_model(self.model_path)
        
        # Make predictions
        y_prob = self.model.predict(X_test)
        y_pred = (y_prob >= threshold).astype(int)
        
        return y_pred.flatten(), y_prob.flatten()
    
    def evaluate(self, X_test, y_test, threshold=0.5, plot_confusion=True):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features, shape (n_samples, n_features)
            y_test (np.ndarray): Test labels, shape (n_samples,)
            threshold (float): Classification threshold for binary prediction
            plot_confusion (bool): Whether to plot confusion matrix
            
        Returns:
            dict: Evaluation metrics
        """
        # Ensure labels are binary and in correct format
        y_test = y_test.astype(int)
        
        # Make predictions
        y_pred, y_prob = self.predict(X_test, threshold)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Plot confusion matrix
        if plot_confusion:
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Non-P300', 'P300'], 
                        yticklabels=['Non-P300', 'P300'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{self.model_name} Confusion Matrix')
            plt.tight_layout()
            
            # Save the plot
            cm_path = os.path.join(self.log_path, f'{self.model_name}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix saved to: {cm_path}")
        
        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        return metrics

def compare_classifiers(original_metrics, translated_metrics, save_path=None):
    """
    Compare performance of classifiers trained on original vs. translated data.
    
    Args:
        original_metrics (dict): Metrics from classifier trained on original data
        translated_metrics (dict): Metrics from classifier trained on translated data
        save_path (str): Path to save the comparison plot
        
    Returns:
        None
    """
    # Create comparison bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Bar width and positions
    bar_width = 0.35
    positions = np.arange(len(metrics))
    
    # Plot bars
    plt.bar(positions - bar_width/2, [original_metrics[m] for m in metrics], bar_width, 
            label='Original Data', color='cornflowerblue')
    plt.bar(positions + bar_width/2, [translated_metrics[m] for m in metrics], bar_width, 
            label='Translated Data', color='salmon')
    
    # Add text labels
    for i, metric in enumerate(metrics):
        plt.text(i - bar_width/2, original_metrics[metric] + 0.01, 
                 f"{original_metrics[metric]:.3f}", ha='center')
        plt.text(i + bar_width/2, translated_metrics[metric] + 0.01, 
                 f"{translated_metrics[metric]:.3f}", ha='center')
    
    # Customize plot
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Performance Comparison: Original vs. Translated Data')
    plt.xticks(positions, metrics)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    plt.close() 