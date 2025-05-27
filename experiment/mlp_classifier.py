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
import config  # Import config to ensure global seed is set
from sklearn.utils import class_weight

class MLPClassifier:
    """
    Multi-Layer Perceptron classifier for P300 detection.
    """
    def __init__(self, input_shape, hidden_units=64, dropout=0.5, random_seed=None):
        """
        Initialize MLP classifier.
        
        Args:
            input_shape (int): Dimension of input features
            hidden_units (int or list): Number of neurons in hidden layer(s)
            dropout (float): Dropout rate for regularization
            random_seed (int): Random seed for reproducibility
        """
        self.input_shape = input_shape
        self.hidden_units = hidden_units if isinstance(hidden_units, list) else [hidden_units]
        self.dropout = dropout
        self.random_seed = random_seed if random_seed is not None else config.RANDOM_SEED
        
        # Ensure reproducibility (global seed should already be set by config import)
        print(f"MLP using random seed: {self.random_seed}")
        
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Build MLP model.
        
        Returns:
            Sequential: Compiled MLP model
        """
        model = Sequential()
        
        # Add input layer
        model.add(Dense(self.hidden_units[0], input_dim=self.input_shape, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
        
        # Add hidden layers
        for units in self.hidden_units[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout))
        
        # Add output layer (binary classification)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, batch_size=32, epochs=100, learning_rate=0.001, validation_split=0.2):
        """
        Train MLP model.
        
        Args:
            X_train (np.ndarray): Training features, shape (n_samples, n_features)
            y_train (np.ndarray): Training labels, shape (n_samples,)
            batch_size (int): Batch size for training
            epochs (int): Maximum number of epochs for training
            learning_rate (float): Learning rate for the optimizer
            validation_split (float): Fraction of training data to use for validation
            
        Returns:
            history: Training history
        """
        # Update optimizer with specified learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate class weights for balanced training
        class_weights = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        # Callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        ]
        
        # Train model
        start_time = time.time()
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions with trained model.
        
        Args:
            X (np.ndarray): Input features, shape (n_samples, n_features)
            threshold (float): Classification threshold for binary prediction
            
        Returns:
            np.ndarray: Binary predictions
        """
        y_prob = self.model.predict(X)
        y_pred = (y_prob >= threshold).astype(int)
        
        return y_pred.flatten()
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate model performance.
        
        Args:
            X_test (np.ndarray): Test features, shape (n_samples, n_features)
            y_test (np.ndarray): Test labels, shape (n_samples,)
            threshold (float): Classification threshold for binary prediction
            
        Returns:
            tuple: (y_pred, metrics_dict) containing predictions and evaluation metrics
        """
        # Make predictions
        y_prob = self.model.predict(X_test)
        y_pred = (y_prob >= threshold).astype(int).flatten()
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print metrics
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Create metrics dictionary
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "probabilities": y_prob.flatten()  # Include raw probabilities for threshold tuning
        }
        
        return y_pred, metrics

def compare_classifiers(model1, model2, X_test, y_test, model_names=['Model 1', 'Model 2']):
    """
    Compare two classifiers and visualize the results.
    
    Args:
        model1: First classifier model
        model2: Second classifier model
        X_test: Test features
        y_test: Test labels
        model_names: Names of the models to display
    """
    # Get predictions and confusion matrices
    y_pred1 = model1.predict(X_test)
    y_pred2 = model2.predict(X_test)
    
    cm1 = confusion_matrix(y_test, y_pred1)
    cm2 = confusion_matrix(y_test, y_pred2)
    
    # Calculate metrics
    acc1 = accuracy_score(y_test, y_pred1)
    acc2 = accuracy_score(y_test, y_pred2)
    
    f1_1 = f1_score(y_test, y_pred1, zero_division=0)
    f1_2 = f1_score(y_test, y_pred2, zero_division=0)
    
    prec1 = precision_score(y_test, y_pred1, zero_division=0)
    prec2 = precision_score(y_test, y_pred2, zero_division=0)
    
    rec1 = recall_score(y_test, y_pred1, zero_division=0)
    rec2 = recall_score(y_test, y_pred2, zero_division=0)
    
    # Print comparison table
    print("\nModel Comparison:")
    print(f"{'Metric':<12} {model_names[0]:<15} {model_names[1]:<15} {'Difference':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<12} {acc1:<15.4f} {acc2:<15.4f} {acc1-acc2:<15.4f}")
    print(f"{'F1 Score':<12} {f1_1:<15.4f} {f1_2:<15.4f} {f1_1-f1_2:<15.4f}")
    print(f"{'Precision':<12} {prec1:<15.4f} {prec2:<15.4f} {prec1-prec2:<15.4f}")
    print(f"{'Recall':<12} {rec1:<15.4f} {rec2:<15.4f} {rec1-rec2:<15.4f}")
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-P300', 'P300'], 
                yticklabels=['Non-P300', 'P300'], 
                ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'{model_names[0]} Confusion Matrix')
    
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-P300', 'P300'], 
                yticklabels=['Non-P300', 'P300'], 
                ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title(f'{model_names[1]} Confusion Matrix')
    
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(config.RESULTS_DIR, 'classifier_comparison.png')
    plt.savefig(save_path)
    print(f"\nComparison visualization saved to: {save_path}")
    plt.close()