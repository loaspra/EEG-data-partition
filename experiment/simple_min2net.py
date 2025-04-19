"""
Simplified Min2Net model for P300 EEG data translation.
This version avoids known issues with Conv2DTranspose in TensorFlow.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, Lambda
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import time
from sklearn.metrics import classification_report, f1_score

class SimpleMin2Net:
    """
    Simplified Min2Net model for P300 translation without Conv2DTranspose.
    """
    def __init__(self,
                input_shape=(1, 204, 8),  # Default shape for P300 data
                latent_dim=64,
                epochs=100,
                batch_size=64,
                lr=0.001,
                min_lr=0.0001,
                factor=0.5,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='models',
                model_name='SimpleMin2Net',
                random_seed=42,
                **kwargs):
        
        D, T, C = input_shape
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_lr = min_lr
        self.factor = factor
        self.patience = patience
        self.es_patience = es_patience
        self.verbose = verbose
        self.log_path = log_path
        self.model_name = model_name
        self.random_seed = random_seed
        
        # Path for saving model weights and logs
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        
        self.weights_dir = os.path.join(log_path, f'{model_name}_weights.h5')
        self.csv_dir = os.path.join(log_path, f'{model_name}_log.log')
        self.time_log = os.path.join(log_path, f'{model_name}_time_log.csv')
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # Default parameters
        self.data_format = 'channels_last'
        self.shuffle = True
        self.metrics = 'accuracy'
        self.monitor = 'val_loss'
        self.mode = 'min'
        self.save_best_only = True
        self.save_weight_only = True
        
        # Model architecture parameters
        self.pool_size = (1, 2)
        self.filter_1 = C
        self.filter_2 = 16
        
        # Update parameters if provided in kwargs
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        
        # Set Keras backend data format
        K.set_image_data_format(self.data_format)
    
    def build(self):
        """
        Build the SimpleMin2Net model with encoder and decoder.
        
        Returns:
            Model: Compiled SimpleMin2Net model
        """
        # Encoder
        encoder_input = Input(self.input_shape)
        
        # First Conv block
        x = Conv2D(self.filter_1, (1, 5), activation='elu', padding='same',
                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(encoder_input)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=self.pool_size)(x)
        
        # Second Conv block
        x = Conv2D(self.filter_2, (1, 5), activation='elu', padding='same',
                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=self.pool_size)(x)
        
        # Third Conv block
        x = Conv2D(32, (1, 5), activation='elu', padding='same',
                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=self.pool_size)(x)
        
        # Flatten and encode to latent space
        x = Flatten()(x)
        encoder_output = Dense(self.latent_dim, activation='elu',
                             kernel_constraint=max_norm(0.5))(x)
        
        encoder = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
        
        if self.verbose > 0:
            encoder.summary()
        
        # Decoder
        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')
        
        # Calculate dimensions after pooling
        pooled_dim = self.input_shape[1] // (self.pool_size[1] ** 3)
        
        # Dense and reshape to start upsampling
        x = Dense(pooled_dim * 32, activation='elu', 
                kernel_constraint=max_norm(0.5))(decoder_input)
        x = Reshape((1, pooled_dim, 32))(x)
        
        # First upsampling block
        x = UpSampling2D(size=self.pool_size)(x)
        x = Conv2D(16, (1, 5), activation='elu', padding='same', 
                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(x)
        x = BatchNormalization()(x)
        
        # Second upsampling block
        x = UpSampling2D(size=self.pool_size)(x)
        x = Conv2D(8, (1, 5), activation='elu', padding='same',
                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(x)
        x = BatchNormalization()(x)
        
        # Third upsampling block
        x = UpSampling2D(size=self.pool_size)(x)
        x = Conv2D(self.filter_1, (1, 5), activation='elu', padding='same',
                 kernel_constraint=max_norm(2., axis=(0, 1, 2)))(x)
        
        # Make sure output shape matches input shape exactly
        def resize_to_match_input(x, target_shape):
            # Resize to match the input dimensions exactly
            return tf.image.resize(x, [1, target_shape], method='nearest')
        
        decoder_output = Lambda(
            lambda x: resize_to_match_input(x, self.input_shape[1]),
            output_shape=(1, self.input_shape[1], self.filter_1)
        )(x)
        
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
        
        if self.verbose > 0:
            decoder.summary()
        
        # Combined model for translation
        latent = encoder(encoder_input)
        translated = decoder(latent)
        
        # Simple domain classifier for adaptation
        domain_classifier = Dense(2, activation='softmax', name='domain_classifier')(latent)
        
        # Combined model
        model = Model(
            inputs=encoder_input, 
            outputs=[translated, latent, domain_classifier],
            name='SimpleMin2Net'
        )
        
        if self.verbose > 0:
            model.summary()
            print(f"Encoder Input Shape: {self.input_shape}")
            print(f"Layer encoder output shape: {encoder_output.shape}")
            print(f"Layer decoder output shape: {decoder_output.shape}")
            print(f"Layer domain_classifier output shape: {domain_classifier.shape}")
        
        return model
    
    def fit(self, source_data, target_data, val_data=None):
        """
        Train the SimpleMin2Net model for P300 translation.
        
        Args:
            source_data: Tuple of (X_source, y_source) for source domain
            target_data: Tuple of (X_target, y_target) for target domain
            val_data: Optional validation data
            
        Returns:
            dict: Training history
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Check input dimensions
        if X_source.ndim != 4:
            raise ValueError(f"ValueError: X_source is incompatible: expected ndim=4, found ndim={X_source.ndim}")
        if X_target.ndim != 4:
            raise ValueError(f"ValueError: X_target is incompatible: expected ndim=4, found ndim={X_target.ndim}")
        
        # Setup callbacks
        csv_logger = CSVLogger(self.csv_dir)
        checkpointer = ModelCheckpoint(
            monitor=self.monitor, 
            filepath=self.weights_dir,
            verbose=self.verbose, 
            save_best_only=self.save_best_only,
            save_weights_only=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor=self.monitor, 
            patience=self.patience,
            factor=self.factor, 
            mode=self.mode, 
            verbose=self.verbose,
            min_lr=self.min_lr
        )
        
        es = EarlyStopping(
            monitor=self.monitor, 
            mode=self.mode, 
            verbose=self.verbose,
            patience=self.es_patience
        )
        
        # Build and compile model
        model = self.build()
        
        # Prepare source and target domain labels (0 for source, 1 for target)
        source_domain_labels = np.zeros((X_source.shape[0], 2))
        source_domain_labels[:, 0] = 1  # One-hot encoding for source domain
        
        target_domain_labels = np.zeros((X_target.shape[0], 2))
        target_domain_labels[:, 1] = 1  # One-hot encoding for target domain
        
        # Combine source and target data for training
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.vstack([y_source, y_target]) if y_source.ndim > 1 else np.concatenate([y_source, y_target])
        domain_labels = np.vstack([source_domain_labels, target_domain_labels])
        
        # Compile model with loss weights
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss=['mse', 'mse', 'categorical_crossentropy'],  # Reconstruction, latent, domain
            loss_weights=[0.7, 0.0, 0.3],  # Focus on reconstruction and domain adaptation
            metrics=['accuracy']
        )
        
        # Prepare validation data if provided
        validation_data = None
        if val_data is not None:
            (X_val_source, y_val_source), (X_val_target, y_val_target) = val_data
            X_val_combined = np.vstack([X_val_source, X_val_target])
            y_val_combined = np.vstack([y_val_source, y_val_target]) if y_val_source.ndim > 1 else np.concatenate([y_val_source, y_val_target])
            val_source_domain_labels = np.zeros((X_val_source.shape[0], 2))
            val_source_domain_labels[:, 0] = 1
            val_target_domain_labels = np.zeros((X_val_target.shape[0], 2))
            val_target_domain_labels[:, 1] = 1
            val_domain_labels = np.vstack([val_source_domain_labels, val_target_domain_labels])
            validation_data = (X_val_combined, [X_val_combined, y_val_combined, val_domain_labels])
        
        # Train model
        start_time = time.time()
        try:
            history = model.fit(
                x=X_combined,
                y=[X_combined, y_combined, domain_labels],
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=[checkpointer, csv_logger, reduce_lr, es],
                shuffle=self.shuffle,
                verbose=self.verbose
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            return history
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def translate(self, X_data):
        """
        Translate EEG data using the trained model.
        
        Args:
            X_data: EEG data to translate
            
        Returns:
            np.ndarray: Translated EEG data
        """
        # Check input dimensions
        if X_data.ndim != 4:
            raise ValueError(f"X_data is incompatible: expected ndim=4, found ndim={X_data.ndim}")
        
        # Load model weights
        model = self.build()
        model.load_weights(self.weights_dir)
        
        # Get encoder and decoder separately
        encoder = model.get_layer('encoder')
        decoder = model.get_layer('decoder')
        
        # Translate data
        latent_repr = encoder.predict(X_data)
        translated_data = decoder.predict(latent_repr)
        
        return translated_data 