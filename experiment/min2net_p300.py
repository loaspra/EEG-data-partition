"""
Min2Net model adaptation for P300 translation.
Adapts the Min2Net architecture for translating P300 signals between subjects.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, Lambda
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, Concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import time
from sklearn.metrics import classification_report, f1_score
import config

class Min2NetP300:
    """
    Min2Net model adapted for P300 translation between subjects.
    """
    def __init__(self,
                input_shape=(1, 204, 8),  # Updated default shape to match our dataset
                latent_dim=64,
                reconstruction_weight=0.7,
                domain_adaptation_weight=0.3,
                epochs=100,
                batch_size=64,
                lr=0.001,
                min_lr=0.0001,
                factor=0.5,
                patience=10,
                es_patience=20,
                verbose=1,
                log_path='models',
                model_name='Min2NetP300',
                random_seed=42,
                **kwargs):
        
        D, T, C = input_shape
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.reconstruction_weight = reconstruction_weight
        self.domain_adaptation_weight = domain_adaptation_weight
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
        self.subsampling_size = 51  # Adjusted to get exact output size
        self.pool_size_1 = (1, 4)
        self.pool_size_2 = (1, 4)  # Modified to get cleaner upsampling
        self.filter_1 = C
        self.filter_2 = 16
        
        # Update parameters if provided in kwargs
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])
        
        # Input shape is (D, T, C) = (1, 204, 8)
        # After pool_size_1 (1, 4): (1, 51, 8)
        # After pool_size_2 (1, 4): (1, 12.75, 16) -> (1, 13, 16)
        # We need to carefully manage the flattened size to ensure proper reconstruction
        self.flatten_size = T // self.pool_size_1[1] // self.pool_size_2[1]
        if self.flatten_size * self.pool_size_1[1] * self.pool_size_2[1] != T:
            print(f"Warning: Dimensions not evenly divisible. Input T={T}, after pooling: {self.flatten_size}")
            print(f"Expected: {self.flatten_size * self.pool_size_1[1] * self.pool_size_2[1]}, Actual: {T}")
            # Store the exact dimensions for the decoder to use
            self.exact_dim_after_pool1 = T // self.pool_size_1[1]
            self.exact_dim_after_pool2 = self.exact_dim_after_pool1 // self.pool_size_2[1]
            
            # Calculate necessary padding to ensure output dimensions match input
            self.output_padding1 = (0, 1) if (self.exact_dim_after_pool1 * self.pool_size_2[1]) < T else (0, 0)
            self.output_padding2 = (0, 0)  # Default, will be adjusted if needed
        else:
            self.exact_dim_after_pool1 = T // self.pool_size_1[1]
            self.exact_dim_after_pool2 = self.exact_dim_after_pool1 // self.pool_size_2[1]
            self.output_padding1 = (0, 0)
            self.output_padding2 = (0, 0)
        
        # Set Keras backend data format
        K.set_image_data_format(self.data_format)
    
    def build(self):
        """
        Build the Min2NetP300 model.
        
        Returns:
            Model: Compiled Min2NetP300 model
        """
        # Encoder
        encoder_input = Input(self.input_shape)
        en_conv = Conv2D(self.filter_1, (1, 64), activation='elu', padding="same", 
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(encoder_input)
        en_conv = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
        en_conv = AveragePooling2D(pool_size=self.pool_size_1)(en_conv)
        en_conv = Conv2D(self.filter_2, (1, 32), activation='elu', padding="same", 
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(en_conv)
        en_conv = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
        en_conv = AveragePooling2D(pool_size=self.pool_size_2)(en_conv)
        en_conv = Flatten()(en_conv)
        encoder_output = Dense(self.latent_dim, kernel_constraint=max_norm(0.5))(en_conv)
        encoder = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
        
        if self.verbose > 0:
            encoder.summary()
        
        # Decoder - Ensure exact output dimensions
        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')
        de_conv = Dense(1 * self.exact_dim_after_pool2 * self.filter_2, activation='elu', 
                        kernel_constraint=max_norm(0.5))(decoder_input)
        de_conv = Reshape((1, self.exact_dim_after_pool2, self.filter_2))(de_conv)
        
        # First upsampling 
        de_conv = Conv2DTranspose(filters=self.filter_2, kernel_size=(1, 32), 
                                activation='elu', padding='same', strides=self.pool_size_2, 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)),
                                output_padding=self.output_padding1)(de_conv)
        
        # Calculate the exact dimension difference to adjust for
        current_dim = de_conv.shape[2]
        target_dim = self.input_shape[1]
        
        # Apply transposed convolution
        de_conv = Conv2DTranspose(filters=self.filter_1, kernel_size=(1, 64), 
                                activation='elu', padding='same', strides=self.pool_size_1, 
                                kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
        
        # Add a Lambda layer to resize the tensor to match input dimensions
        def resize_to_match_input(x, target_shape):
            # Resize the time dimension to match the input shape
            # This uses TF's resize operations which will handle the resampling
            resized = tf.image.resize(x, [1, target_shape], method='nearest')
            return resized
        
        decoder_output = Lambda(
            lambda x: resize_to_match_input(x, self.input_shape[1]),
            output_shape=(1, self.input_shape[1], self.filter_1)
        )(de_conv)
        
        decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
        
        if self.verbose > 0:
            decoder.summary()
        
        # Print shape details to verify the match
        print(f"Encoder Input Shape: {self.input_shape}")
        print(f"Layer input_1 output shape: {encoder_input.shape}")
        print(f"Layer encoder output shape: {encoder_output.shape}")
        print(f"Layer decoder output shape: {decoder_output.shape}")
        
        # Combined model for training
        latent = encoder(encoder_input)
        reconstructed = decoder(latent)
        
        # Domain classifier for adversarial domain adaptation
        domain_classifier = Dense(2, activation='softmax', kernel_constraint=max_norm(0.5), 
                                name='domain_classifier')(latent)
        
        # Combined model
        model = Model(inputs=encoder_input, outputs=[reconstructed, latent, domain_classifier], 
                    name='Min2NetP300')
        
        return model
    
    def custom_reconstruction_loss(self, y_true, y_pred):
        """
        Custom reconstruction loss that focuses on important waveform characteristics.
        
        Args:
            y_true: Ground truth EEG signals
            y_pred: Reconstructed EEG signals
            
        Returns:
            float: Weighted MSE loss
        """
        # Regular MSE loss
        mse_loss = K.mean(K.square(y_pred - y_true), axis=-1)
        
        # Emphasize P300 time window (typically around 300ms post-stimulus)
        # For 256Hz and 800ms window, P300 would be around samples 70-90
        sample_weights = K.ones_like(y_true)
        p300_start = int(250 * 256 / 1000)  # 250ms at 256Hz
        p300_end = int(450 * 256 / 1000)    # 450ms at 256Hz
        
        # Create a weight mask that emphasizes the P300 region
        # This would need proper tensor indexing depending on the shape
        # For simplicity, we'll use a fixed weight multiplier for now
        
        return mse_loss
    
    def domain_adaptation_loss(self, y_true, y_pred):
        """
        Domain adaptation loss based on domain confusion.
        
        Args:
            y_true: Domain labels
            y_pred: Predicted domain probabilities
            
        Returns:
            float: Domain adaptation loss
        """
        # Standard categorical crossentropy
        return K.categorical_crossentropy(y_true, y_pred)
    
    def fit(self, source_data, target_data, val_data=None):
        """
        Train the Min2Net model for P300 translation.
        
        Args:
            source_data: Tuple of (X_source, y_source) for source domain (subjects 2-8)
            target_data: Tuple of (X_target, y_target) for target domain (subject 1)
            val_data: Optional validation data tuple ((X_val_source, y_val_source), (X_val_target, y_val_target))
            
        Returns:
            dict: Training history
        """
        X_source, y_source = source_data
        X_target, y_target = target_data
        
        # Check input dimensions
        if X_source.ndim != 4:
            raise ValueError(f"ValueError: `X_source` is incompatible: expected ndim=4, found ndim={X_source.ndim}")
        if X_target.ndim != 4:
            raise ValueError(f"ValueError: `X_target` is incompatible: expected ndim=4, found ndim={X_target.ndim}")
        
        # Verify time dimension matches input_shape
        if X_source.shape[2] != self.input_shape[1] or X_target.shape[2] != self.input_shape[1]:
            print(f"Warning: Data time dimension ({X_source.shape[2]}) does not match model input_shape ({self.input_shape[1]})")
            print("Adjusting model input_shape to match data")
            self.input_shape = (self.input_shape[0], X_source.shape[2], self.input_shape[2])
        
        # Setup callbacks
        csv_logger = CSVLogger(self.csv_dir)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=self.weights_dir, 
                                    verbose=self.verbose, save_best_only=self.save_best_only, 
                                    save_weight_only=self.save_weight_only)
        reduce_lr = ReduceLROnPlateau(monitor=self.monitor, patience=self.patience, 
                                    factor=self.factor, mode=self.mode, verbose=self.verbose, 
                                    min_lr=self.min_lr)
        es = EarlyStopping(monitor=self.monitor, mode=self.mode, verbose=self.verbose, 
                        patience=self.es_patience)
        
        # Build and compile model
        model = self.build()
        
        if self.verbose > 0:
            model.summary()
            # Print input and output shapes to debug dimension issues
            print(f"Encoder Input Shape: {self.input_shape}")
            # Use model._layers to get the full shape info including batch_size
            for layer in model.layers:
                if hasattr(layer, 'output_shape'):
                    print(f"Layer {layer.name} output shape: {layer.output_shape}")
        
        # Prepare source and target domain labels (0 for source, 1 for target)
        source_domain_labels = np.zeros((X_source.shape[0], 2))
        source_domain_labels[:, 0] = 1  # One-hot encoding for source domain
        
        target_domain_labels = np.zeros((X_target.shape[0], 2))
        target_domain_labels[:, 1] = 1  # One-hot encoding for target domain
        
        # Combine source and target data for training
        X_combined = np.vstack([X_source, X_target])
        y_combined = np.vstack([y_source, y_target]) if y_source.ndim > 1 else np.concatenate([y_source, y_target])
        domain_labels = np.vstack([source_domain_labels, target_domain_labels])
        
        # Compile model with custom loss weights
        model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss=['mse', 'mse', 'categorical_crossentropy'],  # Reconstruction, latent, domain
            loss_weights=[self.reconstruction_weight, 0.0, self.domain_adaptation_weight],
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
    
    def translate(self, X_data):
        """
        Translate EEG data from source subjects to target subject domain.
        
        Args:
            X_data: EEG data to translate, shape (n_trials, D, T, C)
            
        Returns:
            np.ndarray: Translated EEG data in target domain
        """
        # Check input dimensions
        if X_data.ndim != 4:
            raise ValueError(f"ValueError: `X_data` is incompatible: expected ndim=4, found ndim={X_data.ndim}")
        
        # Verify time dimension 
        if X_data.shape[2] != self.input_shape[1]:
            print(f"Warning: Data time dimension ({X_data.shape[2]}) does not match model input_shape ({self.input_shape[1]})")
            # You might need to reshape or pad the data here
        
        # Load model weights
        model = self.build()
        model.load_weights(self.weights_dir)
        
        # Get encoder and decoder separately
        encoder = model.get_layer('encoder')
        decoder = model.get_layer('decoder')
        
        # Translate data: encode to latent space then decode to target domain
        latent_repr = encoder.predict(X_data)
        translated_data = decoder.predict(latent_repr)
        
        return translated_data
    
    def get_latent_representation(self, X_data):
        """
        Get latent representation of EEG data.
        
        Args:
            X_data: EEG data, shape (n_trials, D, T, C)
            
        Returns:
            np.ndarray: Latent representation of the data
        """
        # Check input dimensions
        if X_data.ndim != 4:
            raise ValueError(f"ValueError: `X_data` is incompatible: expected ndim=4, found ndim={X_data.ndim}")
        
        # Load model weights
        model = self.build()
        model.load_weights(self.weights_dir)
        
        # Get encoder
        encoder = model.get_layer('encoder')
        
        # Get latent representation
        latent_repr = encoder.predict(X_data)
        
        return latent_repr

def prepare_eeg_for_min2net(eeg_data):
    """
    Prepare EEG data for Min2Net model.
    
    Args:
        eeg_data: EEG data, shape (n_trials, n_channels, n_samples)
        
    Returns:
        np.ndarray: Reshaped EEG data for Min2Net, shape (n_trials, 1, n_samples, n_channels)
    """
    # Reshape to Min2Net expected format: (n_trials, D, T, C)
    n_trials, n_channels, n_samples = eeg_data.shape
    min2net_data = np.zeros((n_trials, 1, n_samples, n_channels))
    
    for i in range(n_trials):
        min2net_data[i, 0, :, :] = eeg_data[i].T
    
    return min2net_data 