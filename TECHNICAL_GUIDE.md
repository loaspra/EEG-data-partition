# Technical Guide: P300 EEG Classification Experiment

## Overview

This document provides a comprehensive technical guide for the P300 EEG classification experiment, focusing on feature extraction processes, vector transformations, and the Multi-Layer Perceptron (MLP) input pipeline.

## 1. Experiment Architecture

The experiment follows a multi-stage pipeline:

```
Raw EEG Data → Preprocessing → Feature Extraction → MLP Classification
   (1632D)        (204×8)         (256D)           (Binary Output)
```

### 1.1 Data Configuration
- **Subjects**: 8 subjects total
- **EEG Channels**: 8 channels ['Fz', 'Cz', 'Pz', 'Oz', 'P3', 'P4', 'PO7', 'PO8']
- **Sampling Rate**: 256 Hz
- **Time Window**: 0-800ms post-stimulus (204 samples)
- **Raw Data Shape**: (n_trials, 8_channels, 204_samples)

## 2. Feature Extraction Process

The core innovation of this experiment is the transformation from raw EEG signals to engineered features specifically designed for P300 detection. Instead of feeding raw time-series data to the classifier, the system extracts 288 meaningful features per trial.

### 2.1 Standard Time-Domain Features (96 features total)

#### 2.1.1 Statistical Features (32 features)
For each of the 8 channels, 4 statistical measures are calculated:
- **Mean**: Average amplitude across the trial
- **Variance**: Signal variability measure
- **Skewness**: Distribution asymmetry (important for P300 waveform shape)
- **Kurtosis**: Distribution tail heaviness

**Formula Implementation**:
```python
# For each channel ch:
features[ch, 0] = np.mean(eeg_signal[ch])        # Mean
features[ch, 1] = np.var(eeg_signal[ch])         # Variance  
features[ch, 2] = stats.skew(eeg_signal[ch])     # Skewness
features[ch, 3] = stats.kurtosis(eeg_signal[ch]) # Kurtosis
```

#### 2.1.2 Temporal Parameters (24 features)
Critical for P300 detection, as P300 has characteristic timing properties:
- **Peak Amplitude**: Maximum absolute value in the signal
- **Peak Latency**: Time point of maximum amplitude (converted to ms)
- **Area Under Curve**: Integration of absolute signal values

**Implementation**:
```python
peak_amplitude = np.max(np.abs(eeg_signal[ch]))
peak_index = np.argmax(np.abs(eeg_signal[ch]))
peak_latency = peak_index * (1000 / sampling_rate)  # Convert to ms
area = np.trapz(np.abs(eeg_signal[ch]))
```

#### 2.1.3 Hjorth Parameters (24 features)
Neurophysiologically relevant parameters describing signal complexity:
- **Activity**: Signal variance (power measure)
- **Mobility**: Ratio of standard deviations of 1st derivative to signal
- **Complexity**: Ratio of mobility of 1st to 2nd derivative

**Mathematical Definition**:
```python
# First and second derivatives
diff1 = np.diff(eeg_signal[ch], n=1)
diff2 = np.diff(eeg_signal[ch], n=2)

activity = np.var(eeg_signal[ch])
mobility = np.sqrt(np.var(diff1) / (activity + 1e-10))
mobility_diff = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10))
complexity = mobility_diff / (mobility + 1e-10)
```

#### 2.1.4 Zero-Crossing Rate (8 features)
Measures signal frequency content indirectly:
```python
sign_changes = np.sum(np.diff(np.signbit(eeg_signal[ch])))
zcr = sign_changes / (n_samples - 1)
```

#### 2.1.5 Line Length (8 features)
Measures signal complexity through cumulative absolute differences:
```python
line_length = np.sum(np.abs(np.diff(eeg_signal[ch])))
```

### 2.2 P300-Specific Features (160 features total)

**✅ OPTIMIZED**: Feature redundancy has been removed. Only unique P300-specific features are included.

#### 2.2.1 Temporal Window Analysis
The P300 component typically appears 250-500ms post-stimulus. The algorithm divides the 800ms trial into three critical windows:

1. **Early Window** (0-200ms): Pre-P300 baseline activity
2. **P300 Window** (200-400ms): Primary P300 component
3. **Late Window** (400-600ms): Late positive component

For each window and each channel, 5 features are extracted:
- Mean amplitude
- Standard deviation
- Maximum amplitude
- Peak latency within window
- Signal complexity (line length)

**Total**: 3 windows × 5 features × 8 channels = 120 features

#### 2.2.2 Optimized Global Signal Features (16 features)
**Optimization**: Removed duplicates with standard statistical features.

Per-channel unique global statistics across the entire trial:
- **Maximum**: Highest signal value *(unique)*
- **Minimum**: Lowest signal value *(unique)*

**Removed duplicates**: Mean, standard deviation, skewness, kurtosis (already in standard features)

**Total**: 2 features × 8 channels = 16 features

#### 2.2.3 Peak Detection Features (24 features)
Using `scipy.signal.find_peaks`, the algorithm identifies peaks in each channel:
- Position of maximum peak
- Value of maximum peak  
- Total number of peaks detected

**Total**: 3 features × 8 channels = 24 features

## 3. Vector Formation and Shape Transformations

### 3.1 Raw Data Flow

```
Input Trial: shape (8_channels, 204_samples)
                    ↓
Feature Extraction Pipeline
                    ↓
Feature Vector: shape (288_features,)
```

### 3.2 Feature Vector Assembly

The 288-dimensional feature vector is assembled by concatenating all feature groups:

```python
# Standard features (96D)
standard_features = [
    statistical_features,    # 32D
    temporal_features,       # 24D  
    hjorth_features,        # 24D
    zcr_features,           # 8D
    line_length_features    # 8D
]

# P300-specific features (160D)  
p300_features = [
    global_features,        # 16D (optimized - removed duplicates)
    window_features,        # 120D
    peak_features          # 24D
]

# Final concatenation
final_vector = np.concatenate([
    np.concatenate(standard_features),    # 96D
    np.concatenate(p300_features)         # 160D
])  # Total: 256D
```

### 3.3 Data Pipeline Shapes

| Stage | Shape | Description |
|-------|--------|-------------|
| Raw EEG | (n_trials, 8, 204) | Original time-series data |
| Single Trial | (8, 204) | Individual trial for processing |
| Feature Vector | (256,) | Extracted features per trial (optimized) |
| Batch Features | (batch_size, 256) | Mini-batch for training |
| MLP Input | (batch_size, 256) | Direct input to neural network |

## 4. MLP Architecture and Input Processing

### 4.1 Network Architecture

```python
Sequential([
    Dense(64, input_dim=256, activation='relu'),  # Input layer (optimized)
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')                # Output layer
])
```

### 4.2 Input Processing Flow

1. **Feature Normalization**: Features are typically normalized before MLP input
2. **Batch Formation**: Features are organized into batches of size 32
3. **Forward Pass**: 256D feature vector is processed through the network
4. **Output**: Single sigmoid activation for binary P300/non-P300 classification

### 4.3 Training Configuration

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Binary crossentropy
- **Batch Size**: 32 samples
- **Epochs**: 100 (with early stopping)
- **Regularization**: 
  - Dropout (0.5)
  - Batch normalization
  - Class balancing weights

## 5. Key Technical Advantages

### 5.1 Dimensionality Reduction
- **Raw data**: 8 × 204 = 1,632 dimensions per trial
- **Feature vector**: 256 dimensions per trial (optimized, no redundancy)
- **Reduction factor**: ~6.4× smaller while preserving P300-relevant information
- **Optimization**: Removed 32 duplicate features for more efficient processing

### 5.2 Physiological Relevance
- Features capture known P300 characteristics (timing, amplitude, morphology)
- Time-window analysis aligns with neurophysiological knowledge
- Multi-channel features preserve spatial information

### 5.3 Computational Efficiency
- Faster training due to lower dimensionality
- Reduced overfitting risk
- Better generalization across subjects

## 6. Implementation Details

### 6.1 Feature Extraction Pipeline

```python
def extract_all_features(eeg_trial, sampling_rate=256):
    """Extract complete 256D feature vector from single trial (optimized)"""
    features = []
    
    # Standard features (96D)
    features.append(extract_statistical_features(eeg_trial))      # 32D
    features.append(extract_temporal_parameters(eeg_trial))       # 24D  
    features.append(extract_hjorth_parameters(eeg_trial))         # 24D
    features.append(extract_zero_crossing_rate(eeg_trial))        # 8D
    features.append(extract_line_length(eeg_trial))               # 8D
    
    # P300-specific features (160D - optimized, no duplicates)
    p300_features = extract_p300_time_window_features(eeg_trial)  # 160D
    features.append(p300_features)
    
    return np.concatenate(features)  # Final 256D vector
```

### 6.2 Batch Processing

```python
def extract_features_from_dataset(eeg_data):
    """Process entire dataset: (n_trials, 8, 204) → (n_trials, 256)"""
    n_trials = eeg_data.shape[0]
    feature_matrix = np.zeros((n_trials, 256))
    
    for trial in range(n_trials):
        feature_matrix[trial] = extract_all_features(eeg_data[trial])
    
    return feature_matrix
```

## 7. Validation and Quality Assurance

The experiment includes comprehensive validation:
- **Shape verification**: Ensures 288D output consistency
- **Feature range analysis**: Monitors for outliers and scaling issues
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Statistical testing**: Significance testing for model comparisons

## 8. Optimization Results

### 8.1 Feature Redundancy Resolution
**✅ COMPLETED**: Feature redundancy has been successfully removed from the implementation.

**Removed duplicated features (32 features)**:
- Mean: removed from P300 global features (kept in standard statistical)
- Standard deviation: removed from P300 global features (variance kept in standard)
- Skewness: removed from P300 global features (kept in standard statistical)
- Kurtosis: removed from P300 global features (kept in standard statistical)

**Optimization Results**:
- **Previous**: 288 features (with redundancy)
- **Optimized**: 256 features (no redundancy)
- **Efficiency gain**: 11.1% reduction in feature dimensionality

### 8.2 Final Feature Composition
1. **Standard features**: 96 features (unchanged)
2. **Optimized P300 features**: 160 features (reduced from 192)
   - Window analysis: 120 features
   - Peak detection: 24 features  
   - Unique global features: 16 features (max, min only)
3. **Total optimized vector**: 256 features

This feature engineering approach transforms raw EEG signals into a compact, physiologically meaningful representation optimized for P300 detection, enabling efficient and effective machine learning classification. 