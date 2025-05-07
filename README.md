# P300 Translation Experiment using SimpleMin2Net

## Project Overview

This project investigates whether translating EEG data from multiple subjects into a common neural space improves P300 classification performance. We use SimpleMin2Net as a mapping network to translate P300 EEG signals from multiple subjects (2-8) into the neural space of a reference subject (Subject 1), and then evaluate whether this translation improves subsequent classification performance compared to using original untranslated data.

## Experimental Flow

Our experiment follows these key steps:

1. **Data Preparation**: 
   - Load EEG data for all subjects (Subjects 1-8)
   - Preprocess and balance the data to ensure equal representation of P300 and non-P300 classes
   - Partition data into train, validation, and test sets

2. **Training SimpleMin2Net Translator**:
   - Train SimpleMin2Net to map signals from Subjects 2-8 to Subject 1's neural space
   - Use a pair of P300 and non-P300 signals from Subject 1 as the reference target
   - The network learns to translate signals from the source domain (Subjects 2-8) to the target domain (Subject 1)

3. **Translation**:
   - Apply the trained SimpleMin2Net to translate all signals from Subjects 2-8 to the neural space of Subject 1
   - Extract features from both translated and original data

4. **Classification Comparison**:
   - Train an MLP classifier on the translated data (from Subjects 2-8 mapped to Subject 1 space)
   - Train another MLP classifier on the original untranslated data (from Subjects 2-8)
   - Test both classifiers on held-out data from Subject 1 (data not used in training the SimpleMin2Net)
   - Compare classification performance to evaluate if translation improved generalization

5. **Evaluation**:
   - Evaluate both classifiers using accuracy, precision, recall, F1-score, and confusion matrices
   - Analyze whether translating the data to a common neural space improves classification performance

## Dataset

We use the P300 Speller dataset involving 8 subjects with amyotrophic lateral sclerosis (ALS). The P300 is an event-related potential (ERP) component that appears as a positive deflection in voltage with a latency of about 300ms after stimulus onset.

- **EEG Channels**: 8 channels (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8)
- **Sampling Rate**: 256 Hz
- **Task**: P300 Speller paradigm
- **Classes**: P300 present (target stimulus) vs. P300 absent (non-target stimulus)
- **Subjects**: 8 subjects with ALS

## Key Components

- **SimpleMin2Net**: A simplified version of Min2Net with an encoder-decoder architecture and domain classifier to translate EEG data between subject domains
- **Feature Extraction**: Extracts time and frequency domain features from EEG signals
- **MLP Classifier**: Multilayer perceptron for P300 classification

## Expected Outcomes

We aim to determine if translating EEG data to a common neural space improves the generalizability of P300 classification across subjects. Success would be indicated by the translated model showing better performance than the original model when tested on Subject 1's held-out data.

This approach could potentially help address the challenge of inter-subject variability in EEG signals, which is a significant obstacle in building cross-subject BCI systems.

## Methodology

### Data Preprocessing
1. **Balancing**: Downsample the majority class (typically non-target) to match the minority class size
2. **Normalization**: Z-score normalization across channels to standardize signal amplitudes
3. **Temporal windowing**: Extract relevant time windows post-stimulus (typically 0-800ms) where P300 is expected

### Feature Engineering (Time Domain)
We implement the following time-domain features for both the SimpleMin2Net and MLP classifiers:
1. **Statistical features**: Mean, variance, skewness, kurtosis for each channel
2. **Time-domain parameters**: Peak amplitude, peak latency, area under the curve
3. **Hjorth parameters**: Activity, mobility, and complexity
4. **Zero-crossing rate**: Frequency of signal sign changes
5. **Line length**: Sum of absolute differences between consecutive samples

### SimpleMin2Net Architecture
- **Input**: Time-domain EEG signals (8 channels × temporal samples)
- **Encoder**: Convert input EEG to latent representation
- **Decoder**: Reconstruct signals in Subject 1's neural space
- **Loss function**: Combination of reconstruction loss and domain adaptation loss
- **Training**: Using balanced pairs of P300/no-P300 samples from subjects 2-8 with Subject 1 samples as target

### MLP Classifier
- **Architecture**: Multi-layer perceptron optimized for P300 detection
- **Input**: Same feature set used with SimpleMin2Net
- **Hidden layers**: 2-3 layers with appropriate neurons and activations
- **Output**: Binary classification (P300 vs. no-P300)
- **Training**: Two separate instances:
  1. With translated data (subjects 2-8 translated to Subject 1 space)
  2. With original untranslated data (raw subjects 2-8 data)

## Important Considerations

1. **Data Balance**: Ensuring class balance is critical at every stage of the experiment:
   - During SimpleMin2Net training to prevent translation bias
   - For MLP training to avoid classifier bias toward the majority class
   - When evaluating on test data to get accurate performance metrics

2. **Subject Variability**: P300 responses vary significantly between subjects in latency, amplitude, and morphology. Translation must preserve relevant signal characteristics while standardizing to the target space.

3. **One-Shot Learning**: The SimpleMin2Net model effectively performs one-shot learning, attempting to map signals from multiple source subjects to a single target subject's neural representation.

4. **Translation vs. Adaptation**: The approach tests whether subject-specific signal patterns contain shared information that can be mapped to a common space, or if subject-specific features are essential for optimal classification.

## Evaluation

We evaluate classification performance using:
1. **PSD (Power Spectral Density)**: To detect the P300 impulse in frequency domain
2. **Classification metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
3. **Comparative analysis**: Statistical significance tests between:
   - MLP trained on translated data vs. MLP trained on original data
   - Both tested on Subject 1's held-out data

### Additional EEG-specific metrics:
- **ERD/ERS (Event-Related Desynchronization/Synchronization)**
- **Temporal correlation** between predicted and actual signals
- **Information transfer rate (ITR)** to assess BCI performance

## Implementation

The implementation is organized in the `experiment/` directory with the following key files:
- `run_experiment.py`: Main script for running the entire experiment pipeline
- `simple_min2net.py`: Implementation of the SimpleMin2Net model for translation
- `mlp_classifier.py`: MLP classifier implementation
- `data_preprocessing.py`: Data preprocessing and partitioning utilities
- `feature_extraction.py`: Feature extraction methods
- `config.py`: Configuration parameters

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy, SciPy, Pandas
- MNE-Python for EEG processing
- Scikit-learn for evaluation metrics 