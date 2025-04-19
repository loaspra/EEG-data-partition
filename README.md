# P300 Translation Experiment using Min2Net

## Project Overview

This project investigates whether translating EEG data from multiple subjects into a common neural space improves P300 classification performance. We use Min2Net as a mapping network to translate P300 EEG signals from multiple subjects (2-8) into the neural space of a reference subject (Subject 1), and then evaluate whether this translation improves subsequent classification performance compared to using original untranslated data.

## Dataset

We use the P300 Speller dataset involving 8 subjects with amyotrophic lateral sclerosis (ALS). In this paradigm:
- Subjects focused on one of 36 characters in a 6×6 matrix
- Rows and columns flashed randomly at 4Hz (250ms between stimuli)
- EEG was recorded from 8 channels (Fz, Cz, Pz, Oz, P3, P4, PO7, PO8)
- Data was sampled at 256Hz and filtered between 0.1-30Hz
- Each character required focusing through multiple stimulation sequences
- The dataset contains labeled target (P300) and non-target (no P300) responses

## Methodology

### Data Preprocessing
1. **Balancing**: Downsample the majority class (typically non-target) to match the minority class size
2. **Normalization**: Z-score normalization across channels to standardize signal amplitudes
3. **Temporal windowing**: Extract relevant time windows post-stimulus (typically 0-800ms) where P300 is expected

### Feature Engineering (Time Domain)
We will implement the following time-domain features for both the Min2Net and MLP classifiers:
1. **Statistical features**: Mean, variance, skewness, kurtosis for each channel
2. **Time-domain parameters**: Peak amplitude, peak latency, area under the curve
3. **Hjorth parameters**: Activity, mobility, and complexity
4. **Zero-crossing rate**: Frequency of signal sign changes
5. **Line length**: Sum of absolute differences between consecutive samples

### Min2Net Architecture
- **Input**: Time-domain EEG signals (8 channels × temporal samples)
- **Encoder**: Convert input EEG to latent representation
- **Decoder**: Reconstruct signals in Subject 1's neural space
- **Loss function**: Combination of reconstruction loss and domain adaptation loss
- **Training**: Using balanced pairs of P300/no-P300 samples from subjects 2-8 with Subject 1 samples as target

### MLP Classifier
- **Architecture**: Multi-layer perceptron optimized for P300 detection
- **Input**: Same feature set used with Min2Net
- **Hidden layers**: 2-3 layers with appropriate neurons and activations
- **Output**: Binary classification (P300 vs. no-P300)
- **Training**: Two separate instances:
  1. With translated data (subjects 2-8 translated to Subject 1 space)
  2. With original untranslated data (raw subjects 2-8 data)

## Evaluation

We will evaluate classification performance using:
1. **PSD (Power Spectral Density)**: To detect the P300 impulse in frequency domain
2. **Classification metrics**: Accuracy, precision, recall, F1-score, AUC-ROC
3. **Comparative analysis**: Statistical significance tests between:
   - MLP trained on translated data vs. MLP trained on original data
   - Both tested on Subject 1's held-out data

### Additional EEG-specific metrics:
- **ERD/ERS (Event-Related Desynchronization/Synchronization)**
- **Temporal correlation** between predicted and actual signals
- **Information transfer rate (ITR)** to assess BCI performance

## Considerations and Limitations

### Key Considerations
1. **Subject variability**: P300 responses vary significantly between subjects in latency, amplitude, and morphology
2. **Translation target selection**: Random selection of Subject 1 samples may not represent the optimal target space
3. **Feature selection**: Time-domain features may capture different aspects than frequency-domain features
4. **Overfitting risk**: The model may learn to map to specific Subject 1 samples rather than generalizing

### Limitations
1. **Single reference subject**: Results may be highly dependent on Subject 1's data quality
2. **Small dataset**: Limited number of subjects (8) and samples may affect generalizability
3. **Session-to-session variability**: EEG signals vary between recording sessions
4. **ALS progression**: Different stages of ALS may affect signal characteristics

### Critical Analysis
The fundamental assumption that translating signals to match Subject 1's neural patterns will improve classification warrants careful examination. If successful, this could indicate that subject-specific signal patterns contain shared information that can be mapped to a common space. If unsuccessful, it may suggest that subject-specific features are essential for optimal classification.

## Implementation Steps

1. Data loading and preprocessing
2. Feature extraction
3. Min2Net model implementation and training
4. Translation of subjects 2-8 data to Subject 1 space
5. MLP classifier training (on both translated and original data)
6. Evaluation on Subject 1's held-out data
7. Comparative analysis and interpretation

## Requirements

- Python 3.x
- TensorFlow/Keras
- NumPy, SciPy, Pandas
- MNE-Python for EEG processing
- Scikit-learn for evaluation metrics 