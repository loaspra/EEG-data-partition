# P300 Translation Experiment

This experiment investigates whether translating EEG data from multiple subjects into a common neural space improves P300 classification performance. We use SimpleMin2Net as a mapping network to translate P300 EEG signals from subjects 2-8 into the neural space of a reference subject (Subject 1), and then evaluate whether this translation improves subsequent classification performance compared to using original untranslated data.

## Directory Structure

```
experiment/
├── config.py                # Configuration parameters
├── data_preprocessing.py    # Data preprocessing utilities
├── feature_extraction.py    # Feature extraction methods
├── simple_min2net.py        # SimpleMin2Net model for P300 translation
├── mlp_classifier.py        # MLP classifier for P300 detection
├── run_experiment.py        # Main experiment script
└── README.md                # This file
```

## Requirements

- Python 3.8.20
- TensorFlow 2.13.1
- NumPy 1.24.3
- Scikit-learn 1.3.2
- MNE-Python
- SciPy, Pandas
- Matplotlib, Seaborn

## Dataset

We use the P300 Speller dataset involving 8 subjects with amyotrophic lateral sclerosis (ALS). The data should be organized in the following structure:

```
data/
├── raw/                     # Raw EEG data
│   ├── subject_1.npz
│   ├── subject_2.npz
│   └── ...
├── processed/               # Preprocessed data
└── partitioned/             # Data partitioned into train/val/test sets
```

## Running the Experiment

The experiment is executed in three main steps:

1. **Preprocessing**: Load raw EEG data, apply bandpass filtering, extract time windows, normalize, and balance classes
2. **Translation**: Train SimpleMin2Net to translate subjects 2-8 data to Subject 1's neural space
3. **Evaluation**: Train MLP classifiers on both original and translated data, evaluate on held-out test data

### Command Line Usage

```bash
# Run all steps
python run_experiment.py --all

# Run specific steps
python run_experiment.py --preprocess
python run_experiment.py --translate
python run_experiment.py --evaluate

# Run multiple steps
python run_experiment.py --preprocess --translate
```

## Output

The experiment generates the following outputs:

- Preprocessed and partitioned data in `data/processed/` and `data/partitioned/`
- Extracted features in `data/processed/features/`
- Trained models in `experiment/models/`
- Results and visualizations in `experiment/results/`
  - Waveform comparisons between original, translated, and reference signals
  - Classification performance metrics
  - Comparison plots between classifiers trained on original vs. translated data

## Customization

You can modify experiment parameters in `config.py`, including:

- Data parameters (channels, sampling rate, time window)
- SimpleMin2Net parameters (latent dimension)
- MLP classifier parameters (hidden layers, dropout rate)
- Feature extraction methods
- Evaluation metrics

## Implementation Details

- **Data Preprocessing**: Z-score normalization, bandpass filtering (0.1-30Hz), class balancing
- **Feature Extraction**: Statistical features, temporal parameters, Hjorth parameters, zero-crossing rate, line length
- **SimpleMin2Net Translation**: Encoder-decoder architecture with domain adaptation, using UpSampling2D instead of Conv2DTranspose
- **MLP Classification**: Multi-layer perceptron with BatchNormalization and Dropout 

# SimpleMin2Net: EEG Data Translation Model

This repository contains code for training and using a deep learning model to translate EEG signals between different subjects. The implementation uses TensorFlow and Keras.

## Models

### SimpleMin2Net

The `SimpleMin2Net` model is designed to translate EEG signals from source subjects to match the signals from a target subject. It uses an encoder-decoder architecture with a domain classifier to perform domain adaptation.

Key features:
- Encoder-decoder architecture for signal translation
- Domain adaptation to reduce domain shift between subjects
- Works with P300 EEG data in shape `(trials, 1, time_samples, channels)`
- Uses UpSampling2D instead of Conv2DTranspose for better compatibility

## Requirements

The model is tested with:
- Python 3.8.20
- TensorFlow 2.13.1
- NumPy 1.24.3
- Scikit-learn 1.3.2

## Usage

### Environment Setup

To use the models, use the conda environment:

```bash
/home/loaspra/miniconda3/envs/eeg-env/bin/python your_script.py
```

### Training a Model

```python
from simple_min2net import SimpleMin2Net

# Create the model
model = SimpleMin2Net(
    input_shape=(1, 204, 8),  # (samples, time_dimension, channels)
    latent_dim=64,            # Dimension of latent space
    epochs=100,               # Number of training epochs
    batch_size=64,            # Training batch size
    log_path='./models'       # Where to save model weights
)

# Prepare your data
# X_source: EEG data from source subjects, shape (n_trials, 1, time_samples, channels)
# y_source: Labels from source subjects
# X_target: EEG data from target subject
# y_target: Labels from target subject

# Train the model
model.fit(
    source_data=(X_source, y_source),
    target_data=(X_target, y_target)
)
```

### Translating New EEG Data

```python
# Load a pre-trained model
model = SimpleMin2Net(
    input_shape=(1, 204, 8),
    latent_dim=64,
    log_path='./models'  # Directory containing model weights
)

# Translate new EEG data
new_data = ...  # Shape (n_trials, 1, time_samples, channels)
translated_data = model.translate(new_data)
```

## Data Format

The models expect EEG data in the following format:
- Shape: `(n_trials, 1, time_samples, channels)`
- For P300 data, typically time_samples = 204 (representing an 800ms window at 256Hz)
- Number of channels depends on your EEG montage

## Notes

- The model adapts to different input shapes but works best with the shapes it was designed for
- For optimal performance, preprocess your EEG data (filtering, artifact removal) before training
- Use validation data to monitor training progress and prevent overfitting

## Example

See `test_simple_min2net.py` for a complete example of how to use the model. 