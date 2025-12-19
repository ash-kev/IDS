# Intrusion Detection System (IDS) with Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

A Machine Learning-based Intrusion Detection System (IDS) that classifies network traffic as Normal or Attack using a Feed-Forward Neural Network trained on the KDD99/NSL-KDD dataset. This project provides an end-to-end ML pipeline for binary intrusion detection, handling both numeric and categorical features with class imbalance mitigation.

## Table of Contents

- [Features](#features)
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Architecture](#model-architecture)
- [Sample Predictions](#sample-predictions)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Binary Intrusion Detection**: Classifies network traffic as Normal (0) or Attack (1).
- **End-to-End Pipeline**: Includes data loading, preprocessing, model training, and evaluation.
- **Feature Handling**: Processes 41 network features (numeric and categorical).
- **Class Imbalance Handling**: Uses class weights to balance Normal vs. Attack samples.
- **Overfitting Prevention**: Implements early stopping and dropout layers.
- **Evaluation**: Comprehensive metrics including accuracy, precision, F1-score, and confusion matrix.

## Model Overview

- **Model Type**: Feed-Forward Artificial Neural Network (ANN)
- **Framework**: TensorFlow/Keras
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam (learning rate: 0.001)
- **Output**: Attack probability (Sigmoid activation)
- **Architecture**: 4-layer network with dropout for regularization

## Dataset

- **Dataset**: KDD99 / NSL-KDD (sampled version: `kdd_sample.csv`)
- **Records**: Network connection logs with 41 features
- **Features**:
  - Numeric: `duration`, `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`, `count`, etc.
  - Categorical: `protocol_type`, `service`, `flag`
- **Labels**:
  - `normal` → 0
  - `attack` → 1 (all attack types merged into binary classification)

The dataset is preprocessed using `StandardScaler` for numeric features and `OneHotEncoder` for categorical features.

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone Repository

```bash
git clone https://github.com/ash-kev/IDS.git
cd IDS
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

### Dataset Setup

Place the `kdd_sample.csv` file in the project root directory. Ensure it contains the required columns as per the KDD99 format.

## Usage

### Training the Model

Run the training script to build and train the neural network:

```bash
python Model_code.py
```

This will:
- Load and preprocess the data
- Train the model with class weights and early stopping
- Evaluate on the test set
- Display confusion matrix and classification report

### Loading a Pre-trained Model

If you have a saved model (`ids_model.h5`), load it for predictions:

```python
import tensorflow as tf

model = tf.keras.models.load_model('ids_model.h5')
```

### Making Predictions

Use the trained model to predict on new data. Here's a sample prediction function from the codebase:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Assuming preprocess is fitted as in training
def live_predict(sample_dict, preprocess, model):
    df = pd.DataFrame([sample_dict])
    X_processed = preprocess.transform(df)
    prob = model.predict(X_processed)[0][0]
    pred = "attack" if prob >= 0.5 else "normal"
    print(f"Prediction: {pred.upper()} | Confidence: {prob:.4f}")
    return pred, prob

# Example normal sample
normal_sample = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 181,
    'dst_bytes': 5450,
    'land': 0,
    'wrong_fragment': 0,
    'urgent': 0,
    'count': 1,
    # ... (add other features as needed)
}

# Load model and preprocess (from training)
# live_predict(normal_sample, preprocess, model)
```

## Model Architecture

The neural network architecture is defined as follows:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

- **Input Layer**: Matches the number of processed features.
- **Hidden Layers**: 64 → 32 → 16 neurons with ReLU activation.
- **Dropout**: 30% dropout after first two hidden layers to prevent overfitting.
- **Output Layer**: Single neuron with sigmoid for binary classification.

Training includes early stopping with patience=5 to monitor loss.

## Evaluation Metrics

After training, the model is evaluated on the test set. Example output:

```
CONFUSION MATRIX:
[[TP, FP],
 [FN, TN]]

CLASSIFICATION REPORT:
              precision    recall  f1-score   support

           0       0.95      0.92      0.93      1000
           1       0.88      0.93      0.90       500

    accuracy                           0.92      1500
   macro avg       0.91      0.92      0.92      1500
weighted avg       0.92      0.92      0.92      1500
```

- **Accuracy**: Overall correctness.
- **Precision**: True positives / (True positives + False positives).
- **Recall**: True positives / (True positives + False negatives).
- **F1-Score**: Harmonic mean of precision and recall.

## Sample Predictions

The script provides sample predictions on test data:

```
Sample Predictions:
Sample 0: Predicted = normal | Confidence = 0.1234
Sample 1: Predicted = attack | Confidence = 0.9876
...
```

## Future Enhancements

- **Live Packet Capture**: Integrate with Wireshark/Scapy for real-time detection.
- **Multi-class Classification**: Extend to classify specific attack types.
- **Deployment**: Use Flask/FastAPI for API-based predictions.
- **Visualization Dashboard**: Build a SOC monitoring interface.
- **Advanced Models**: Incorporate LSTM/CNN for sequential traffic analysis.
- **Model Saving**: Automatically save trained models.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, open an issue first to discuss.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
