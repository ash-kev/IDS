# IDS
A basic and easy to understand cnn based model trained upon kd99 dataset to identify suspicious network behavior . 

A Machine Learning–based Intrusion Detection System (IDS) that classifies network traffic as Normal or Attack using a Neural Network trained on the KDD99 / NSL-KDD dataset.

# Features

-Binary intrusion detection (Normal vs Attack)
-End-to-end ML pipeline (Data → Preprocessing → Model → Evaluation)
-Handles numeric + categorical network features
-Class imbalance handling using class weights
-Early stopping to prevent overfitting

# Model Overview

-Model Type: Feed-Forward Artificial Neural Network (ANN)
-Framework: TensorFlow / Keras
-Loss Function: Binary Cross-Entropy
-Optimizer: Adam
-Output: Attack probability (Sigmoid)

# Dataset

-Dataset: KDD99 / NSL-KDD (sampled)
-Records: Network connection logs
-Features: 41 (numeric + categorical)
-Labels:
  -normal → 0
  -attack → 1 (all attack types merged)

# Installation & Setup

Clone Repository:

```git clone https://github.com/yourusername/ids-neural-network.git```
```cd ids-neural-network```


Install Dependencies:
```pip install pandas numpy scikit-learn tensorflow matplotlib```

Dataset:
-Place kdd_sample.csv in the project root directory.

Run the Model
```python nn_f_v2.py```

# Evaluation Metrics

-Accuracy
-Precision
-Recall (important for IDS)
-F1-Score
-Confusion Matrix

# Future Enhancements

-Integration with live packet capture (Wireshark / Scapy)
-Multi-class attack classification
-Deployment using Flask / FastAPI
-Visualization dashboard for SOC monitoring
-Use of LSTM/CNN for sequential traffic analysis
