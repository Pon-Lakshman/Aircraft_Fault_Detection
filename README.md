# Aircraft Engine Fault Detection using Deep Learning

A Deep Learning-based Predictive Maintenance System for Aircraft Engines using the NASA CMAPSS FD004 dataset.

This project leverages a Bidirectional LSTM Autoencoder to analyze multivariate engine sensor data and identify potential engine degradation before failure occurs. The solution supports proactive maintenance decisions and helps improve aircraft reliability and operational safety.

---

## Project Overview

Aircraft engines operate under harsh conditions and are subject to wear and degradation over time. Unexpected engine failures can result in:

- Increased maintenance costs
- Flight delays and cancellations
- Reduced operational efficiency
- Potential safety risks

The objective of this project is to develop a predictive maintenance system capable of detecting abnormal engine behavior using sensor measurements collected during engine operation.

A Bidirectional LSTM Autoencoder is trained to learn normal engine operating patterns and identify degradation through reconstruction-based anomaly detection.

---

## Project Objectives

- Analyze aircraft engine sensor data from NASA CMAPSS
- Detect early signs of engine degradation
- Predict potential fault conditions
- Support predictive maintenance strategies
- Develop an interactive dashboard for fault analysis
- Reduce the risk of unexpected engine failures

---

## Dataset

### NASA CMAPSS Turbofan Engine Degradation Dataset

The project uses the FD004 subset of the NASA CMAPSS dataset.

### Dataset Characteristics

- Multiple operating conditions
- Multiple fault modes
- 249 simulated engines
- 21 sensor measurements
- 3 operational settings
- Complete engine degradation trajectories

### Files Used

```text
train_FD004.txt
test_FD004.txt
RUL_FD004.txt
```

## Data Preprocessing

The following preprocessing steps were performed before model training.

### 1. Data Cleaning

- Removed unnecessary empty columns
- Assigned meaningful column names
- Verified data consistency
- Checked for missing values

### 2. Feature Selection

Several sensors exhibit little variation and provide limited predictive information.

The following 17 features were selected:

```text
setting1
setting2
setting3
s2
s3
s4
s7
s8
s9
s11
s12
s13
s14
s15
s17
s20
s21
```

### 3. Normalization

Sensor values were scaled to improve neural network training stability.

### 4. Sequence Generation

Sliding window sequences were generated for temporal learning.

```text
Sequence Length = 50 cycles
```

Each sample becomes:

```text
(50 × 17)
```

### 5. Fault Label Generation

Remaining Useful Life (RUL) values were used to generate fault labels.

```text
Healthy Engine = 0
Fault Risk Engine = 1
```

---

## Model Architecture

### Bidirectional LSTM Autoencoder

The model learns normal engine behavior and identifies abnormal degradation patterns through reconstruction error.

### Architecture

```text
Input Sequence (50 × 17)

↓

Bidirectional LSTM (128 Units)

↓

Dropout

↓

Bidirectional LSTM (64 Units)

↓

Latent Representation

↓

RepeatVector

↓

Bidirectional LSTM Decoder

↓

Dropout

↓

Bidirectional LSTM Decoder

↓

TimeDistributed Output Layer
```

### Model Summary

- Input Shape: (50, 17)
- Sequence Length: 50
- Features: 17
- Deep Bidirectional Architecture
- Autoencoder-based Anomaly Detection

---

## Model Performance

### Classification Results

| Metric | Value |
|----------|----------|
| Accuracy | 98% |
| Precision | 55% |
| Recall | 65% |
| F1 Score | 0.597 |
| ROC-AUC | 0.978 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---------|---------|---------|---------|
| Healthy (0) | 0.99 | 0.99 | 0.99 |
| Fault Risk (1) | 0.55 | 0.65 | 0.60 |

### Overall Accuracy

```text
98%
```

### Final Results

```text
Best Threshold = 0.6376

F1 Score = 0.5967

ROC-AUC = 0.9775
```

---

## Model Evaluation

The model was evaluated using multiple performance metrics.

### ROC Curve

Measures the model's ability to distinguish between healthy and faulty engines.

```text
ROC-AUC = 0.9775
```

A value close to 1 indicates excellent discrimination capability.

### Precision-Recall Curve

Used to determine the optimal anomaly threshold for fault detection.

### Confusion Matrix

Provides insight into:

- True Positives
- True Negatives
- False Positives
- False Negatives

### Training and Validation Loss

Monitors model convergence and helps detect overfitting.

---

## Interactive Dashboard

A Streamlit dashboard was developed to demonstrate real-time engine fault analysis.

### Dashboard Features

Manual Sensor Input

Engine Health Assessment

Failure Probability Estimation

Risk Categorization

Maintenance Recommendations

Interactive Fault Probability Gauge

Model Information Panel

### Example Prediction

```text
Failure Probability:
0.0000349

Prediction:
ENGINE IS HEALTHY

Risk Level:
LOW
```

---

## Project Structure

```text
Aircraft_Fault_Detection/
│
├── Data/
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
│
├── Model/
│   └── fault_detector.keras
│
├── Aircraft_Fault_Detection.ipynb
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── images/
    ├── confusion_matrix.png
    ├── dashboard1.png
    ├── dashboard2.png
    └── roc_curve.png
    └── training_loss.png
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Aircraft_Fault_Detection.git

cd Aircraft_Fault_Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Dashboard

```bash
streamlit run app.py
```

The application will launch at:

```text
http://localhost:8501
```

---

## Technologies Used

### Programming Language

- Python

### Deep Learning

- TensorFlow
- Keras

### Data Processing

- NumPy
- Pandas
- Scikit-Learn

### Visualization

- Matplotlib
- Plotly

### Deployment

- Streamlit

---

## Learning Outcomes

This project provided hands-on experience in:

- Predictive Maintenance
- Time Series Analysis
- Aircraft Engine Health Monitoring
- Deep Learning for Anomaly Detection
- Bidirectional LSTM Networks
- Autoencoder Architectures
- Threshold Optimization
- Model Evaluation
- Dashboard Development
- End-to-End Machine Learning Workflows

---

## Future Enhancements

Potential improvements include:

- Remaining Useful Life (RUL) Prediction
- Real-Time Sensor Data Streaming
- Explainable AI using SHAP
- Fleet-Level Monitoring Dashboard
- Cloud Deployment

---

## Screenshots

### Dashboard

Add screenshot:

```text
images/dashboard.png
```

### ROC Curve

Add screenshot:

```text
images/roc_curve.png
```

### Confusion Matrix

Add screenshot:

```text
images/confusion_matrix.png
```

### Training Loss

Add screenshot:

```text
images/training_loss.png
```

---

## Author

**Pon Lakshman**

B.Sc. (Honours) Data Science and Artificial Intelligence

Indian Institute of Technology Guwahati

---