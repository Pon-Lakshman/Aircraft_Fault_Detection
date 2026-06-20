import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Aircraft Engine Fault Detection",
    page_icon="✈️",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

MODEL_PATH = r"Models\best_fd004_bilstm_classifier.keras"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file not found.\n\nExpected location:\n{MODEL_PATH}"
    )
    st.stop()

@st.cache_resource
def load_fault_model():
    return load_model(MODEL_PATH)

model = load_fault_model()

# =====================================================
# HEADER
# =====================================================

st.title("✈️ Aircraft Engine Fault Detection System")

st.markdown("""
Deep Learning Based Predictive Maintenance System using
NASA CMAPSS FD004 Dataset and Bidirectional LSTM.
""")

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙ Settings")

threshold = st.sidebar.slider(
    "Fault Detection Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.64,
    step=0.01
)

# =====================================================
# FEATURES
# =====================================================

feature_columns = [
    'setting1',
    'setting2',
    'setting3',
    's2',
    's3',
    's4',
    's7',
    's8',
    's9',
    's11',
    's12',
    's13',
    's14',
    's15',
    's17',
    's20',
    's21'
]

# =====================================================
# INPUT SECTION
# =====================================================

st.subheader("Engine Sensor Inputs")

col1, col2, col3 = st.columns(3)

inputs = {}

for i, feature in enumerate(feature_columns):

    if i % 3 == 0:
        with col1:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f"
            )

    elif i % 3 == 1:
        with col2:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f"
            )

    else:
        with col3:
            inputs[feature] = st.number_input(
                feature,
                value=0.0,
                format="%.4f"
            )

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Analyze Engine"):

    # Create dataframe
    input_df = pd.DataFrame([inputs])

    # Convert to numpy
    data = input_df.values

    # Create sequence
    sequence_length = 50

    sequence = np.repeat(
        data,
        sequence_length,
        axis=0
    )

    sequence = np.expand_dims(sequence, axis=0)

    # Prediction
    probability = float(
        model.predict(
            sequence,
            verbose=0
        )[0][0]
    )

    prediction = int(probability > threshold)

    health_score = max(
        0,
        (1 - probability) * 100
    )

    st.divider()

    st.subheader("📊 Prediction Results")

    # =================================================
    # METRICS
    # =================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fault Probability",
            f"{probability:.2%}"
        )

    with col2:
        st.metric(
            "Engine Health",
            f"{health_score:.2f}%"
        )

    with col3:
        st.metric(
            "Threshold",
            f"{threshold:.2f}"
        )

    st.progress(float(min(probability, 1.0)))

    # =================================================
    # RISK CATEGORY
    # =================================================

    if probability >= 0.80:

        st.error(
            "Critical Failure Risk"
        )

    elif probability >= 0.60:

        st.warning(
            "High Failure Risk"
        )

    elif probability >= 0.30:

        st.info(
            "Moderate Failure Risk"
        )

    else:

        st.success(
            "Engine Operating Normally"
        )

    # =================================================
    # GAUGE CHART
    # =================================================

    st.subheader("Fault Probability Gauge")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={
                "text": "Fault Probability (%)"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "green"
                    },
                    {
                        "range": [30, 60],
                        "color": "yellow"
                    },
                    {
                        "range": [60, 80],
                        "color": "orange"
                    },
                    {
                        "range": [80, 100],
                        "color": "red"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # MAINTENANCE RECOMMENDATION
    # =================================================

    st.subheader("🛠 Maintenance Recommendation")

    if probability >= 0.80:

        st.error("""
Immediate inspection required.

Possible severe engine degradation detected.

Aircraft should be scheduled for maintenance immediately.
""")

    elif probability >= 0.60:

        st.warning("""
Engine degradation detected.

Monitor closely and schedule inspection soon.
""")

    elif probability >= 0.30:

        st.info("""
Engine is showing early signs of degradation.

Continue monitoring operating conditions.
""")

    else:

        st.success("""
Engine operating within normal range.

No maintenance action required at this time.
""")

# =====================================================
# MODEL INFORMATION
# =====================================================

st.divider()

with st.expander("Model Information"):

    st.markdown("""
### Dataset
NASA CMAPSS FD004

### Model
Bidirectional LSTM Fault Classifier

### Input Features
17 Selected Sensor Features

### Sequence Length
50 Cycles

### Performance

- Accuracy: 98%
- Precision: 56%
- Recall: 63%
- F1 Score: 0.595
- ROC-AUC: 0.975

### Purpose

Predict aircraft engine degradation before failure occurs,
allowing predictive maintenance and reducing operational risk.
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Developed using TensorFlow, Keras, Plotly and Streamlit"
)