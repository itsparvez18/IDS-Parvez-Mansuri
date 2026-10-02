import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real-Time IDS Monitor",
    page_icon="📡",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
}

.monitor-title {
    font-size: 38px;
    font-weight: 800;
}

.monitor-subtitle {
    opacity: 0.7;
    font-size: 16px;
    margin-bottom: 25px;
}

.alert-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    border: 1px solid rgba(255,255,255,0.15);
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("models/ids_model.joblib")
DATASET_PATH = Path("data/processed/cleaned_ddos_dataset.csv")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_status = True
except Exception as e:
    model = None
    model_status = False
    model_error = str(e)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    df = load_dataset()
    dataset_status = True
except Exception:
    df = None
    dataset_status = False

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="monitor-title">📡 Real-Time IDS Monitor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="monitor-subtitle">'
    'AI-powered network traffic monitoring and DDoS detection simulation'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# SYSTEM STATUS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    if model_status:
        st.success("🟢 Detection Engine Online")
    else:
        st.error("🔴 Detection Engine Offline")

with col2:

    if dataset_status:
        st.success("🟢 Traffic Source Available")
    else:
        st.error("🔴 Traffic Source Unavailable")

with col3:

    st.info("🟡 Simulation Mode")

st.markdown("---")

# ============================================================
# SESSION STATE
# ============================================================

if "monitor_running" not in st.session_state:
    st.session_state.monitor_running = False

if "traffic_count" not in st.session_state:
    st.session_state.traffic_count = 0

if "benign_count" not in st.session_state:
    st.session_state.benign_count = 0

if "ddos_count" not in st.session_state:
    st.session_state.ddos_count = 0

if "history" not in st.session_state:
    st.session_state.history = []

if "logs" not in st.session_state:
    st.session_state.logs = []

# ============================================================
# CONTROL PANEL
# ============================================================

st.subheader("🎛️ Monitoring Control")

col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "▶️ Start Monitoring",
        use_container_width=True
    ):
        st.session_state.monitor_running = True
        st.rerun()

with col2:

    if st.button(
        "⏹️ Stop Monitoring",
        use_container_width=True
    ):
        st.session_state.monitor_running = False
        st.rerun()

with col3:

    if st.button(
        "🔄 Reset Statistics",
        use_container_width=True
    ):

        st.session_state.traffic_count = 0
        st.session_state.benign_count = 0
        st.session_state.ddos_count = 0
        st.session_state.history = []
        st.session_state.logs = []

        st.rerun()

# ============================================================
# CURRENT STATUS
# ============================================================

if st.session_state.monitor_running:

    st.success("📡 Monitoring is ACTIVE")

else:

    st.warning("⏸️ Monitoring is STOPPED")

st.markdown("---")

# ============================================================
# METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Flows",
        f"{st.session_state.traffic_count:,}"
    )

with col2:

    st.metric(
        "BENIGN",
        f"{st.session_state.benign_count:,}"
    )

with col3:

    st.metric(
        "DDoS",
        f"{st.session_state.ddos_count:,}"
    )

with col4:

    if st.session_state.traffic_count > 0:

        detection_rate = (
            st.session_state.ddos_count
            / st.session_state.traffic_count
        ) * 100

        st.metric(
            "DDoS Rate",
            f"{detection_rate:.2f}%"
        )

    else:

        st.metric(
            "DDoS Rate",
            "0.00%"
        )

st.markdown("---")

# ============================================================
# LIVE CHART
# ============================================================

st.subheader("📈 Live Traffic Monitoring")

if st.session_state.history:

    chart_df = pd.DataFrame(
        st.session_state.history
    )

    chart_df = chart_df.set_index("Time")

    st.line_chart(
        chart_df[
            ["BENIGN", "DDoS"]
        ]
    )

else:

    st.info(
        "Start monitoring to display live traffic statistics."
    )

# ============================================================
# MONITORING SIMULATION
# ============================================================

if st.session_state.monitor_running:

    if not model_status:

        st.error(
            "The trained model could not be loaded."
        )

        st.stop()

    if not dataset_status:

        st.error(
            "The processed dataset could not be loaded."
        )

        st.stop()

    # --------------------------------------------------------
    # RANDOMLY SELECT NETWORK FLOWS
    # --------------------------------------------------------

    sample_size = min(20, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=None
    )

    # Remove target column
    X = sample.drop(
        columns=["Label"],
        errors="ignore"
    )

    # Remove identifiers
    columns_to_remove = [
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp"
    ]

    X = X.drop(
        columns=[
            col for col in columns_to_remove
            if col in X.columns
        ],
        errors="ignore"
    )

    # Replace infinity
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Numeric features
    X = X.select_dtypes(
        include=[np.number]
    )

    # Fill missing values
    X = X.fillna(
        X.median()
    )

    # --------------------------------------------------------
    # FEATURE CHECK
    # --------------------------------------------------------

    if X.shape[1] != 80:

        st.error(
            f"Expected 80 features, but found "
            f"{X.shape[1]}."
        )

        st.stop()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(X)

    benign_new = int(
        np.sum(predictions == 0)
    )

    ddos_new = int(
        np.sum(predictions == 1)
    )

    # Update counters
    st.session_state.traffic_count += sample_size

    st.session_state.benign_count += benign_new

    st.session_state.ddos_count += ddos_new

    current_time = datetime.now().strftime(
        "%H:%M:%S"
    )

    # Add chart data
    st.session_state.history.append(
        {
            "Time": current_time,
            "BENIGN": benign_new,
            "DDoS": ddos_new
        }
    )

    # Keep last 20 points
    st.session_state.history = (
        st.session_state.history[-20:]
    )

    # --------------------------------------------------------
    # ALERT SYSTEM
    # --------------------------------------------------------

    if ddos_new > 0:

        alert_message = (
            f"⚠️ DDoS traffic detected! "
            f"{ddos_new} suspicious flows detected "
            f"at {current_time}."
        )

        st.error(
            alert_message
        )

        st.session_state.logs.append(
            {
                "Time": current_time,
                "Event": "DDoS DETECTED",
                "Flows": ddos_new,
                "Status": "ALERT"
            }
        )

    else:

        st.success(
            f"🟢 No DDoS detected in current batch "
            f"({sample_size} flows)."
        )

        st.session_state.logs.append(
            {
                "Time": current_time,
                "Event": "Traffic Normal",
                "Flows": benign_new,
                "Status": "NORMAL"
            }
        )

    # Keep last 50 logs
    st.session_state.logs = (
        st.session_state.logs[-50:]
    )

    # --------------------------------------------------------
    # AUTO REFRESH
    # --------------------------------------------------------

    time.sleep(2)

    st.rerun()

# ============================================================
# DETECTION LOG
# ============================================================

st.markdown("---")

st.subheader("📋 Detection Log")

if st.session_state.logs:

    logs_df = pd.DataFrame(
        st.session_state.logs
    )

    st.dataframe(
        logs_df.iloc[::-1],
        use_container_width=True,
        height=350
    )

else:

    st.info(
        "No detection events recorded yet."
    )

# ============================================================
# DOWNLOAD LOG
# ============================================================

if st.session_state.logs:

    log_csv = pd.DataFrame(
        st.session_state.logs
    ).to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "📥 Download Detection Log",
        log_csv,
        "ids_detection_log.csv",
        "text/csv",
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Network Intrusion Detection System • "
    "Real-Time Monitoring Simulation • "
    "Random Forest"
)
