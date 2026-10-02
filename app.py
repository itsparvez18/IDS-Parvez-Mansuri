import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0b1120;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    font-weight: 700;
}

.dashboard-title {
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    font-size: 16px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.metric-card {
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    background: rgba(128, 128, 128, 0.08);
}

.metric-title {
    font-size: 14px;
    opacity: 0.7;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
}

.status-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.25);
}

.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 40px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("models/ids_model.joblib")
DATASET_PATH = Path("data/processed/cleaned_ddos_dataset.csv")

CONFUSION_PATH = Path("results/confusion_matrix.png")
COMPARISON_PATH = Path("results/model_comparison.png")
DISTRIBUTION_PATH = Path("results/class_distribution.png")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception:
    model = None
    model_loaded = False

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


try:
    df = load_dataset()
    dataset_loaded = True
except Exception:
    df = None
    dataset_loaded = False

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🛡️ IDS")

    st.caption("AI Network Intrusion Detection")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📊 Dataset",
            "🤖 Model Performance",
            "🔍 Prediction"
        ]
    )

    st.markdown("---")

    st.markdown("### System")

    if model_loaded:
        st.success("Model Online")
    else:
        st.error("Model Offline")

    if dataset_loaded:
        st.success("Dataset Available")
    else:
        st.warning("Dataset Missing")

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="dashboard-title">🛡️ AI Network Intrusion Detection System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Machine Learning Based Network Traffic Monitoring & DDoS Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🤖 Detection Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "🎯 Accuracy",
            "100.00%"
        )

    with col3:
        st.metric(
            "🔎 Precision",
            "100.00%"
        )

    with col4:
        st.metric(
            "📡 Recall",
            "99.99%"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:

        if model_loaded:

            st.success(
                "🟢 Detection Engine Online"
            )

            st.write(
                "Random Forest intrusion detection model "
                "loaded successfully."
            )

        else:

            st.error(
                "🔴 Detection Engine Offline"
            )

    with col2:

        if dataset_loaded:

            st.success(
                "🟢 Network Dataset Available"
            )

            st.write(
                f"{len(df):,} processed network-flow records available."
            )

        else:

            st.warning(
                "🟡 Dataset unavailable"
            )

    st.markdown("---")

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    if dataset_loaded:

        st.subheader("Network Traffic Overview")

        benign_count = int((df["Label"] == 0).sum())
        ddos_count = int((df["Label"] == 1).sum())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Network Flows",
                f"{len(df):,}"
            )

        with col2:
            st.metric(
                "BENIGN Traffic",
                f"{benign_count:,}"
            )

        with col3:
            st.metric(
                "DDoS Traffic",
                f"{ddos_count:,}"
            )

        chart_data = pd.DataFrame(
            {
                "Traffic Type": ["BENIGN", "DDoS"],
                "Records": [benign_count, ddos_count]
            }
        )

        st.bar_chart(
            chart_data.set_index("Traffic Type")
        )

# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.title("📊 Dataset Analysis")

    if not dataset_loaded:

        st.error("Dataset could not be loaded.")

    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Records",
                f"{len(df):,}"
            )

        with col2:
            st.metric(
                "Features",
                f"{df.shape[1] - 1}"
            )

        with col3:
            size_mb = DATASET_PATH.stat().st_size / (1024 ** 2)

            st.metric(
                "Dataset Size",
                f"{size_mb:.1f} MB"
            )

        st.markdown("---")

        st.subheader("Traffic Distribution")

        distribution = df["Label"].value_counts().sort_index()

        distribution_df = pd.DataFrame(
            {
                "Traffic Type": ["BENIGN", "DDoS"],
                "Records": [
                    int(distribution.get(0, 0)),
                    int(distribution.get(1, 0))
                ]
            }
        )

        st.bar_chart(
            distribution_df.set_index("Traffic Type")
        )

        st.markdown("---")

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(100),
            use_container_width=True,
            height=450
        )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Performance of the machine-learning models evaluated "
        "on the held-out test split."
    )

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", "100.00%")

    with col2:
        st.metric("Precision", "100.00%")

    with col3:
        st.metric("Recall", "99.99%")

    with col4:
        st.metric("F1 Score", "100.00%")

    st.markdown("---")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    if CONFUSION_PATH.exists():

        st.image(
            str(CONFUSION_PATH),
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    st.subheader("Model Comparison")

    if COMPARISON_PATH.exists():

        st.image(
            str(COMPARISON_PATH),
            use_container_width=True
        )

    else:

        st.warning(
            "Model comparison image not found."
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Class Distribution")

    if DISTRIBUTION_PATH.exists():

        st.image(
            str(DISTRIBUTION_PATH),
            use_container_width=True
        )

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🔍 Prediction":

    st.title("🔍 Network Traffic Detection")

    st.write(
        "Upload a CSV containing network-flow features and "
        "the AI model will classify each record."
    )

    st.info(
        "Expected input: numeric network-flow features similar "
        "to the processed training dataset."
    )

    uploaded_file = st.file_uploader(
        "📁 Upload Network Traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.success(
                f"File loaded successfully — {len(data):,} records."
            )

            st.subheader("Uploaded Data")

            st.dataframe(
                data.head(50),
                use_container_width=True
            )

            # ------------------------------------------------
            # PREPARE FEATURES
            # ------------------------------------------------

            if "Label" in data.columns:

                X = data.drop(
                    columns=["Label"]
                )

            else:

                X = data.copy()

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

            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            X = X.select_dtypes(
                include=[np.number]
            )

            X = X.fillna(
                X.median()
            )

            # ------------------------------------------------
            # FEATURE COUNT CHECK
            # ------------------------------------------------

            if X.shape[1] != 80:

                st.warning(
                    f"The uploaded file contains {X.shape[1]} "
                    f"numeric features, while the trained model "
                    f"expects 80 features."
                )

            else:

                st.success(
                    "Feature format matches the trained model."
                )

                # --------------------------------------------
                # PREDICTION BUTTON
                # --------------------------------------------

                if st.button(
                    "🚨 Detect Intrusions",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing network traffic..."
                    ):

                        predictions = model.predict(X)

                    results = np.where(
                        predictions == 1,
                        "DDoS",
                        "BENIGN"
                    )

                    result_df = data.copy()

                    result_df["Prediction"] = results

                    ddos_count = int(
                        np.sum(predictions == 1)
                    )

                    benign_count = int(
                        np.sum(predictions == 0)
                    )

                    st.markdown("---")

                    st.subheader(
                        "🚨 Detection Summary"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Total Records",
                            f"{len(results):,}"
                        )

                    with col2:
                        st.metric(
                            "BENIGN",
                            f"{benign_count:,}"
                        )

                    with col3:
                        st.metric(
                            "DDoS",
                            f"{ddos_count:,}"
                        )

                    if ddos_count > 0:

                        st.error(
                            f"⚠️ DDoS traffic detected in "
                            f"{ddos_count:,} records."
                        )

                    else:

                        st.success(
                            "🟢 No DDoS traffic detected."
                        )

                    st.markdown("---")

                    st.subheader(
                        "Detection Results"
                    )

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        height=450
                    )

                    # ----------------------------------------
                    # DOWNLOAD
                    # ----------------------------------------

                    result_csv = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "📥 Download Detection Results",
                        result_csv,
                        "ids_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Error processing the uploaded file: {e}"
            )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'AI Network Intrusion Detection System • '
    'Machine Learning Based DDoS Detection'
    '</div>',
    unsafe_allow_html=True
)