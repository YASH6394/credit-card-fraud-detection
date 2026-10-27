import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FinGuard AI | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = os.path.join("models", "fraud_model.pkl")
FEATURE_PATH = os.path.join("models", "feature_names.pkl")

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #38bdf8;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #263449;
        padding: 18px;
        border-radius: 12px;
    }

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🛡️ FinGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Credit Card Fraud Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# CHECK MODEL FILES
# --------------------------------------------------

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error(
        "Trained model not found. Run "
        "`python train_model.py` before starting the dashboard."
    )
    st.stop()

try:
    model, expected_features = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Control Panel")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fraud Detection",
        "Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Upload transaction data containing the same feature "
    "columns used during model training."
)

# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def prepare_transactions(data):
    """Validate and arrange uploaded transaction features."""

    missing = [
        column for column in expected_features
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing[:10]}"
        )

    features = data[expected_features].copy()

    for column in features.columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise"
        )

    values = features.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains missing or infinite feature values."
        )

    return features


def predict_transactions(data):
    """Return transaction predictions and fraud probabilities."""

    features = prepare_transactions(data)

    predictions = model.predict(features).astype(int)

    result = data.copy()

    result["Prediction"] = np.where(
        predictions == 1,
        "Potential Fraud",
        "Legitimate"
    )

    result["Fraud Probability (%)"] = (
        model.predict_proba(features)[:, 1] * 100
    ).round(2)

    return result


# --------------------------------------------------
# PAGE 1: DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("📊 Transaction Analytics")

    st.write(
        "Upload a labelled transaction dataset to explore "
        "transaction patterns and model predictions."
    )

    file = st.file_uploader(
        "Upload labelled transaction CSV",
        type=["csv"],
        key="dashboard_upload"
    )

    if file is not None:

        try:
            data = pd.read_csv(file)

            if data.empty:
                st.warning("The uploaded CSV is empty.")
                st.stop()

            st.success("Dataset loaded successfully.")

            st.write("Dataset preview")
            st.dataframe(data.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Transactions",
                f"{len(data):,}"
            )

            if "Class" in data.columns:

                fraud_count = int(
                    (data["Class"] == 1).sum()
                )

                legitimate_count = int(
                    (data["Class"] == 0).sum()
                )

                col2.metric(
                    "Actual Fraud Cases",
                    f"{fraud_count:,}"
                )

                col3.metric(
                    "Legitimate Transactions",
                    f"{legitimate_count:,}"
                )

                chart_data = pd.DataFrame({
                    "Transaction Type": [
                        "Legitimate",
                        "Fraud"
                    ],
                    "Count": [
                        legitimate_count,
                        fraud_count
                    ]
                })

                fig = px.pie(
                    chart_data,
                    names="Transaction Type",
                    values="Count",
                    title="Actual Transaction Distribution",
                    hole=0.55
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info(
                    "No Class column found. Actual fraud statistics "
                    "are unavailable for this dataset."
                )

                st.caption(
                    "Use the Fraud Detection page to generate predictions."
                )

            if "Amount" in data.columns:

                amounts = pd.to_numeric(
                    data["Amount"],
                    errors="coerce"
                )

                amount_data = pd.DataFrame({
                    "Amount": amounts
                }).dropna()

                fig = px.histogram(
                    amount_data,
                    x="Amount",
                    nbins=50,
                    title="Transaction Amount Distribution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as error:
            st.error(f"Could not analyse the dataset: {error}")


# --------------------------------------------------
# PAGE 2: FRAUD DETECTION
# --------------------------------------------------

elif page == "Fraud Detection":

    st.subheader("🚨 Transaction Fraud Detection")

    st.write(
        "Upload a CSV containing the same input features "
        "used during model training."
    )

    uploaded_file = st.file_uploader(
        "Choose transaction CSV",
        type=["csv"],
        key="prediction_upload"
    )

    if uploaded_file is not None:

        try:
            transaction_data = pd.read_csv(uploaded_file)

            if transaction_data.empty:
                st.warning("The uploaded CSV is empty.")
                st.stop()

            st.write(
                f"Uploaded records: {len(transaction_data):,}"
            )

            st.dataframe(
                transaction_data.head(10),
                use_container_width=True
            )

            if len(transaction_data) > 100_000:
                st.warning(
                    "Large dataset detected. Processing may take time."
                )

            if st.button(
                "🔍 Analyse Transactions",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analysing transactions..."
                ):

                    results = predict_transactions(
                        transaction_data
                    )

                fraud_mask = (
                    results["Prediction"] == "Potential Fraud"
                )

                fraud_count = int(fraud_mask.sum())

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Transactions Analysed",
                    f"{len(results):,}"
                )

                col2.metric(
                    "Flagged Transactions",
                    f"{fraud_count:,}"
                )

                col3.metric(
                    "Flagged Percentage",
                    f"{fraud_count / len(results) * 100:.2f}%"
                )

                st.divider()

                st.subheader("Prediction Results")

                filter_option = st.selectbox(
                    "Filter transactions",
                    [
                        "All Transactions",
                        "Potential Fraud",
                        "Legitimate"
                    ]
                )

                if filter_option == "Potential Fraud":
                    display_results = results[
                        fraud_mask
                    ]

                elif filter_option == "Legitimate":
                    display_results = results[
                        ~fraud_mask
                    ]

                else:
                    display_results = results

                st.dataframe(
                    display_results,
                    use_container_width=True
                )

                fig = px.pie(
                    names=[
                        "Potential Fraud",
                        "Legitimate"
                    ],
                    values=[
                        fraud_count,
                        len(results) - fraud_count
                    ],
                    title="Model Prediction Distribution",
                    hole=0.55
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.warning(
                    "Predictions are model estimates, not confirmed fraud. "
                    "Review flagged transactions before taking action."
                )

        except Exception as error:
            st.error(
                f"Prediction failed. Check the uploaded columns and data. "
                f"Details: {error}"
            )


# --------------------------------------------------
# PAGE 3: MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "This application loads the trained model generated "
        "by train_model.py."
    )

    st.markdown("""
    **Models evaluated during training**

    - Logistic Regression
    - Random Forest

    **Evaluation metrics**

    - Precision
    - Recall
    - F1-score
    - Average Precision

    **Input features**

    - Time
    - Amount
    - Anonymised PCA-derived features, including V1–V28
    """)

    st.info(
        "The exact feature columns depend on the dataset used "
        "during training. Uploaded files must contain all required "
        "features in the expected format."
    )

    comparison_path = os.path.join(
        "reports",
        "model_comparison.csv"
    )

    if os.path.exists(comparison_path):

        st.subheader("Model Performance Comparison")

        comparison = pd.read_csv(
            comparison_path
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        if "Average Precision" in comparison.columns:

            fig = px.bar(
                comparison,
                x="Model",
                y="Average Precision",
                title="Average Precision by Model",
                text_auto=".3f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

st.divider()

st.caption(
    "FinGuard AI | Educational Machine Learning Project | "
    "Not a production banking security system"
)

