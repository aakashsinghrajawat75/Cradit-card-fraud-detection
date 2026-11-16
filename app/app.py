from __future__ import annotations

import io
import math
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_DIR = PROJECT_DIR / "fraud_model"
MODEL_PATH = MODEL_DIR / "fraud_detection_model.pkl"
THRESHOLD_PATH = MODEL_DIR / "threshold.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"

st.set_page_config(
    page_title="Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1440px;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, 0.18);}
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(37, 99, 235, 0.08), rgba(37, 99, 235, 0.025));
        border: 1px solid rgba(37, 99, 235, 0.14);
        padding: 1rem 1.1rem;
        border-radius: 0.85rem;
        min-height: 108px;
    }
    [data-testid="stMetricLabel"] p {font-size: 0.84rem;}
    [data-testid="stMetricValue"] {font-size: 1.7rem;}
    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(37, 99, 235, 0.35);
        border-radius: 0.8rem;
        padding: 0.3rem;
    }
    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(128, 128, 128, 0.14);
        border-radius: 0.85rem;
        padding: 0.35rem;
    }
    .hero {
        padding: 1.55rem 1.7rem;
        border: 1px solid rgba(37, 99, 235, 0.18);
        border-radius: 1.1rem;
        background: linear-gradient(120deg, rgba(37, 99, 235, 0.11), rgba(14, 165, 233, 0.035));
        margin-bottom: 1.25rem;
    }
    .hero-kicker {
        color: #2563eb;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .hero-title {font-size: 2rem; font-weight: 750; margin: 0.25rem 0;}
    .hero-copy {opacity: 0.75; margin-bottom: 0;}
    .section-note {opacity: 0.7; font-size: 0.9rem;}
    @media (max-width: 700px) {
        .hero {padding: 1.2rem;}
        .hero-title {font-size: 1.55rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def normalize_class_labels(labels: pd.Series) -> pd.Series:
    normalized = labels.astype("string").str.strip().str.lower()
    known_labels = {
        "fraud": 1,
        "fraudulent": 1,
        "true": 1,
        "yes": 1,
        "genuine": 0,
        "legitimate": 0,
        "legit": 0,
        "normal": 0,
        "false": 0,
        "no": 0,
    }
    mapped = normalized.map(known_labels)
    numeric = pd.to_numeric(normalized, errors="coerce")
    values = mapped.fillna(numeric)
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError("The `Class` column must contain only binary 0/1 labels.")
    return values.astype("int64")


@st.cache_resource
def load_model_bundle() -> tuple[Any, float, list[str], int]:
    model = joblib.load(MODEL_PATH)
    threshold = float(joblib.load(THRESHOLD_PATH))
    feature_names = joblib.load(FEATURES_PATH)

    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("The saved feature_names.pkl must contain a list of strings.")
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("The saved model feature names must be non-empty and unique.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The saved fraud threshold must be between 0 and 1.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model must support predict_proba().")

    model_feature_count = getattr(model, "n_features_in_", len(feature_names))
    if model_feature_count != len(feature_names):
        raise ValueError(
            "The saved model expects "
            f"{model_feature_count} features, but feature_names.pkl contains "
            f"{len(feature_names)}."
        )

    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None and list(model_feature_names) != feature_names:
        raise ValueError(
            "The order in feature_names.pkl does not match the saved model."
        )

    classes = np.asarray(getattr(model, "classes_", []))
    fraud_indexes = np.flatnonzero(classes == 1)
    if len(fraud_indexes) != 1:
        raise ValueError("The saved model must expose a single fraud class labeled 1.")

    return model, threshold, feature_names, int(fraud_indexes[0])


def fraud_probabilities(
    model: Any, features: pd.DataFrame, fraud_index: int
) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.ndim != 2 or probabilities.shape[1] <= fraud_index:
        raise ValueError("The saved model returned an invalid probability array.")

    fraud_scores = probabilities[:, fraud_index]
    if len(fraud_scores) != len(features) or not np.isfinite(fraud_scores).all():
        raise ValueError("The saved model returned invalid fraud probabilities.")
    return fraud_scores


def prepare_model_features(
    frame: pd.DataFrame, feature_names: list[str]
) -> pd.DataFrame:
    missing_features = [name for name in feature_names if name not in frame.columns]
    if missing_features:
        raise ValueError(
            "The uploaded CSV is missing model features: "
            + ", ".join(missing_features[:8])
            + (" ..." if len(missing_features) > 8 else "")
        )

    features = frame.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    invalid_rows = features.isna().any(axis=1) | ~np.isfinite(features).all(axis=1)
    if invalid_rows.any():
        raise ValueError(
            f"{int(invalid_rows.sum()):,} row(s) contain missing, non-numeric, "
            "or infinite model feature values. Correct them before scoring."
        )
    return features


def render_empty_state() -> None:
    st.info(
        "Upload a CSV from the sidebar to explore its transactions. "
        "The notebook's source dataset is OpenML data_id=1597."
    )
    st.markdown(
        """
        **Expected dataset columns**

        - `V1` through `V28` and `Amount` for model scoring
        - Optional `Class` (0 = genuine, 1 = fraud) for labeled analysis
        - Optional `Time` for time-based exploration
        """
    )


def render_about(
    model: Any | None,
    threshold: float | None,
    feature_names: list[str] | None,
) -> None:
    st.subheader("About this project")
    st.markdown(
        """
        **Credit Card Fraud Detection** is Aakash's machine-learning project for
        exploring anonymized card transactions and helping reviewers prioritize
        potentially fraudulent activity.

        The companion notebook explores the OpenML credit-card dataset
        (`data_id=1597`), compares classification approaches, handles class
        imbalance, tunes a Random Forest, and selects a fraud probability
        threshold using validation data.
        """
    )

    first, second = st.columns(2)
    with first:
        st.markdown("#### What this dashboard does")
        st.markdown(
            """
            - Summarizes an uploaded transaction CSV.
            - Visualizes feature and transaction patterns.
            - Evaluates predictions when the CSV includes `Class`.
            - Scores transactions and lets reviewers download the results.
            """
        )
    with second:
        st.markdown("#### Model bundle loaded by this app")
        if model is not None and threshold is not None and feature_names is not None:
            pipeline_steps = getattr(model, "steps", [])
            algorithm = (
                " → ".join(name for name, _ in pipeline_steps)
                if pipeline_steps
                else type(model).__name__
            )
            st.markdown(
                f"""
                - Estimator: **{type(model).__name__}**
                - Pipeline: `{algorithm}`
                - Model input: **{len(feature_names)} ordered features**
                - Fraud probability cutoff: **{threshold:.4f}**
                - Feature columns: `V1`–`V28`, `Amount` (from the saved feature list)
                """
            )
        else:
            st.info("The trained model bundle could not be loaded.")

    st.warning(
        "Predictions are for educational and review-support purposes only. "
        "They are not proof of fraud and should not be used as the sole basis "
        "for financial or customer decisions."
    )


def render_overview(frame: pd.DataFrame) -> None:
    st.subheader("Dataset overview")
    transaction_count = len(frame)
    fraud_count = int(frame["Class"].sum()) if "Class" in frame else None
    fraud_rate = fraud_count / transaction_count if fraud_count and transaction_count else 0

    metrics = st.columns(4)
    metrics[0].metric("Transactions", f"{transaction_count:,}")
    metrics[1].metric("Features", f"{frame.shape[1]:,}")
    metrics[2].metric("Fraud cases", f"{fraud_count:,}" if fraud_count is not None else "—")
    metrics[3].metric("Fraud rate", f"{fraud_rate:.2%}" if fraud_count is not None else "—")

    left, right = st.columns((1, 1))
    with left:
        st.markdown("#### Transaction labels")
        if "Class" in frame:
            class_counts = (
                frame["Class"]
                .map({0: "Genuine", 1: "Fraud"})
                .value_counts()
                .reindex(["Genuine", "Fraud"], fill_value=0)
                .rename_axis("Transaction")
                .reset_index(name="Count")
            )
            chart = px.bar(
                class_counts,
                x="Transaction",
                y="Count",
                color="Transaction",
                color_discrete_map={"Genuine": "#2563eb", "Fraud": "#ef4444"},
                text_auto=True,
            )
            chart.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Count")
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.caption("Add a `Class` column to see the genuine/fraud breakdown.")

    with right:
        st.markdown("#### Transaction amount")
        if "Amount" in frame:
            amount_chart = px.histogram(
                frame,
                x="Amount",
                color="Class" if "Class" in frame else None,
                nbins=50,
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={0: "#2563eb", 1: "#ef4444"},
                labels={"Class": "Transaction class"},
            )
            amount_chart.update_layout(yaxis_title="Transactions")
            st.plotly_chart(amount_chart, use_container_width=True)
        else:
            st.caption("No `Amount` column was found.")

    st.markdown("#### Recent rows")
    st.dataframe(frame.head(10), use_container_width=True, hide_index=True)


def render_explore(frame: pd.DataFrame) -> None:
    st.subheader("Explore transaction data")
    st.caption(
        "Use these distributions to inspect the uploaded data. "
        "PCA-transformed features are anonymized and may not have direct business meanings."
    )
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        st.warning("No numeric columns are available to chart.")
        return

    selected_feature = st.selectbox(
        "Feature distribution",
        options=numeric_columns,
        index=numeric_columns.index("Amount") if "Amount" in numeric_columns else 0,
    )
    if "Class" in frame:
        chart = px.histogram(
            frame,
            x=selected_feature,
            color="Class",
            nbins=50,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map={0: "#2563eb", 1: "#ef4444"},
            labels={"Class": "Transaction class"},
        )
    else:
        chart = px.histogram(frame, x=selected_feature, nbins=50, color_discrete_sequence=["#2563eb"])
    chart.update_layout(yaxis_title="Transactions")
    st.plotly_chart(chart, use_container_width=True)

    if "Time" in frame and pd.api.types.is_numeric_dtype(frame["Time"]):
        st.markdown("#### Transactions over time")
        time_frame = frame.assign(
            _time_bucket=(frame["Time"] // 3600).astype("Int64")
        )
        if "Class" in frame:
            time_counts = (
                time_frame.groupby(["_time_bucket", "Class"], observed=True)
                .size()
                .reset_index(name="Transactions")
            )
            time_chart = px.line(
                time_counts,
                x="_time_bucket",
                y="Transactions",
                color="Class",
                color_discrete_map={0: "#2563eb", 1: "#ef4444"},
                labels={"_time_bucket": "Time (hour bucket)", "Class": "Class"},
            )
        else:
            time_counts = (
                time_frame.groupby("_time_bucket", observed=True)
                .size()
                .reset_index(name="Transactions")
            )
            time_chart = px.line(
                time_counts,
                x="_time_bucket",
                y="Transactions",
                labels={"_time_bucket": "Time (hour bucket)"},
            )
        st.plotly_chart(time_chart, use_container_width=True)

    if len(numeric_columns) > 1:
        st.markdown("#### Feature correlations")
        correlation = frame[numeric_columns].corr(numeric_only=True)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        heatmap.update_layout(height=650, margin=dict(l=20, r=20, t=25, b=20))
        st.plotly_chart(heatmap, use_container_width=True)


def render_model_evaluation(
    frame: pd.DataFrame,
    model: Any,
    threshold: float,
    feature_names: list[str],
    fraud_index: int,
) -> None:
    st.subheader("Model evaluation")
    if "Class" not in frame:
        st.info("Upload a labeled CSV with a binary `Class` column to evaluate predictions.")
        return

    actual = normalize_class_labels(frame["Class"])
    if actual.nunique() != 2:
        st.warning("Evaluation metrics require both genuine and fraud labels.")
        return

    try:
        features = prepare_model_features(frame, feature_names)
        probabilities = fraud_probabilities(model, features, fraud_index)
    except ValueError as exc:
        st.error(str(exc))
        return
    predicted = (probabilities >= threshold).astype("int64")

    metric_columns = st.columns(6)
    metric_columns[0].metric("Accuracy", f"{accuracy_score(actual, predicted):.2%}")
    metric_columns[1].metric(
        "Precision", f"{precision_score(actual, predicted, zero_division=0):.2%}"
    )
    metric_columns[2].metric("Recall", f"{recall_score(actual, predicted):.2%}")
    metric_columns[3].metric("F1 score", f"{f1_score(actual, predicted, zero_division=0):.3f}")
    metric_columns[4].metric("ROC-AUC", f"{roc_auc_score(actual, probabilities):.3f}")
    metric_columns[5].metric("PR-AUC", f"{average_precision_score(actual, probabilities):.3f}")

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    matrix_chart = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        x=["Predicted genuine", "Predicted fraud"],
        y=["Actual genuine", "Actual fraud"],
        labels={"x": "", "y": "", "color": "Transactions"},
    )
    matrix_chart.update_layout(title="Confusion matrix", coloraxis_showscale=False)
    st.plotly_chart(matrix_chart, use_container_width=True)
    st.caption(
        f"Fraud threshold loaded from the trained model artifacts: {threshold:.4f}. "
        "Metrics are only meaningful when this CSV is an unseen evaluation set."
    )


def render_scoring(
    frame: pd.DataFrame | None,
    model: Any | None,
    threshold: float | None,
    feature_names: list[str] | None,
    fraud_index: int | None,
    model_error: str | None,
) -> None:
    st.subheader("Score transactions")
    if (
        model is None
        or threshold is None
        or feature_names is None
        or fraud_index is None
    ):
        st.warning("A trained model bundle is not available yet.")
        st.caption(
            "The app expects the model, threshold, and ordered feature names "
            "in the project's `fraud_model/` directory."
        )
        if model_error:
            st.error(f"Could not load the model bundle: {model_error}")
        return

    if frame is None:
        st.info("Upload a CSV containing all model features to score transactions.")
        return

    st.caption(f"Loaded model threshold: {threshold:.4f}")
    if st.button("Score uploaded transactions", type="primary"):
        try:
            features = prepare_model_features(frame, feature_names)
            probabilities = fraud_probabilities(model, features, fraud_index)
        except ValueError as exc:
            st.error(str(exc))
            return

        results = frame.copy()
        results["Fraud probability"] = probabilities
        results["Prediction"] = np.where(
            probabilities >= threshold, "Fraud", "Genuine"
        )

        fraud_predictions = int((probabilities >= threshold).sum())
        first, second = st.columns(2)
        first.metric("Transactions scored", f"{len(results):,}")
        second.metric("Flagged for review", f"{fraud_predictions:,}")
        st.dataframe(results.head(100), use_container_width=True, hide_index=True)
        st.download_button(
            "Download scored CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="fraud_predictions.csv",
            mime="text/csv",
        )
        st.warning(
            "Treat flags as review candidates, not definitive fraud determinations. "
            "Validate the model and threshold for your intended use before deployment."
        )


st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Transaction intelligence</div>
      <div class="hero-title">Credit Card Fraud Detection</div>
      <p class="hero-copy">
        Explore transaction patterns, evaluate model quality, and flag suspicious
        activity for review.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 💳 Fraud Monitor")
    st.caption("A compact workspace for transaction analysis.")
    st.divider()
    st.markdown("### 1 · Load data")
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    st.markdown("### 2 · Choose a view")
    page = st.radio(
        "Dashboard section",
        [
            "Overview",
            "Explore data",
            "Model evaluation",
            "Score transactions",
            "About project",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("**Target:** `Class` · 0 genuine · 1 fraud")
    st.caption("Scoring features: `V1`–`V28` and `Amount`")

data: pd.DataFrame | None = None
if uploaded_file is not None:
    try:
        data = parse_csv(uploaded_file.getvalue())
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
    if data.empty:
        st.error("The uploaded CSV has no rows.")
        st.stop()
    if "Class" in data.columns:
        try:
            data["Class"] = normalize_class_labels(data["Class"])
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

model_bundle: tuple[Any, float, list[str]] | None = None
model_error: str | None = None
artifact_paths = [MODEL_PATH, THRESHOLD_PATH, FEATURES_PATH]
if all(path.is_file() for path in artifact_paths):
    try:
        model_bundle = load_model_bundle()
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        ValueError,
        pickle.UnpicklingError,
    ) as exc:
        model_error = str(exc)
elif any(path.exists() for path in artifact_paths):
    missing = [path.name for path in artifact_paths if not path.is_file()]
    model_error = "Missing model artifact(s): " + ", ".join(missing)

if model_error and page != "Score transactions":
    st.warning(f"Model bundle unavailable: {model_error}")

loaded_model, loaded_threshold, loaded_features, fraud_index = (
    model_bundle if model_bundle is not None else (None, None, None, None)
)

if page == "About project":
    render_about(loaded_model, loaded_threshold, loaded_features)
elif data is None:
    st.markdown("### Start with your transaction data")
    render_empty_state()
elif page == "Overview":
    render_overview(data)
elif page == "Explore data":
    render_explore(data)
elif page == "Model evaluation":
    if model_bundle is None:
        st.warning("Export a trained model bundle before evaluating model predictions.")
    else:
        render_model_evaluation(
            data,
            loaded_model,
            loaded_threshold,
            loaded_features,
            fraud_index,
        )
elif page == "Score transactions":
    render_scoring(
        data,
        loaded_model,
        loaded_threshold,
        loaded_features,
        fraud_index,
        model_error,
    )
