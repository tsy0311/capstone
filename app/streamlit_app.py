"""
Streamlit dashboard for Startup Risk Profiling and Funding Prediction.

Features:
- Single startup prediction with risk categorization
- Batch prediction via CSV upload
- Model information display
- SHAP-based explanation (if available)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import (
    load_model, predict_single, predict_batch,
    explain_prediction_shap, get_model_info,
    RISK_CATEGORIES, RISK_LABELS
)


@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "best_model.joblib"
    )
    try:
        return load_model(model_path)
    except FileNotFoundError:
        return None


def main():
    st.set_page_config(
        page_title="Startup Risk Profiler",
        page_icon="📊",
        layout="wide"
    )

    st.title("LLM-Based Startup Risk Profiling & Funding Outcome Prediction")
    st.markdown("---")

    model = load_trained_model()

    if model is None:
        st.error(
            "Model not found. Run `python -m src.train` first to train the model."
        )
        st.info(
            "1. Download dataset from Kaggle: "
            "[startup-success-prediction](https://www.kaggle.com/datasets/manishkc06/startup-success-prediction)\n"
            "2. Place CSV in `data/startup data.csv`\n"
            "3. Run `python -m src.train`"
        )
        return

    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", [
        "Single Prediction", "Batch Prediction", "Model Info"
    ])

    if mode == "Single Prediction":
        render_single_prediction(model)
    elif mode == "Batch Prediction":
        render_batch_prediction(model)
    elif mode == "Model Info":
        render_model_info(model)


def render_single_prediction(model):
    """Single startup prediction form."""
    st.header("Single Startup Prediction")
    st.write("Enter startup attributes to predict funding outcome probability.")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Startup Name", "ExampleStartup")
        category = st.selectbox("Category", [
            "software", "web", "mobile", "enterprise", "advertising",
            "gamesvideo", "ecommerce", "biotech", "consulting", "other"
        ], index=0)
        state = st.selectbox("State Code", [
            "CA", "NY", "MA", "TX", "other"
        ], index=0)
        city = st.text_input("City", "San Francisco")
        founded_year = st.number_input(
            "Founded Year", min_value=1990, max_value=2026, value=2018
        )

    with col2:
        funding_total = st.number_input(
            "Total Funding (USD)", min_value=0, value=5000000, step=500000
        )
        funding_rounds = st.number_input(
            "Funding Rounds", min_value=0, max_value=10, value=3
        )
        first_funding = st.date_input("First Funding Date", value=pd.Timestamp("2018-06-01"))
        last_funding = st.date_input("Last Funding Date", value=pd.Timestamp("2022-01-15"))
        relationships = st.number_input(
            "Relationships (partners/advisors)", min_value=0, max_value=100, value=5
        )

    # Advanced options expander
    with st.expander("Advanced Options (optional)"):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            has_vc = st.checkbox("Has VC backing", value=False)
            has_angel = st.checkbox("Has angel investor", value=False)
            milestones = st.number_input("Milestones", min_value=0, max_value=20, value=2)
            avg_participants = st.number_input(
                "Avg Participants per Round", min_value=0.0, max_value=20.0, value=3.0, step=0.5
            )
        with adv_col2:
            is_top500 = st.checkbox("Is Top 500 Startup", value=False)
            age_first_funding = st.number_input(
                "Age at First Funding (years)", min_value=0.0, max_value=30.0, value=1.0, step=0.5
            )
            age_last_funding = st.number_input(
                "Age at Last Funding (years)", min_value=0.0, max_value=30.0, value=4.0, step=0.5
            )
            age_first_milestone = st.number_input(
                "Age at First Milestone (years)", min_value=0.0, max_value=30.0, value=2.0, step=0.5
            )
            age_last_milestone = st.number_input(
                "Age at Last Milestone (years)", min_value=0.0, max_value=30.0, value=5.0, step=0.5
            )

    threshold = st.slider(
        "Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Adjust to trade off precision vs recall based on business cost"
    )

    if st.button("Predict", type="primary"):
        startup_data = {
            "name": name,
            "category_code": category,
            "state_code": state,
            "city": city,
            "founded_year": int(founded_year),
            "founded_at": f"{int(founded_year)}-01-01",
            "funding_total_usd": int(funding_total),
            "funding_rounds": int(funding_rounds),
            "first_funding_at": str(first_funding),
            "last_funding_at": str(last_funding),
            "relationships": int(relationships),
            "milestones": int(milestones),
            "avg_participants": float(avg_participants),
            "has_VC": 1 if has_vc else 0,
            "has_angel": 1 if has_angel else 0,
            "is_top500": 1 if is_top500 else 0,
            "age_first_funding_year": float(age_first_funding),
            "age_last_funding_year": float(age_last_funding),
            "age_first_milestone_year": float(age_first_milestone),
            "age_last_milestone_year": float(age_last_milestone),
            "description": f"{name} - {category} startup",
        }

        with st.spinner("Predicting..."):
            result = predict_single(model, startup_data, threshold=threshold)

        st.markdown("---")

        # Risk category color
        risk_colors = {
            "low": "green", "medium": "orange",
            "high": "red", "very_high": "darkred"
        }
        color = risk_colors.get(result["risk_category"], "gray")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Outcome", result["predicted_outcome"])

        with col2:
            st.metric("Success Probability", f"{result['funding_success_probability']:.1%}")

        with col3:
            st.metric("Risk Level", result["risk_category"].upper())

        st.markdown(f"**{result['risk_label']}**")

        st.json(result)

        # SHAP explanation
        st.subheader("Explanation (SHAP)")
        explanation = explain_prediction_shap(model, startup_data)
        if explanation:
            st.write("Top feature contributions:")
            contrib_df = pd.DataFrame(explanation["feature_contributions"])
            st.dataframe(contrib_df, use_container_width=True)
        else:
            st.info("SHAP explanation not available. Install SHAP: pip install shap")


def render_batch_prediction(model):
    """Batch prediction via CSV upload."""
    st.header("Batch Prediction")
    st.write("Upload a CSV file with startup data to get batch predictions.")
    st.info(
        "CSV should contain columns like: name, category_code, state_code, city, "
        "funding_total_usd, funding_rounds, founded_year/at, first_funding_at, "
        "last_funding_at, relationships, milestones, etc. "
        "Missing columns will be filled with defaults."
    )

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} startups")
        st.dataframe(df.head(), use_container_width=True)

        threshold = st.slider(
            "Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05
        )

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Predicting..."):
                results = predict_batch(model, df, threshold=threshold)

            st.success(f"Prediction complete for {len(results)} startups")

            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                n_success = (results["predicted_outcome"] == "Success").sum()
                st.metric("Predicted Success", n_success)
            with col2:
                n_failure = (results["predicted_outcome"] == "Failure").sum()
                st.metric("Predicted Failure", n_failure)
            with col3:
                avg_prob = results["funding_success_probability"].mean()
                st.metric("Avg Success Prob", f"{avg_prob:.1%}")

            # Full results
            st.dataframe(results, use_container_width=True)

            # Download
            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions", csv,
                "startup_predictions.csv", "text/csv"
            )


def render_model_info(model):
    """Display model metadata."""
    st.header("Model Information")
    info = get_model_info(model)

    st.write(f"**Model Type:** {info['model_type']}")
    st.write(f"**Pipeline Steps:** {', '.join(info['pipeline_steps'])}")

    st.subheader("Model Parameters")
    st.json(info["model_params"])

    st.subheader("Risk Categories")
    risk_df = pd.DataFrame([
        {"Category": k.upper(), "Probability Range": f"{lo:.1f} - {hi:.1f}",
         "Description": RISK_LABELS[k]}
        for k, (lo, hi) in RISK_CATEGORIES.items()
    ])
    st.table(risk_df)

    st.markdown("---")
    st.caption(
        "⚠️ Predictions are decision-support outputs, not certainties. "
        "Always apply human judgment and due diligence."
    )


if __name__ == "__main__":
    main()
