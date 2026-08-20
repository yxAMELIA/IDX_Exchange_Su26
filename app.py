
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="California Home Value Predictor",
    page_icon="🏠",
    layout="centered"
)


# ============================================================
# Model Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "week7_xgboost_model.joblib"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "week7_xgboost_preprocessor.joblib"
)

METADATA_PATH = os.path.join(
    MODEL_DIR,
    "week7_xgboost_metadata.json"
)


# ============================================================
# Load Model, Preprocessor, and Metadata
# ============================================================

@st.cache_resource
def load_model_artifacts():

    model = joblib.load(MODEL_PATH)

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model, preprocessor, metadata


try:
    model, preprocessor, metadata = load_model_artifacts()

except Exception as error:

    st.error(
        "Unable to load the trained model files."
    )

    st.exception(error)

    st.stop()


# ============================================================
# App Header
# ============================================================

st.title("🏠 California Home Value Predictor")

st.write(
    """
    Enter the property characteristics below to estimate
    the expected closing price of a California
    single-family residence.
    """
)

st.caption(
    "Powered by an XGBoost automated valuation model "
    "trained on CRMLS sold-property data."
)


# ============================================================
# User Inputs
# ============================================================

st.subheader("Property Information")

living_area = st.number_input(
    "Living Area (sq ft)",
    min_value=100.0,
    max_value=20000.0,
    value=2000.0,
    step=100.0
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)

lot_size = st.number_input(
    "Lot Size (sq ft)",
    min_value=100.0,
    max_value=500000.0,
    value=6000.0,
    step=500.0
)


# ============================================================
# Build Model Input
# ============================================================

def build_input_dataframe(
    living_area,
    bedrooms,
    bathrooms,
    lot_size,
    metadata
):

    numeric_features = metadata[
        "numeric_features"
    ]

    categorical_features = metadata[
        "categorical_features"
    ]

    all_features = (
        numeric_features
        + categorical_features
    )

    # Create all features required by the trained model.
    # Features not entered by the user remain missing
    # and will be handled by the saved preprocessor.

    input_data = {
        feature: np.nan
        for feature in all_features
    }

    # User-provided features

    input_data["LivingArea"] = float(
        living_area
    )

    input_data["BedroomsTotal"] = float(
        bedrooms
    )

    input_data[
        "BathroomsTotalInteger"
    ] = float(
        bathrooms
    )

    input_data[
        "LotSizeSquareFeet"
    ] = float(
        lot_size
    )

    # Missing indicators for supplied variables

    if "LivingArea_missing" in input_data:
        input_data[
            "LivingArea_missing"
        ] = 0.0

    if (
        "BathroomsTotalInteger_missing"
        in input_data
    ):
        input_data[
            "BathroomsTotalInteger_missing"
        ] = 0.0

    if (
        "LotSizeSquareFeet_missing"
        in input_data
    ):
        input_data[
            "LotSizeSquareFeet_missing"
        ] = 0.0

    input_df = pd.DataFrame(
        [input_data]
    )

    return input_df


# ============================================================
# Prediction
# ============================================================

st.divider()

if st.button(
    "Predict Home Value",
    type="primary",
    use_container_width=True
):

    input_df = build_input_dataframe(
        living_area=living_area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        lot_size=lot_size,
        metadata=metadata
    )

    try:

        # Apply the exact preprocessing pipeline
        # learned during model training.

        processed_input = preprocessor.transform(
            input_df
        )

        # Generate prediction.

        prediction = model.predict(
            processed_input
        )

        predicted_price = float(
            prediction[0]
        )

        st.success(
            "Prediction completed successfully."
        )

        st.subheader(
            "Estimated Property Value"
        )

        st.metric(
            label="Predicted Close Price",
            value=f"${predicted_price:,.0f}"
        )

        # Check whether prediction is outside
        # the main training target range.

        lower_bound = metadata.get(
            "target_lower_bound"
        )

        upper_bound = metadata.get(
            "target_upper_bound"
        )

        if (
            lower_bound is not None
            and upper_bound is not None
        ):

            if (
                predicted_price < lower_bound
                or predicted_price > upper_bound
            ):

                st.warning(
                    "This prediction falls outside the "
                    "main closing-price range used during "
                    "model training."
                )

    except Exception as error:

        st.error(
            "The model could not generate a prediction."
        )

        st.exception(error)


# ============================================================
# Model Information
# ============================================================

st.divider()

with st.expander("About This Model"):

    st.write(
        f"""
        **Model:** {metadata["model"]}

        **Training window:** {metadata["training_window_months"]} months

        **Validation month:** {metadata["validation_month"]}

        **Test month:** {metadata["test_month"]}

        **Test R²:** {metadata["test_metrics"]["R2"]:.4f}

        **Test MAE:** ${metadata["test_metrics"]["MAE"]:,.0f}

        **Test MAPE:** {metadata["test_metrics"]["MAPE"]:.2f}%

        **Test MdAPE:** {metadata["test_metrics"]["MdAPE"]:.2f}%
        """
    )

    st.info(
        """
        This simplified application uses four user-provided
        property characteristics: living area, bedrooms,
        bathrooms, and lot size.

        Other features required by the trained model are
        handled using preprocessing defaults learned from
        the training data. Therefore, this application is
        intended as a simplified AVM demonstration rather
        than a complete location-specific property appraisal.
        """
    )


# ============================================================
# Footer
# ============================================================

st.caption(
    "Automated Valuation Model (AVM) | "
    "California Single-Family Residences"
)
