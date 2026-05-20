# ---------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Background */

.main {
    background-color: #f5f7fb;
}

/* Title */

.title {
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    color: #1d3557;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #6c757d;
    margin-bottom: 30px;
}

/* Headers */

h1, h2, h3 {
    color: #1d3557;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #eef2f7;
}

/* Labels */

.stNumberInput label {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #0b2545 !important;
}

/* Input Box */

.stNumberInput div[data-baseweb="input"] {
    border-radius: 10px !important;
    border: 2px solid #dce3ea !important;
    background-color: white !important;
}

/* Input Text */

.stNumberInput input {
    font-size: 18px !important;
    font-weight: 500 !important;
    color: black !important;
}

/* Buttons */

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 10px;
    background-color: #1d3557;
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

/* Metrics */

[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    '<p class="title">House Price Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Machine Learning Prediction using KNN Regression</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

@st.cache_data
def load_data():

    housing = fetch_california_housing()

    X = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    y = pd.Series(housing.target)

    return X, y

X, y = load_data()

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------

@st.cache_resource
def train_model():

    model = KNeighborsRegressor(
        n_neighbors=5
    )

    model.fit(X_train_scaled, y_train)

    return model

model = train_model()

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

y_pred = model.predict(X_test_scaled)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Prediction",
        "Visualizations",
        "Model Performance",
        "Dataset"
    ]
)

# ===================================================
# PREDICTION PAGE
# ===================================================

if page == "Prediction":

    st.header("House Price Prediction")

    st.write("Enter house details below:")

    feature_names = {
        "MedInc": "Median Income",
        "HouseAge": "House Age",
        "AveRooms": "Average Rooms",
        "AveBedrms": "Average Bedrooms",
        "Population": "Population",
        "AveOccup": "Average Occupancy",
        "Latitude": "Latitude",
        "Longitude": "Longitude"
    }

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        input_data = {}

        for i, column in enumerate(X.columns):

            label = feature_names.get(column, column)

            mean_value = int(X[column].mean())

            if i % 2 == 0:

                with col1:

                    value = st.number_input(
                        label,
                        value=mean_value,
                        step=1
                    )

            else:

                with col2:

                    value = st.number_input(
                        label,
                        value=mean_value,
                        step=1
                    )

            input_data[column] = value

        submit = st.form_submit_button(
            "Predict House Price"
        )

    if submit:

        input_df = pd.DataFrame([input_data])

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        st.markdown("---")

        st.success(
            f"Predicted House Price: ${prediction * 100000:,.2f}"
        )

# ===================================================
# VISUALIZATION PAGE
# ===================================================

elif page == "Visualizations":

    st.header("Data Visualizations")

    tab1, tab2, tab3 = st.tabs([
        "Distribution",
        "Correlation",
        "Scatter Plot"
    ])

    # DISTRIBUTION

    with tab1:

        feature = st.selectbox(
            "Select Feature",
            X.columns
        )

        fig1, ax1 = plt.subplots(figsize=(10, 5))

        sns.histplot(
            X[feature],
            kde=True,
            ax=ax1,
            color="skyblue"
        )

        plt.title(feature)

        st.pyplot(fig1)

    # CORRELATION

    with tab2:

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        correlation = X.corr()

        sns.heatmap(
            correlation,
            cmap="Blues",
            ax=ax2
        )

        st.pyplot(fig2)

    # SCATTER PLOT

    with tab3:

        x_feature = st.selectbox(
            "Select X Axis",
            X.columns,
            key="x"
        )

        y_feature = st.selectbox(
            "Select Y Axis",
            X.columns,
            key="y"
        )

        fig3, ax3 = plt.subplots(figsize=(8, 5))

        ax3.scatter(
            X[x_feature],
            X[y_feature]
        )

        plt.xlabel(x_feature)
        plt.ylabel(y_feature)

        st.pyplot(fig3)

# ===================================================
# MODEL PERFORMANCE
# ===================================================

elif page == "Model Performance":

    st.header("Model Performance")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MSE", f"{mse:.2f}")

    with col2:
        st.metric("MAE", f"{mae:.2f}")

    with col3:
        st.metric("R2 Score", f"{r2:.2f}")

    st.subheader("Actual vs Predicted")

    fig4, ax4 = plt.subplots(figsize=(8, 5))

    ax4.scatter(y_test, y_pred)

    ax4.set_xlabel("Actual")
    ax4.set_ylabel("Predicted")

    st.pyplot(fig4)

# ===================================================
# DATASET PAGE
# ===================================================

elif page == "Dataset":

    st.header("Dataset Overview")

    st.subheader("Dataset Preview")

    st.dataframe(X.head())

    st.subheader("Dataset Shape")

    st.write(f"Rows: {X.shape[0]}")
    st.write(f"Columns: {X.shape[1]}")

    st.subheader("Statistical Summary")

    st.dataframe(X.describe())

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <center>
        <h4>
            Developed using Streamlit & KNN Regression
        </h4>
    </center>
    """,
    unsafe_allow_html=True
)
