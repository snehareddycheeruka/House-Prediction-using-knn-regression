import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(layout="wide")

st.title("💳 Loan Prediction Dashboard")

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload dataset")
    st.stop()

# ---------------------------
# LOAD DATA (CACHED)
# ---------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

# ---------------------------
# PREPROCESSING (CACHED)
# ---------------------------
@st.cache_data
def preprocess(df):

    df = df.dropna()

    # Fix DATE issue
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df.drop("date", axis=1, inplace=True)

    # Encode
    df = pd.get_dummies(df, drop_first=True)

    return df

df = preprocess(df)

# ---------------------------
# SPLIT (CACHED)
# ---------------------------
@st.cache_data
def split_data(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = split_data(df)

# ---------------------------
# SIDEBAR
# ---------------------------
page = st.sidebar.radio("Navigation", [
    "⚙️ Train Model",
    "🔮 Predict",
    "📈 Insights"
])

# ---------------------------
# TRAIN MODEL
# ---------------------------
if page == "⚙️ Train Model":

    model_name = st.selectbox("Choose Model", ["KNN", "Random Forest","Decision Tree"])

    if model_name == "KNN":
        k = st.slider("K Value", 1, 20, 5)

    elif model_name == "Decision Tree":
        n = st.slider("Number of Trees", 10, 200, 100)

    else:
        n = st.slider("Number of Trees", 10, 200, 100)

    @st.cache_resource
    def train_model(model_name, param):
        if model_name == "KNN":
            model = KNeighborsRegressor(n_neighbors=param)
        elif model_name == "Decision Tree":
            model = RandomForestRegressor(n_estimators=param)
        else:
            model = RandomForestRegressor(n_estimators=param)

        model.fit(X_train, y_train)
        return model

    model = train_model(model_name, k if model_name=="KNN" else n if model_name=="Decision Tree" else n)

    # Save model
    st.session_state["model"] = model

    # Evaluate
    y_pred = model.predict(X_test)

    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):,.2f}")
    st.metric("MAE", f"{mean_squared_error(y_test, y_pred):,.2f}")
    st.metric("R2 Score", f"{r2_score(y_test, y_pred):.4f}")

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif page == "🔮 Predict":

    st.subheader("Enter Input Values")

    model = st.session_state.get("model")

    if model is None:
        st.warning("⚠️ Train model first!")
        st.stop()

    # FORM (no auto refresh)
    with st.form("prediction_form"):

        age = st.number_input("Age")
        income = st.number_input("Income")
        loan_amount = st.number_input("Loan Amount")
        credit_score = st.number_input("Credit Score")
        num_transactions = st.number_input("Transactions")
        annual_spend = st.number_input("Annual Spend")

        city = st.selectbox("City", ["Bangalore", "Hyderabad", "Delhi"])
        employment = st.selectbox("Employment", ["Salaried", "Self-employed", "Student"])
        loan_type = st.selectbox("Loan Type", ["Personal", "Home", "Auto"])

        submit = st.form_submit_button("Predict")

    if submit:

        input_dict = {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "num_transactions": num_transactions,
            "annual_spend": annual_spend,
            f"city_{city}": 1,
            f"employment_type_{employment}": 1,
            f"loan_type_{loan_type}": 1,
        }

        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Prediction: {prediction:,.2f}")

# ---------------------------
# INSIGHTS PAGE
# ---------------------------
elif page == "📈 Insights":

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    tab1, tab2 = st.tabs(["Actual vs Predicted", "Residuals"])

    with tab1:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val])

        st.pyplot(fig)

    with tab2:
        residuals = y_test - y_pred

        fig, ax = plt.subplots()
        ax.scatter(y_pred, residuals)
        ax.axhline(0)

        st.pyplot(fig)

    # Feature importance
    importance = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(feat_df.set_index("Feature"))