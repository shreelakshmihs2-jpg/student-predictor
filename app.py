import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

st.set_page_config(page_title="Student Performance Predictor", page_icon="📊")

st.title("📊 Student Performance Predictor")
st.write("Predict a student's final score based on study hours, attendance, and past scores.")

# Generate the same sample dataset (or load your CSV here if you have one)
np.random.seed(42)
data = pd.DataFrame({
    "study_hours": np.random.uniform(1, 10, 200),
    "attendance": np.random.uniform(50, 100, 200),
    "past_score": np.random.uniform(40, 100, 200),
})
data["final_score"] = (
    data["study_hours"] * 5 +
    data["attendance"] * 0.3 +
    data["past_score"] * 0.4 +
    np.random.normal(0, 5, 200)
)

X = data[["study_hours", "attendance", "past_score"]]
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

st.subheader("Model Performance")
st.metric("R² Accuracy Score", f"{accuracy:.3f}")

st.divider()

st.subheader("Try a Prediction")
col1, col2, col3 = st.columns(3)
with col1:
    study_hours = st.slider("Study Hours (per day)", 0.0, 12.0, 5.0)
with col2:
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
with col3:
    past_score = st.slider("Past Score", 0.0, 100.0, 70.0)

if st.button("Predict Final Score"):
    input_data = pd.DataFrame([[study_hours, attendance, past_score]],
                                columns=["study_hours", "attendance", "past_score"])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Final Score: **{prediction:.2f}**")

st.divider()
st.subheader("Actual vs Predicted (Test Data)")
chart_data = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})
st.scatter_chart(chart_data)