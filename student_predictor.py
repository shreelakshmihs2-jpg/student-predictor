import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

np.random.seed(42)
n = 200

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
past_score = np.random.uniform(40, 100, n)

final_score = (study_hours * 5) + (attendance * 0.3) + (past_score * 0.4) + np.random.normal(0, 5, n)

data = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "past_score": past_score,
    "final_score": final_score
})

print(data.head())
X = data[["study_hours", "attendance", "past_score"]]
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print("Model Accuracy (R2 Score):", accuracy)
plt.scatter(y_test, predictions)
plt.xlabel("Actual Final Score")
plt.ylabel("Predicted Final Score")
plt.title("Actual vs Predicted Student Scores")
plt.show()
