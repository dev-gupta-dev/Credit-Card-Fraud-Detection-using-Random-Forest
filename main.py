import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset
data = {
    "Transaction_Amount": [200, 1500, 50, 5000, 300, 7000, 80, 600, 12000, 100],
    "Transaction_Time":   [5, 12, 8, 2, 16, 1, 9, 18, 3, 10],
    "Location_Match":     [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],   # 1 = location matches user’s profile
    "Fraud":              [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]    # Target variable
}
df = pd.DataFrame(data)
print("📂 Credit Card Dataset:\n", df)

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🌀 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example: Amount=4000, Time=2AM, Location mismatch
new_transaction = np.array([[4000, 2, 0]])
prediction = model.predict(new_transaction)
probability = model.predict_proba(new_transaction)

print("\n🔍 Prediction:", "Fraud 🚨" if prediction[0] == 1 else "Legit ✅")
print("📌 Fraud Probability:", probability[0][1])
