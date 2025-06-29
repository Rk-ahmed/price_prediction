import pandas as pd
url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)
df.to_csv("data/housing.csv", index=False)

# File: train_model.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Assume df already loaded
df.dropna(inplace=True)
df.drop("ocean_proximity", axis=1, inplace=True)
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train_scaled, y_train)
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("RMSE:", rmse)

import joblib
joblib.dump(model, "models/price_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")