from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("models/price_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            features = [
                float(request.form["longitude"]),
                float(request.form["latitude"]),
                float(request.form["housing_median_age"]),
                float(request.form["total_rooms"]),
                float(request.form["total_bedrooms"]),
                float(request.form["population"]),
                float(request.form["households"]),
                float(request.form["median_income"])
            ]
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
# To run the app, use the command: python app.py
# Make sure to have Flask installed in your environment.