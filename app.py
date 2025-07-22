from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
with open("model/customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
features = model_data["features_names"]

with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # 1. Capture form input as dict
        input_data = {key: request.form[key] for key in request.form}

        # 2. Convert to DataFrame
        df = pd.DataFrame([input_data])

        # 3. Convert numeric columns to proper types
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
        df["tenure"] = df["tenure"].astype(int)
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(float)
        df["TotalCharges"] = df["TotalCharges"].astype(float)

        # 4. Apply label encoders to categorical columns
        for column, encoder in encoders.items():
            if column in df.columns:
                df[column] = encoder.transform(df[column])
            else:
                return f"Missing expected column: {column}"

        # 5. Reorder columns exactly as training features
        df = df[features]

        # 6. Predict churn and probability
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        result = "Churn" if prediction == 1 else "No Churn"
        return render_template("result.html", prediction=result, probability=round(prob * 100, 2))

    except Exception as e:
        return f"❌ Error: {str(e)}"

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

