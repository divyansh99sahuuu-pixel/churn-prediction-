from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("random.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    plan_type = int(request.form["plan_type"])
    monthly_fee = float(request.form["monthly_fee"])
    avg_weekly_usage_hours = float(request.form["avg_weekly_usage_hours"])
    support_tickets = int(request.form["support_tickets"])
    payment_failures = int(request.form["payment_failures"])
    tenure_months = int(request.form["tenure_months"])
    last_login_days_ago = int(request.form["last_login_days_ago"])

    data = np.array([[plan_type,
                      monthly_fee,
                      avg_weekly_usage_hours,
                      support_tickets,
                      payment_failures,
                      tenure_months,
                      last_login_days_ago]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "⚠ Customer is likely to Churn"
    else:
        result = "✅ Customer is likely to Stay"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)