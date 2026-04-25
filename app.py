from flask import Flask, render_template, request
import pickle
import numpy as np

# ── 1. Load the saved model ──
app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# ── 2. Home page ──
@app.route('/')
def home():
    return render_template('index.html')

# ── 3. Prediction page ──
@app.route('/predict', methods=['POST'])
def predict():
    features = [
        int(request.form['age']),
        int(request.form['sex']),
        int(request.form['cp']),
        int(request.form['trestbps']),
        int(request.form['chol']),
        int(request.form['fbs']),
        int(request.form['restecg']),
        int(request.form['thalach']),
        int(request.form['exang']),
        float(request.form['oldpeak']),
        int(request.form['slope']),
        int(request.form['ca']),
        int(request.form['thal'])
    ]

    # Make prediction
    prediction = model.predict([features])[0]

    # Result message
    if prediction == 1:
        result = "⚠️ Heart Disease Detected"
        color = "red"
    else:
        result = "✅ No Heart Disease Detected"
        color = "green"

    return render_template('index.html', result=result, color=color)

@app.route('/charts')
def charts():
    return render_template('charts.html')

# ── 4. Run the app ──
if __name__ == '__main__':
    app.run(debug=True)