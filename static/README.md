# ❤️ Smart Healthcare Assistant — AI Disease Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-2.0-lightgrey)
![ML](https://img.shields.io/badge/ML-RandomForest-green)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen)

A machine learning web application that predicts the likelihood of heart disease based on patient clinical data. Built with Python, Flask, and Scikit-Learn.

---

## 🎯 Project Overview

This project implements an **AI-powered diagnostic assistant** that helps predict heart disease risk using patient data. The model is trained on the UCI Heart Disease dataset containing **1025 patient records** and achieves **99% accuracy**.

---

## 🚀 Features

- ✅ Real-time heart disease prediction
- ✅ 99% model accuracy (Random Forest Classifier)
- ✅ Clean and responsive web interface
- ✅ Model performance charts (Confusion Matrix, ROC Curve, Feature Importance)
- ✅ Input validation and error handling
- ✅ Trained on real clinical patient data

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| Web Framework | Flask |
| ML Library | Scikit-Learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, CSS |

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 98.54% |
| XGBoost | 98.54% |
| **Best Model** | **Random Forest (99%)** |

---

## 📁 Project Structure

```
Smart-Healthcare-Assistant/
├── templates/
│   ├── index.html          # Main prediction page
│   └── charts.html         # Model performance page
├── static/
│   ├── distribution.png    # Dataset distribution chart
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curve.png
├── app.py                  # Flask web application
├── model.py                # ML model training
├── charts.py               # Chart generation
├── explore.py              # Data exploration
├── heart.csv               # Dataset
├── model.pkl               # Trained model
└── README.md
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/MettuSurendraReddy/Smart-Healthcare-Assistant.git
cd Smart-Healthcare-Assistant
```

**2. Install dependencies**
```bash
pip install flask pandas scikit-learn xgboost matplotlib seaborn numpy
```

**3. Run the application**
```bash
python app.py
```

**4. Open in browser**
```
http://127.0.0.1:5000
```

---

## 🔍 Input Features

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Sex | Gender (Male/Female) |
| Chest Pain Type | Type of chest pain (0-3) |
| Resting Blood Pressure | In mm Hg |
| Cholesterol | Serum cholesterol in mg/dl |
| Fasting Blood Sugar | > 120 mg/dl (Yes/No) |
| Resting ECG | Electrocardiographic results |
| Max Heart Rate | Maximum heart rate achieved |
| Exercise Induced Angina | Yes/No |
| ST Depression | Oldpeak value |
| Slope | Slope of peak exercise ST segment |
| Major Vessels | Number of major vessels (0-3) |
| Thalassemia | Thalassemia type |

---

## 📸 Screenshots

### Prediction Page
![Prediction Page](static/distribution.png)

### Model Performance
![Charts Page](static/confusion_matrix.png)

---

## 📄 Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease Dataset
- **Records:** 1025 patients
- **Features:** 13 clinical features + 1 target variable
- **Missing Values:** None

---

## 👨‍💻 Author

**Mettu Surendra Reddy**
- 🎓 MSc Artificial Intelligence — Brandenburg University of Technology (BTU), Cottbus
- 💼 [LinkedIn](https://www.linkedin.com/in/-surendrareddy)
- 🐙 [GitHub](https://github.com/MettuSurendraReddy)
- 📧 surendrareddy.mettu25@gmail.com

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).