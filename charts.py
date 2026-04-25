import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle
import os

# Create folder for charts
os.makedirs('static', exist_ok=True)

# Load data and model
df = pd.read_csv('heart.csv')
df = df.drop_duplicates()
print(f"Unique records: {len(df)}")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)

# ── Chart 1: Target Distribution ──
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Heart Disease Distribution in Dataset')
plt.xticks([0, 1], ['No Disease', 'Heart Disease'])
plt.xlabel('')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.savefig('static/distribution.png')
plt.close()
print("Chart 1 saved ✓")

# ── Chart 2: Confusion Matrix ──
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['No Disease', 'Heart Disease'],
            yticklabels=['No Disease', 'Heart Disease'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()
print("Chart 2 saved ✓")

# ── Chart 3: Feature Importance ──
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=True)
plt.figure(figsize=(7, 5))
importance.plot(kind='barh', color='#e63946')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()
print("Chart 3 saved ✓")

# ── Chart 4: ROC Curve ──
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='#e63946', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('static/roc_curve.png')
plt.close()
print("Chart 4 saved ✓")

print("\nAll charts saved in static/ folder!")