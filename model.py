import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

# ── 1. Load data ──
df = pd.read_csv('heart.csv')

# Remove duplicate rows
df = df.drop_duplicates()
print(f"Dataset size after removing duplicates: {len(df)}")

# ── 2. Split features and target ──
X = df.drop('target', axis=1)  # everything except target
y = df['target']                # only the target column

# ── 3. Split into training and testing sets ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── 4. Train Random Forest ──
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Accuracy: {rf_acc * 100:.2f}%")

# ── 5. Train XGBoost ──
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy:       {xgb_acc * 100:.2f}%")

# ── 6. Pick the best model ──
if rf_acc >= xgb_acc:
    best_model = rf_model
    print("\nBest model: Random Forest ✓")
else:
    best_model = xgb_model
    print("\nBest model: XGBoost ✓")

# ── 7. Show detailed report ──
best_pred = best_model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, best_pred))

# Probabilities for ROC-AUC
best_proba = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, best_proba)
print(f"\nROC-AUC Score: {roc_auc:.3f}")


from sklearn.model_selection import cross_val_score

# Cross validation - proves accuracy is real
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"\n=== 5-Fold Cross Validation ===")
print(f"Scores: {cv_scores.round(3)}")
print(f"Mean Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Standard Deviation: {cv_scores.std()*100:.2f}%")

# ── 8. Save the model ──
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("Model saved as model.pkl ✓")