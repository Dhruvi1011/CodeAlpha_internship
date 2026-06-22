import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)
# LOAD DATASET

df = pd.read_csv(r"D:\Internships_2ndYear\CodeAlpha_ML\german_credit_cleaned.csv")

print("Dataset Shape:", df.shape)
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# ENCODE CATEGORICAL COLUMNS

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# FEATURE ENGINEERING

if 'Income' in df.columns and 'Debt' in df.columns:
    df['Debt_Income_Ratio'] = df['Debt'] / (df['Income'] + 1)

# FEATURES AND TARGET

target_column = "target" 
df["loan_per_month"] = df["loan_amt"] / df["duration"]

X = df.drop(target_column, axis=1)
y = df[target_column]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# RANDOM FOREST MODEL

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# PREDICTIONS

y_pred = rf_model.predict(X_test)

# ACCURACY

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# CLASSIFICATION REPORT


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.title("Confusion Matrix - Random Forest")
plt.show()

# ROC-AUC SCORE

if len(y.unique()) == 2:
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nROC-AUC Score:", round(roc_auc, 4))

    RocCurveDisplay.from_estimator(
        rf_model,
        X_test,
        y_test
    )

    plt.title("ROC Curve - Random Forest")
    plt.show()

# FEATURE IMPORTANCE

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# MODEL SAVING

import joblib

joblib.dump(rf_model, "credit_scoring_model.pkl")

print("\nModel saved successfully!")