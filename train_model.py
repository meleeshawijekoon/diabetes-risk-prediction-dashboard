import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("diabetes.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 2. CHECK DATA
# ==========================================

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 3. REPLACE INVALID ZERO VALUES
# ==========================================

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_with_zero:
    df[column] = df[column].replace(0, np.nan)

print("\nMissing values after replacing invalid zeros:")
print(df.isnull().sum())


# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 6. CREATE MACHINE LEARNING PIPELINE
# ==========================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ==========================================
# 8. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ==========================================
# 9. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy:", round(accuracy, 3))
print("ROC-AUC:", round(auc, 3))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    y_pred
))

print("\n==============================")
print("PROJECT COMPLETED")
print("==============================")