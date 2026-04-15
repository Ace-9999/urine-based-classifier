import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("kidney_disease.csv")

# -------------------------------
# CLEAN DATA
# -------------------------------

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop ID column if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Convert everything to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
df = df.fillna(df.mode().iloc[0])

# -------------------------------
# CREATE DIET-BASED LABELS
# -------------------------------
def create_label(row):
    sg = row.get('sg', 1.02)     # specific gravity
    al = row.get('al', 0)        # albumin
    su = row.get('su', 0)        # sugar
    bgr = row.get('bgr', 100)    # blood glucose
    bu = row.get('bu', 30)       # blood urea
    sc = row.get('sc', 1.0)      # serum creatinine
    hemo = row.get('hemo', 15)   # hemoglobin

    # 1. High Blood Sugar (Diabetes risk)
    if su > 0 or bgr > 140:
        return 1

    # 2. Kidney Stress
    elif al >= 1 or sc > 1.2 or bu > 40:
        return 2

    # 3. Dehydration
    elif sg < 1.015:
        return 3

    # 4. Infection / Weak condition
    elif hemo < 12:
        return 4

    # 0. Normal
    else:
        return 0

df['target'] = df.apply(create_label, axis=1)

# -------------------------------
# SPLIT DATA
# -------------------------------
X = df.drop('target', axis=1)
y = df['target']

print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation Accuracy:", scores.mean())

# -------------------------------
# SAVE MODEL
# -------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")