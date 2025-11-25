import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------
# 1. Load Dataset
# ------------------------------------
print("Loading dataset...")
df = pd.read_csv("portscan_dataset.csv")   # change filename if needed

# ------------------------------------
# 2. Clean Data
# ------------------------------------
df = df.dropna()
df = df.drop_duplicates()

# ------------------------------------
# 3. Encode Label Column
# ------------------------------------
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# ------------------------------------
# 4. Split Features & Labels
# ------------------------------------
X = df.drop(["label"], axis=1)
y = df["label"]

# 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------------
# 5. Train Random Forest Model
# ------------------------------------
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ------------------------------------
# 6. Predict
# ------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------
# 7. Accuracy
# ------------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# ------------------------------------
# 8. Safe Classification Report
# (Fixes the error you got)
# ------------------------------------

unique_classes = np.unique(y_test)
unique_class_names = label_encoder.inverse_transform(unique_classes)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=unique_classes,
    target_names=unique_class_names
))

# ------------------------------------
# 9. Confusion Matrix
# ------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=unique_class_names,
    yticklabels=unique_class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
