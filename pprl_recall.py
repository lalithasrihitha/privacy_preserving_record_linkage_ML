import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

X = pd.read_csv("data/sample_data/binary_features.csv")
y = pd.read_csv("data/sample_data/labels.csv").iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42

)
# handling class imbalance
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale = neg / pos

print("scale_pos_weight:", scale)

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05
)

model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >=0.25).astype(int)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test , y_pred)
recall = recall_score(y_test , y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", cm)

print("\nFeature Importance:")

importance = model.feature_importances_

for i , score in enumerate(importance):
        print(f"Feature {i}: {score:.4f}")

