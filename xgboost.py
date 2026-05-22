# PPRL RECORD LINKAGE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# loading the data
df = pd.read_csv("data/sample_data/data.txt", sep="|", header=None)


print("Data shape:", df.shape)

# separating raw feature columns and label columns
X_raw = df.iloc[:, 0:56]  # except last column
y = df.iloc[:, 56]

print("X_raw shape :", X_raw.shape)
print("y shape:", y.shape)

# binary features
binary_features = []

for i in range(0, 56, 2):
    col_a = X_raw.iloc[:, i]
    col_b = X_raw.iloc[:, i + 1]
    match = (col_a == col_b)
    feature = match.astype(int)
    binary_features.append(feature)

X = pd.concat(binary_features, axis=1)
print("Binary features shape:", X.shape)


#training and testing

X_train, X_test, y_train, y_test = train_test_split(
                               X,y, test_size=0.25,random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)
print("model training done")

y_pred = model.predict(X_test)
accuracy = accuracy_score (y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Accuracy:" , accuracy)

print("Precision:", precision)
print("Recall:" , recall)

print("y value counts:\n", pd.Series(y).value_counts())


#Standard deviation
import numpy as np
from sklearn.metrics import accuracy_score

#total test samples
n = len(y_test)

#accuracy
p = accuracy_score(y_test, y_pred)

#binomial standard deviation

sd_count = np.sqrt(n * p * (1-p))

#standard error (proportion)

se_accuracy = np.sqrt(p * (1-p) / n)

print("\n Binomial statistics for xgboost")
print("test size(n):" , n)
print("Accuracy(p):" , round(p,6))
print("Binomial SD count:", round(sd_count, 4))
print("standard error (SE):", round(se_accuracy, 6))



import pandas as pd

pd.DataFrame(y_test).to_csv("y_test.csv", index=False)
pd.DataFrame(y_pred).to_csv("xgb_pred.csv", index=False)

