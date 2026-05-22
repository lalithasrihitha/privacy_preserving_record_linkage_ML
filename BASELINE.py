
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

X = pd.read_csv("data/sample_data/binary_features.csv")
y = pd.read_csv("data/sample_data/labels.csv").iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
)
#PCA
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("After PCA:")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


#training model
model = XGBClassifier(use_label_encoder=False, eval_metric ="logloss")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:" ,accuracy)
print("Precesion:" ,precision)
print("Recall:" , recall)
print("Confusion Matrix:\n" ,cm)

