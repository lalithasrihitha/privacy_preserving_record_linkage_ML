
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix, f1_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

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
#APPLYING SMOTE:
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE :")
print("X_train shape:" , X_train.shape)
print("y_train_distribution:\n" , pd.Series(y_train).value_counts())

#training model
#training model
model = XGBClassifier(use_label_encoder=False, eval_metric ="logloss")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Accuracy:" ,accuracy)
print("Precesion:" ,precision)
print("Recall:" , recall)
print("F1_score:" ,f1)
print("Confusion Matrix:\n" ,cm)

print("True Negatives (TN):", tn)
print("False Positives(FP):" , fp)
print("False Negatives (FN):" ,fn)
print("True Positives (TP):", tp)

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, label ='XGBoost ROC Curve (AUC =%0.4f)' % roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve-XGBoost')
plt.legend(loc = "lower right")
plt.show()
plt.savefig("xgboost_roc.png")

#Precision Recall Curve
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_probs)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve - XGBoost')
plt.show()
plt.savefig("xgboost_pr.png")

print("PR curve saved as xgboost_pr.png")


