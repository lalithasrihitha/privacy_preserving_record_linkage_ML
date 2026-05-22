  GNU nano 2.9.8                                                pprl_upsample.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score ,f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc , precision_recall_curve
import matplotlib.pyplot as plt

X = pd.read_csv("data/sample_data/binary_features.csv")
y = pd.read_csv("data/sample_data/labels.csv").iloc[:,0]



from sklearn.utils import resample

df = X.copy()
df["label"]= y

#separate classes
matches = df[df["label"] == 1]
non_matches = df[df["label"] == 0]
print("Before Matching:")
print("Matches:" , len(matches))
print("Non -matches :", len(non_matches))



#train test split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
)

# upsample only training data
train_data = X_train.copy()
train_data["label"] = y_train

matches = train_data[train_data["label"] == 1]
non_matches = train_data[train_data["label"] == 0]

print("Before upsampling (training only):")
print(train_data["label"].value_counts())

non_matches_upsampled = resample(
    non_matches,
    replace=True,
    n_samples=len(matches),
    random_state=42
)

upsampled_train = pd.concat([matches, non_matches_upsampled])
upsampled_train = upsampled_train.sample(frac=1, random_state=42).reset_index(drop=True)

print("After upsampling (training only):")
print(upsampled_train["label"].value_counts())

X_train = upsampled_train.drop("label", axis=1)
y_train = upsampled_train["label"]

#PCA
pca = PCA(n_components=10)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("After PCA:")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


print("X_train shape:" , X_train.shape)
print("y_train_distribution:\n" , pd.Series(y_train).value_counts())

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
print("Accuracy:" ,accuracy)
print("Precesion:" ,precision)
print("Recall:" , recall)
print("f1 score:" ,f1)
print("Confusion Matrix:\n" ,cm)

tn, fp, fn, tp = cm.ravel()

print("True Negatives (TN):" ,tn)
print("False positives (FP):" ,fp)
print("False Negatives (FN):" ,fn)
print("True Positives (TP):" ,tp)



# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='XGBoost Upsample ROC (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Upsample')
plt.legend(loc="lower right")
plt.savefig("xgb_upsample_roc.png")
plt.show()
# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - XGBoost Upsample')
plt.savefig("xgb_upsample_pr.png")
plt.show()


#Standard Deviation and Confidence Interval
import numpy as np
from sklearn.metrics import accuracy_score

# Get number of test samples
n = len(y_test)

# Calculating accuracy
p = accuracy_score(y_test, y_pred)

# Calculating Binomial Standard Deviation (count)
sd_count = np.sqrt(n * p * (1 - p))

# Calculating Standard Error (accuracy)
se_accuracy = np.sqrt(p * (1 - p) / n)

# Calculating 95% Confidence Interval
ci_low = p - 1.96 * se_accuracy
ci_high = p + 1.96 * se_accuracy

# Printing results
print("\n Binomial Statistics")
print("Test size (n):", n)
print("Accuracy:", round(p, 6))
print("Binomial SD (count):", round(sd_count, 4))
print("Standard Error (SE):", round(se_accuracy, 6))
print("95% Confidence Interval for Accuracy:", round(ci_low, 6), "to", round(ci_high, 6))


import pandas as pd

pd.DataFrame(y_pred).to_csv("xgb_upsample_pred.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test_upsample.csv", index=False)

