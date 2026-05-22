  GNU nano 2.9.8                                             pprl_log_reg_upsample.py

import pandas as pd


X = pd.read_csv("data/sample_data/binary_features.csv")
y = pd.read_csv("data/sample_data/labels.csv")

#converting y to 1D array
y = y.values.ravel()


print("X shape:" , X.shape)
print("y shape:" , y.shape)
print("First 5 labels:", y[:5])
print("Unique label values:" , set(y))


#Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
     X,y,
     test_size=0.25 ,
     random_state=42
)

from sklearn.utils import resample
import pandas as pd

#combine X_train AND y_train
train_data = pd.DataFrame(X_train)
train_data['label'] = y_train
print("Before UpSampling:")
print(train_data['label'].value_counts())

#separate classes
majority = train_data[train_data['label'] == 1]
minority = train_data[train_data['label'] == 0]

#upsample minority
minority_upsampled = resample(minority,
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)
#combine
upsampled = pd.concat([majority, minority_upsampled])
upsampled = upsampled.sample(frac=1, random_state=42).reset_index(drop=True)
  GNU nano 2.9.8                                             pprl_log_reg_upsample.py

print("After upsampling:")
print(upsampled['label'].value_counts())

#split back
X_train = upsampled.drop('label' , axis =1).values
y_train = upsampled['label'].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Scaled train shape:" , X_train.shape)
print("Scaled test shape:" , X_test.shape)
print("First row after scaling:")
print(X_train[0])


#Building Logistic Regression using Sklearn

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200, random_state=42)
#Train model
model.fit(X_train , y_train)

#prediction
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
#Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Accuracy:" ,accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1_score:" ,f1)

print("Confusion  Matrix:\n", cm)

print("True Negatives (TN):" ,tn)
print("False Positives (FP):" ,fp)
print("False Negatives (FN):" ,fn)
print("True Positives (TP):" ,tp)


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label='LR ROC (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LR upsample')
plt.legend(loc="lower right")
plt.savefig("LR_upsample_roc.png")
plt.show()
# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_probs)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - LR upsample')
plt.savefig("LR_upsample_pr.png")
plt.show()


import numpy as np
from sklearn.metrics import accuracy_score

n = len(y_test)
p = accuracy_score(y_test, y_pred)

sd_count = np.sqrt(n * p * (1 - p))
se_accuracy = np.sqrt(p * (1 - p) / n)
ci_low = p - 1.96 * se_accuracy
ci_high = p + 1.96 * se_accuracy

print("\n--- Binomial Statistics ---")
print("Test size (n):", n)
print("Accuracy:", round(p, 6))
print("Binomial SD (count):", round(sd_count, 4))
print("Standard Error (SE):", round(se_accuracy, 6))
print("95% Confidence Interval:", round(ci_low, 6), "to", round(ci_high, 6))

import pandas as pd

pd.DataFrame(y_pred).to_csv("logreg_upsample_pred.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test_upsample.csv", index=False)

