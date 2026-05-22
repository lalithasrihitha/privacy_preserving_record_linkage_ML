  GNU nano 2.9.8                                            pprl_xgboosttuned.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score , recall_score , confusion_matrix, f1_score
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve , auc, precision_recall_curve
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

#Hyperparameter tuning
from sklearn.model_selection  import RandomizedSearchCV

#DEFINING BASE MODEL
xgb = XGBClassifier(use_label_encoder=False , eval_metric="logloss")

#Parameter grid
param_dist = {
	"n_estimators": [50,100,150],
        "max_depth": [3,5,7],
        "learning_rate": [0.01, 0.1 ,0.2],
        "subsample": [0.8 ,1.0],
        "colsample_bytree": [0.8, 1.0]
}

#Randomized search
random_search = RandomizedSearchCV(


#Randomized search
random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter =10 ,
        scoring="f1",
        cv=3,
	verbose=1,
        random_state=42,
        n_jobs=-1,
)

#TRAIN
random_search.fit(X_train, y_train)

#BEST MODEL
model = random_search.best_estimator_


print("Best Parameters:", random_search.best_params_)



y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test,y_pred)
TN, FP ,FN ,TP =cm.ravel()

print("Accuracy:" ,accuracy)
print("Precesion:" ,precision)
print("Recall:" , recall)
print("Confusion Matrix:\n" ,cm)
print("F1_Score:" ,f1)
print("true negatives(TN):" ,TN)
print("false positives(FP):" ,FP)
print("False Negatives(FN):" ,FN)
print("True Positives(TP):" ,TP)

#ROC Curve
fpr, tpr, _ =roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (AUC =%0.4f)' % roc_auc)
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc ="lower right")
plt.savefig("xgboosttuned_roc.png")
plt.show()

#Precision_Recall_Curve
precision_vals, recall_vals , _ = precision_recall_curve(y_test, y_probs)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.title('ROC Curve')
plt.legend(loc ="lower right")
plt.savefig("xgboosttuned_roc.png")
plt.show()

#Precision_Recall_Curve
precision_vals, recall_vals , _ = precision_recall_curve(y_test, y_probs)

plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()
plt.savefig("xgboosttuned_prc.png")

#standard deviation
import numpy as np
from sklearn.metrics import accuracy_score
#total test samples
n = len(y_test)
#accuracy
p = accuracy_score(y_test, y_pred)
#binomial standard deviation count
sd_count = np.sqrt(n * p * (1 - p))
#standard error
se_accuracy = np.sqrt(p * (1 - p) / n)

print("\n Binomial Statistics ")
print("Test size (n):", n)
print("Accuracy (p):", round(p, 6))
print("Binomial SD count:", round(sd_count, 4))
print("Standard Error (SE):", round(se_accuracy, 6))

import pandas as pd

# Save predictions
pd.DataFrame(y_pred).to_csv("xgb_tuned_pred.csv", index=False)

# Save test labels
pd.DataFrame(y_test).to_csv("y_test_original.csv", index=False)





