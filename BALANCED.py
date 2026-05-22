
#PPRL RECORD LINKAGE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score ,precision_score , recall_score
from xgboost import XGBClassifier

# loading the data
df = pd.read_csv("data/sample_data/data.txt", sep="|", header=None)

print("Data shape:",df.shape)

#separating raw feature columns and label columns
X_raw = df.iloc[:, 0:56]  #except last column
y =  df.iloc [:, 56]


print ("X_raw shape :",X_raw.shape)
print("y shape:", y.shape)

#binary features
binary_features = []

for i in range (0,56,2):
        col_a = X_raw.iloc[:, i]
        col_b = X_raw.iloc[:, i+1]
        match = (col_a == col_b)
        feature = match.astype(int)
        binary_features.append(feature)

X = pd.concat(binary_features, axis =1)
print("Binary features shape:",  X.shape)

#training and testing

X_train, X_test, y_train, y_test = train_test_split(
                               X,y, test_size=0.25,random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


import numpy as np
train_data = np.column_stack((X_train.to_numpy(),y_train.to_numpy()))

class_1 = train_data[y_train.to_numpy() == 1]
class_0  = train_data[y_train.to_numpy() == 0]

#printing class counts to see imbalance
print("Before Balancing- class1:" ,len(class_1))
print("beforr Balancing-class0:", len(class_0))


#downsampling the majority class:

np.random.seed(42)
class_1_downsampled = class_1[np.random.choice(len(class_1), len(class_0), replace = False)]


#combining and shuffling balanced training set

balanced_train = np.vstack((class_1_downsampled, class_0))

#downsampling the majority class:

np.random.seed(42)
class_1_downsampled = class_1[np.random.choice(len(class_1), len(class_0), replace = False)]


#combining and shuffling balanced training set

balanced_train = np.vstack((class_1_downsampled, class_0))
np.random.shuffle(balanced_train)

#spliting balanced data into X and Y

X_train_bal = balanced_train[:,:-1]
y_train_bal = balanced_train[:,-1].astype(int)



model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train_bal, y_train_bal)

print("After balancing - X_train_bal:" , X_train_bal.shape , "y_train_bal:", y_train_bal.shape)


print("model training done")

y_pred = model.predict(X_test)
accuracy = accuracy_score (y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)


print("Accuracy:" , accuracy)
print("Precision:", precision)
print("Recall:" , recall)

print("y value counts:\n", pd.Series(y).value_counts())



