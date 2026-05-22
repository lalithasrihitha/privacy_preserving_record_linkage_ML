
import pandas as pd
import numpy as np


# loading the data
df = pd.read_csv("data/sample_data/data.txt", sep="|", header=None)


#replacing  missing values with NaN
df.replace(to_replace=r'XXX.*', value=np.nan, regex=True, inplace=True)
print("Shape after replacing XXX with NaN:", df.shape)
print(df.isna().sum())

#features and labels
X_raw = df.iloc[:,0:56]
y = df.iloc[:, 56]
binary_features =[]
for i in range(0, 56, 2):
         col_a = X_raw.iloc[:, i]
         col_b = X_raw.loc[:,i+1]
         match = (col_a == col_b) & ~(col_a.isna() & col_b.isna())

        #checking NaN agreement
         nan_agree = col_a.isna() & col_b.isna()
         print("NaN pairs in this column:", nan_agree.sum())
         print("Total matches in this column:" , match.sum())
         feature = match.astype(int)
         binary_features.append(feature)
X = pd.concat(binary_features , axis =1)
X.to_csv("binary_features.csv" , index = False)
y.to_csv("labels.csv" , index = False)
print("Files saved successfully")



