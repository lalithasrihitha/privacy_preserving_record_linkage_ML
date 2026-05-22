  GNU nano 2.9.8                                               mcnemar_sampling.py

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# Loading test labels
y_test = pd.read_csv("y_test_original.csv").values.ravel()

#  KNN
knn_orig = pd.read_csv("knn_pred.csv").values.ravel()
knn_up = pd.read_csv("knn_upsample_pred.csv").values.ravel()
knn_down = pd.read_csv("knn_downsample_pred.csv").values.ravel()

#  XGBoost
xgb_orig = pd.read_csv("xgb_pred.csv").values.ravel()
xgb_up = pd.read_csv("xgb_upsample_pred.csv").values.ravel()
xgb_down = pd.read_csv("xgb_downsample_pred.csv").values.ravel()

# SVM
svm_orig = pd.read_csv("svm_pred.csv").values.ravel()
svm_up = pd.read_csv("svm_upsample_pred.csv").values.ravel()
svm_down = pd.read_csv("svm_downsample_pred.csv").values.ravel()

#  Logistic Regression
log_orig = pd.read_csv("logreg_pred.csv").values.ravel()
log_up = pd.read_csv("logreg_upsample_pred.csv").values.ravel()
log_down = pd.read_csv("logreg_downsample_pred.csv").values.ravel()

# SLNN
slnn_orig = pd.read_csv("slnn_original_pred.csv").values.ravel()
slnn_up = pd.read_csv("slnn_upsample_pred.csv").values.ravel()
slnn_down = pd.read_csv("slnn_downsample_pred.csv").values.ravel()

#MLNN
#MLNN
mlnn_orig = pd.read_csv("mlnn_original_pred.csv").values.ravel()
mlnn_up = pd.read_csv("mlnn_upsample_pred.csv").values.ravel()
mlnn_down = pd.read_csv("mlnn_downsample_pred.csv").values.ravel()


def run_mcnemar(y_true, pred1, pred2, name1, name2):
    b, c = 0, 0

    for i in range(len(y_true)):
        m1 = pred1[i] == y_true[i]
        m2 = pred2[i] == y_true[i]

        if m1 and not m2:
            b += 1
        elif not m1 and m2:
            c += 1

    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)

    print("\n-")
    print(name1, "vs", name2)
    print("b:", b, " c:", c)
    print("p-value:", result.pvalue)

    if result.pvalue < 0.05:
        print("Significant difference")
    else:
	print("No significant difference")
  GNU nano 2.9.8                                               mcnemar_sampling.py

# COMPARISONS

# KNN
run_mcnemar(y_test, knn_orig, knn_up, "KNN Original", "KNN Upsample")
run_mcnemar(y_test, knn_orig, knn_down, "KNN Original", "KNN Downsample")

# XGBoost
run_mcnemar(y_test, xgb_orig, xgb_up, "XGB Original", "XGB Upsample")
run_mcnemar(y_test, xgb_orig, xgb_down, "XGB Original", "XGB Downsample")

# SVM
run_mcnemar(y_test, svm_orig, svm_up, "SVM Original", "SVM Upsample")
run_mcnemar(y_test, svm_orig, svm_down, "SVM Original", "SVM Downsample")

# LogReg
run_mcnemar(y_test, log_orig, log_up, "LogReg Original", "LogReg Upsample")
run_mcnemar(y_test, log_orig, log_down, "LogReg Original", "LogReg Downsample")

#SLNN
run_mcnemar(y_test, slnn_orig, slnn_up, "SLNN Original", "SLNN Upsample")
run_mcnemar(y_test, slnn_orig, slnn_down, "SLNN Original", "SLNN Downsample")

# MLNN
run_mcnemar(y_test, mlnn_orig, mlnn_up, "MLNN Original", "MLNN Upsample")
run_mcnemar(y_test, mlnn_orig, mlnn_down, "MLNN Original", "MLNN Downsample")



