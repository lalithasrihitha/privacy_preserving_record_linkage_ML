  GNU nano 2.9.8                                                 mcnemar_test.py

import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import ttest_rel, wilcoxon

# Load true labels from the original test set
y_test = pd.read_csv("y_test_original.csv").values.ravel()

# Load predictions from models you want to compare
xgb_pred = pd.read_csv("xgb_pred.csv").values.ravel()
svm_pred = pd.read_csv("svm_pred.csv").values.ravel()
logreg_pred = pd.read_csv("logreg_pred.csv").values.ravel()
slnn_pred = pd.read_csv("slnn_original_pred.csv").values.ravel()
mlnn_pred = pd.read_csv("mlnn_original_pred.csv").values.ravel()
knn_pred = pd.read_csv("knn_pred.csv").values.ravel()


def run_mcnemar(y_true, pred1, pred2, name1, name2):
    # b = model 1 correct, model 2 wrong
    # c = model 1 wrong, model 2 correct
    b = 0
    c = 0

    for i in range(len(y_true)):
        model1_correct = (pred1[i] == y_true[i])
        model2_correct = (pred2[i] == y_true[i])

        if model1_correct and not model2_correct:
            b += 1
        elif not model1_correct and model2_correct:
            c += 1

  GNU nano 2.9.8                                                 mcnemar_test.py

    table = [[0, b],
             [c, 0]]

    result = mcnemar(table, exact=True)

    print("\n-")
    print("McNemar Test:", name1, "vs", name2)
    print("b (", name1, "correct,", name2, "wrong ):", b)
    print("c (", name1, "wrong,", name2, "correct ):", c)
    print("Test Statistic:", result.statistic)
    print("p-value:", result.pvalue)

    if result.pvalue < 0.05:
        print("Conclusion: Statistically significant difference")
    else:
	print("Conclusion: No statistically significant difference")


# Main comparisons
run_mcnemar(y_test, slnn_pred, xgb_pred, "SLNN", "XGBoost")
run_mcnemar(y_test, slnn_pred, svm_pred, "SLNN", "SVM")
run_mcnemar(y_test, slnn_pred, logreg_pred, "SLNN", "LogReg")
run_mcnemar(y_test, slnn_pred, mlnn_pred, "SLNN", "MLNN")
run_mcnemar(y_test, slnn_pred, knn_pred, "SLNN", "KNN")
run_mcnemar(y_test, svm_pred, logreg_pred, "SVM", "LogReg")
run_mcnemar(y_test, xgb_pred, svm_pred, "XGBoost", "SVM")
run_mcnemar(y_test, xgb_pred, logreg_pred, "XGBoost", "LogReg")
#t-test

from scipy.stats import ttest_rel
import numpy as np


  def run_paired_ttest(y_true, pred1, pred2, name1, name2):
      # Convert predictions to correctness (1/0)
      acc1 = (pred1 == y_true).astype(int)
      acc2 = (pred2 == y_true).astype(int)

      # Run paired t-test
      t_stat, p_value = ttest_rel(acc1, acc2)

      print("\n")
      print("Paired T-Test:", name1, "vs", name2)
      print("t-statistic:", t_stat)
      print("p-value:", p_value)

      if p_value < 0.05:
          print("Conclusion: Significant difference")
      else:
          print("Conclusion: No significant difference")


  # wilcoxon test

  def run_wilcoxon(y_true, pred1, pred2, name1, name2):
      acc1 = (pred1 == y_true).astype(int)
      acc2 = (pred2 == y_true).astype(int)

      diff = acc1 - acc2
      diff = diff[diff != 0]

      if len(diff) == 0:
          print("\nWilcoxon:", name1, "vs", name2, "→ No differences")
          return

    stat, p_value = wilcoxon(diff)

    print("\n--")
    print("Wilcoxon Test:", name1, "vs", name2)
    print("p-value:", p_value)

    if p_value < 0.05:
        print("Significant difference")
    else:
	print("No significant difference")


# Paired t-test and Wilcoxon calls

run_paired_ttest(y_test, slnn_pred, xgb_pred, "SLNN", "XGBoost")
run_wilcoxon(y_test, slnn_pred, xgb_pred, "SLNN", "XGBoost")

run_paired_ttest(y_test, slnn_pred, svm_pred, "SLNN", "SVM")
run_wilcoxon(y_test, slnn_pred, svm_pred, "SLNN", "SVM")

run_paired_ttest(y_test, slnn_pred, logreg_pred, "SLNN", "LogReg")
run_wilcoxon(y_test, slnn_pred, logreg_pred, "SLNN", "LogReg")

run_paired_ttest(y_test, slnn_pred, mlnn_pred, "SLNN", "MLNN")
run_wilcoxon(y_test, slnn_pred, mlnn_pred, "SLNN", "MLNN")

run_paired_ttest(y_test, slnn_pred, knn_pred, "SLNN", "KNN")
run_wilcoxon(y_test, slnn_pred, knn_pred, "SLNN", "KNN")

run_paired_ttest(y_test, svm_pred, logreg_pred, "SVM", "LogReg")
run_wilcoxon(y_test, svm_pred, logreg_pred, "SVM", "LogReg")

run_paired_ttest(y_test, xgb_pred, svm_pred, "XGBoost", "SVM")
run_wilcoxon(y_test, xgb_pred, svm_pred, "XGBoost", "SVM")

run_paired_ttest(y_test, xgb_pred, logreg_pred, "XGBoost", "LogReg")
run_wilcoxon(y_test, xgb_pred, logreg_pred, "XGBoost", "LogReg")
