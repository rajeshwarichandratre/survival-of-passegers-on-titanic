import pandas as pd
import numpy as np
from scipy.stats import hmean, pearsonr
from scipy import stats
import math

def pearsonscorrTest(final_df, keyslist):
    print("\n")
    # ~~~~~~~~~~ on all variables ~~~~~~~~~~~~~~~~~~~~~
    # get Pearson correlation coeff
    print (">>> Starting Pearson Correlation Hypothesis Test on all variables... <<<")

    r_list = []
    for key in keyslist:
        for key2 in keyslist:
            if key == key2:
                continue
            r = pearsonr(final_df[key].values.T, final_df[key2].values.T)[0]
            r_list.append([key + "_" + key2, r])

    # get p_values given t
    a = 0.05
    p_values = r_list
    N = len(final_df[keyslist].values)
    # print(N)
    for i, (k, r) in enumerate(r_list):
        df = N - 2
        t = (r * math.sqrt(df)) / math.sqrt(1 - r ** 2)
        p = 1 - stats.t.cdf(t, df=df)
        p_values[i][1] = p

    # test!
    hypothesis_tests = p_values
    hypothesis_AR = []
    for i, (k, p) in enumerate(p_values):
        if p <= a:
            hypothesis_AR.append("Reject")
        else:
            hypothesis_AR.append("Accept")

    print("H0: rho = 0 (uncorrelated)")
    print("HA: rho != 0 (correlated)")
    print("\n+++ Results of Test (pair, p-value, accept or reject) +++")
    for x in list(zip(hypothesis_tests, hypothesis_AR)):
        print(x)
    #print(np.array(hypothesis_tests))

    # ~~~~~~~~~~~~~~~ on target variable ~~~~~~~~~~~~~~~~
    print ("\n>>> Starting Pearson Correlation Hypothesis Test on target variable... <<<")
    # get Pearson correlation coeff
    r_list = []
    for key in keyslist:
        r = pearsonr(final_df[key].values.T, final_df['Survived'].values.T)[0]
        r_list.append([key + "_Survived", r])
    #print(np.array(r_list))

    # get p_values given t
    p_values = r_list
    for i, (k, r) in enumerate(r_list):
        if k == 'Survived_Survived':
            continue
        df = N - 2
        t = (r * math.sqrt(df)) / math.sqrt(1 - r ** 2)
        p = 1 - stats.t.cdf(t, df=df)
        p_values[i][1] = p

    #print(np.array(p_values))

    # hypothesis test
    hypothesis_tests = p_values
    hypothesis_AR = []
    for i, (k, p) in enumerate(p_values):
        if p <= a:
            hypothesis_AR.append("Reject")
        else:
            hypothesis_AR.append("Accept")

    print("H0: rho = 0 (uncorrelated)")
    print("HA: rho != 0 (correlated)")
    print("\n+++ Results of Test (pair, p-value, accept or reject) +++")
    for x in list(zip(hypothesis_tests, hypothesis_AR)):
        print(x)

    print("\n")