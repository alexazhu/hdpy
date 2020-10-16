import numpy as np
import os.path
import time
from sklearn.linear_model import Lasso,lasso_path
import seaborn as sns
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    # # Sensitivity,true positive rate
    TPR = TP / (TP + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)

    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    if TP == FP == 0:
        FDR = -1
    else:
        FDR = FP / (TP + FP)
    # Negative predictive value
    # NPV = TN / (TN + FN)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return FPR, TPR, FNR, ACC, FDR, TP, FP


scaleZ = np.loadtxt('/Users/alexazhu/Desktop/XY6_R/scaleZ.csv', delimiter=",")
Z = np.loadtxt('/Users/alexazhu/Desktop/XY6_R/Z.csv', delimiter=",")

savepath = '/Users/alexazhu/PycharmProjects/HDI/Data/Equi_corr/Equi_corr100_500_s3_b1_pvals/'
path = '/Users/alexazhu/PycharmProjects/HDI/Data/Equi_corr/Equi_corr100_500_s3_b1_XY/'
files = os.listdir(path)
coef = np.zeros(shape=(50, 500))
pval = np.zeros((50, 500))
count = 1

for file in files:
    data = np.loadtxt(path+file,delimiter=',')
    start = time.process_time()
    X = data[:,1:]
    Y = data[:,0:1]
    Y = Y.reshape((100,))
    est = lasso_path(X,Y)
    print(est.alphas_)

    #est = hd.hd("lassoproj",X=X,y=Y)
    #pvals_corr = est.pvals
    #print("I am here")

    #sns.set_style("whitegrid")
    #sns.distplot(pvals_corr, color="m")
    #plt.show()
    #hd.manhattanplot(None,pvals_corr)
    #lci = est.lci
    #print(lci)
    #uci = est.uci
    #print(lci)
    #print(uci)
    print("time used:",time.process_time()-start)
    #print(file)
    #infohead = savepath+file
    #np.savetxt(fname="pvals_multi"+file, X=pvals, delimiter=',')


