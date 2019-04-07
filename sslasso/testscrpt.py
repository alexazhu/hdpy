from debiasedLasso_skLearn import debiasedLasso as dLasso
import numpy as np
from scipy import stats
from Simulate import simulation

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)

    return FPR,TPR

testsize = 20
FPR = np.zeros(testsize)
TPR = np.zeros(testsize)
notUniform = np.zeros(testsize)

i=0
while i < testsize:

    s = simulation()
    beta, X,Y = s.GetSample(type='normal',n=1000,p=600,s0=10,b=0.5)
    true = beta!=0
    debias=dLasso()
    result=debias.fit(X=X,y=Y)
    pval=debias.pvalues
    hat = pval<0.05
    FPR[i],TPR[i] = perf_measure(true,hat)
    _, p_uni = stats.kstest(pval, 'uniform')
    notUniform[i] = p_uni<0.05
    i = i + 1
np.savetxt('20normal_s_30.csv', np.column_stack((FPR, TPR, notUniform)), delimiter=',')

i=0
while i < testsize:

    s = simulation()
    beta, X,Y = s.GetSample(type='circulant',n=1000,p=600,s0=10,b=0.5)
    true = beta!=0
    debias=dLasso()
    result=debias.fit(X=X,y=Y)
    pval=debias.pvalues
    hat = pval<0.05
    FPR[i],TPR[i] = perf_measure(true,hat)
    _, p_uni = stats.kstest(pval, 'uniform')
    notUniform[i] = p_uni<0.05
    i = i + 1
np.savetxt('20circulant_s_30.csv', np.column_stack((FPR, TPR, notUniform)), delimiter=',')



i = 0
while i < testsize:
    s = simulation()
    beta, X,Y = s.GetSample(type='Toeplitz',n=1000,p=600,s0=10,b=0.5)
    true = beta!=0
    print('start')
    debias=dLasso()
    result=debias.fit(X=X,y=Y)
    pval=debias.pvalues
    hat = pval<0.05
    FPR[i],TPR[i] = perf_measure(true,hat)
    _, p_uni = stats.kstest(pval, 'uniform')
    notUniform[i] = p_uni<0.05
    i = i + 1
np.savetxt('20Toeplitz.csv', np.column_stack((FPR, TPR, notUniform)), delimiter=',')

print('start')
i=0
while i < testsize:

    s = simulation()
    beta, X,Y = s.GetSample(type='Exp_decay',n=1000,p=600,s0=10,b=0.5)
    true = beta!=0
    debias=dLasso()
    result=debias.fit(X=X,y=Y)
    pval=debias.pvalues
    hat = pval<0.05
    FPR[i],TPR[i] = perf_measure(true,hat)
    _, p_uni = stats.kstest(pval, 'uniform')
    notUniform[i] = p_uni<0.05
    i = i + 1
np.savetxt('20Exp_decay.csv', np.column_stack((FPR, TPR, notUniform)), delimiter=',')

