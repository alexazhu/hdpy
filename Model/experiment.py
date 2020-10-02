from Data_Simulate import simulation
import numpy as np
import pandas as pd
from LDPE import lassoproj
from Multisplit import multi_split
from debiasedLasso_skLearn import debiasedLasso
from RidgeProj import ridge_proj
from scipy.stats import kstest

class experiment:
    def __init__(self,testsize,saveup,n,p,s0,b,sigma,random_coefs=False):
        self.testsize = testsize
        self.saveup = saveup
        self.n = n
        self.p = p
        self.s0 = s0
        self.b = b
        self.sigma = sigma
        self.random_coefs = random_coefs

    def perf_measure(self,y_actual, y_hat):
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


        # # Sensitivity,true positive rate
        TPR = TP / (TP + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)

       # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        if TP==FP==0:
            FDR = -1
        else:
            FDR = FP / (TP + FP)
        # Negative predictive value
        #NPV = TN / (TN + FN)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)

        return FPR,TPR,FNR,ACC,FDR,TP,FP

    def exp(self,method='multisplit'):
        """
                -----------
                Parameters:
                -----------
                testsize : number of independent simulations

                s0 : Cardinality of the active set

                random_coef : [Default: False]
                             The nonzero coefficients were picked randomly from a uniform distribution.

                rand_up : [Optional] A value. The upper bound of the uniform distribution.

                rand_lw : [Optional] A value. The lower bound of the uniform distribution.

                b : [Optional] A fixed value used as the sizes of all nonzero coefficients.
                -----------
                Returns:
                -----------
                betas : array-like, shape(n_features, )
                """
        testsize = self.testsize
        n = self.n
        p = self.p

        FPR = []
        TPR = []
        FNR = []
        FDR = []
        ACC = []
        TP = []
        FP = []
        notUniform = []

        coef = np.zeros(shape=(testsize,p))
        pval = np.zeros((testsize,p))
        i=0
        while i < testsize:
            simu = simulation(n,p,snr=0.25,type=self.sigma,seed=i)
            beta,active = simu.get_coefs(self.s0,self.random_coefs,self.b)
            true = np.zeros(p)
            true[active]=1

            X,Y = simu.get_data(verbose=False)

            if self.saveup:
                Xname = './data/X'+str(i+1)+'.csv'
                Yname = './data/Y'+str(i+1)+'.csv'
                np.savetxt(fname=Xname, X=X, delimiter=',')
                np.savetxt(fname=Yname, X=Y, delimiter=',')

            if method == 'multisplit':
                model = multi_split(B=100,ci=False)
                model.fit(X, Y)
                pvals = model.pvals_corr
                hat = pvals<0.05
                #hat = np.ones(p)
                #hat[model.active_FDR]=1

            if method =='lasso_proj':
                model = lassoproj()
                model.fit(X,Y)
                pvals = model.pvals
                hat = pvals < 0.05

            if method == 'sslasso':
                model = debiasedLasso()
                model.fit(X,Y)
                pvals = model.pvals
                hat = model.bhat > 0

            if method == 'ridge_proj':
                model = ridge_proj()
                model.fit(X,Y)
                pvals = model.pvals
                hat = pvals < 0.05


            fpr, tpr, fnr, acc,fdr,tp,fp = self.perf_measure(true,hat)
            FPR.append(fpr)
            TPR.append(tpr)
            FNR.append(fnr)
            FDR.append(fdr)
            ACC.append(acc)
            TP.append(tp)
            FP.append(fp)

            _, p_uni = kstest(pvals[true == 0], 'uniform')
            notUniform.append(p_uni<0.05)

            pval[i,:] = pvals
            coef[i,:] = beta

            i = i + 1
            print(i)

        head = self.sigma + str(n)+'_'+ str(p)+'_'+str(self.s0)+'_b'+str(self.b)+'.csv'
        Coefname = './coef_'+method+head
        Pvalname = './pval'+method+head

        np.savetxt(fname=Coefname,X=coef, delimiter=',')
        np.savetxt(fname=Pvalname, X=pval, delimiter=',')

        info = pd.DataFrame({'FPR':FPR,'TPR':TPR,'FNR':FNR,'ACC':ACC,'FDR':FDR,'NotUniform':notUniform,'TP':TP,'FP':FP})
        infohead = './info_'+method+head
        info.to_csv(path_or_buf=infohead)

#e = experiment(testsize=50,saveup=True,n=100,p=500,s0=10,b=1,sigma='Equi_corr',random_coefs=False)
e = experiment(testsize=50,saveup=True,n=100,p=500,s0=10,b=1,sigma='Toeplitz',random_coefs=False)
#e = experiment(testsize=50,saveup=True,n=100,p=500,s0=10,b=1,sigma='Circulant',random_coefs=False)
#e = experiment(testsize=50,saveup=True,n=100,p=500,s0=10,b=1,sigma='Exp_decay',random_coefs=False)

#e.exp(method='ridge_proj')
#e.exp(method='multisplit')
#e.exp(method='sslasso')
e.exp(method='lasso_proj')

pvals = hd.hd(method="multi-split", X=X, y=Y, family="gaussian",
       ci=True, ci_level=0.95, alpha=None, intercept=False, B=100, fraction=0.5,
       model_selector=LassoCV(cv=10, n_jobs=-1, fit_intercept=False, tol=0.0001),
       exact=True, repeat_max=20, robust=True, standardize=True,
       ridge_unprojected=False)