__author__ = 'Ziyan Zhu, Haohan Wang'

from sklearn.metrics import auc,roc_curve
from sklearn.preprocessing import scale
from sklearn import linear_model
import numpy as np
import math
from scipy.stats import norm


class HDILasso:
    def __init__(self, standardize=True, robust=True):
        self.standardize = standardize
        self.robust = robust

    def fit(self, x, y):
        n = x.shape[0]
        p = x.shape[1]

        if self.standardize:
            sds = np.std(x, axis=0, ddof=1)
        else:
            sds = np.ones(p)

        sds = sds.reshape((p, 1))

        # center the columns to get rid of the intercept#
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0, ddof=1))
        y = scale(y, axis=0, with_mean=True, with_std=False)
        ######################################
        ## Calculate Z using nodewise lasso ##
        ######################################

        ##### Z is the residuals of the nodewise regressions under 10-folds cv #####

        print("Calculating Z ...")
        Z = self.nodewiselasso(x)

        ###################################
        ## Projection estimator and bias ##
        ###################################
        ## Bias estimate based on scaled lasso. Y Versus X
        if p > (10 ** 6):  # universal lambda used in big data
            lam0 = math.sqrt(2 * math.log(p) / n)
        else:  # quantile lambda
            L = 0.1
            Lold = 0
            while (abs(L - Lold) > 0.001):
                k = (math.pow(L, 4) + 2 * math.pow(L, 2))
                Lold = L
                L = -norm.ppf(min(k / p, 0.99))
                L = (L + Lold) / 2
            if (p == 1):
                L = 0.5
            lam0 = math.sqrt(2 / n) * L

        sigmaint = 0.1
        sigmanew = 5
        flag = 0
        while abs(sigmaint - sigmanew) > 0.0001 and flag <= 100:
            flag = flag + 1
            sigmaint = sigmanew
            lam = lam0 * sigmaint
            s = lam * n
            objlasso = linear_model.LassoLars(alpha=s, fit_path=False, fit_intercept=False, normalize=False,precompute=False)
            objlasso.fit(x, np.ravel(y, order='C'))
            hy = objlasso.predict(x)
            sigmanew = np.sqrt(np.mean(np.square(y - hy)))

        hsigma = sigmanew
        hbeta = objlasso.coef_
        hbeta = hbeta.reshape((1, p))

        # Projection estimator and bias
        bproj = np.inner(Z.T, y.T) / n

        print("Calculating bias ... ")
        bias = np.zeros(p)
        for j in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, j, axis=1)
            betatemp = np.copy(hbeta)
            betalasso = np.delete(betatemp, j)
            gear = np.dot(Z[:, j:j + 1].T, temp)
            bias[j] = np.inner(gear, betalasso) / n
        bias = bias.reshape((p, 1))

        #### Subtract bias ####
        bproj = bproj - bias
        ###################################
        ######## calculate p-values #######
        ###################################
        if self.robust == True:
            ## an implementation of the calculation of the robust standard error
            ## based on the sandwich variance estimator from
            ## http://arxiv.org/abs/1503.06426

            # check if normalization is fullfiled
            Zx = np.multiply(Z, x)
            if ~np.allclose(np.ones(p), np.sum(Zx / n, axis=0)):
                dz = np.dot(Z.T, x)
                dz = np.diag(dz)
                scaleZ = dz / n
                Z = np.true_divide(Z, scaleZ)

            if len(hbeta) > p:
                one = np.ones(n)
                x = np.concatenate(one, x, axis=1)
            prediction = np.dot(hbeta, x.T)
            eps_tmp = y.T - prediction
            ## force esp_tmp to have mean 0 as if we fit with intercept
            eps_tmp = eps_tmp - np.mean(eps_tmp)

            save = np.empty(shape=(n, p))
            for j in range(p):
                for i in range(n):
                    save[i, j] = eps_tmp[:, i] * Z[i, j]
            stats = np.dot(eps_tmp, Z) / n
            sweep = save - stats
            sigmahatZ = np.sqrt(np.sum(sweep ** 2, axis=0)) / n
            scaleb = 1 / sigmahatZ
        else:
            scaleb = n / hsigma * np.sqrt(np.sum(Z ** 2, axis=0))

        se = 1 / scaleb

        bprojrescaled = np.zeros(p)
        for i in range(p):
            bprojrescaled[i] = bproj[i] * scaleb[i]
        pval = 2 * norm.sf(np.abs(bprojrescaled))
        bhat = bproj / sds

        return pval, bhat

    def getlambdasequence(self, x):
        p=x.shape[1]
        lams=np.empty(shape=(1,1))
        lambdas=np.empty(100)
        seq=np.linspace(0,100,100)
        for c in range(p):
            temp=np.copy(x)
            temp=np.delete(x,c,axis=1)
            lam,_,_=linear_model.lasso_path(X=temp,y=np.ravel(x[:,c]),n_alphas=100)
            lams=np.append(lams,lam)
        lams=np.delete(lams,0)
        for i in range(100):
            lambdas[i]=np.percentile(lams,seq[i])
        lambdas=-np.sort(-lambdas)

        return lambdas

    def totalerr_unit(self, c,K,dataselects,x,lambdas):
        n = x.shape[0]
        p = x.shape[1]
        l=lambdas.shape[0]
        totalerr=np.empty(shape=(l,K))


        tempx = np.copy(x)
        tempx = np.delete(tempx, c, axis=1)

        for i in range(K):
            whichj = (dataselects==i+1)
            temp=np.copy(x)
            temp = np.delete(temp, c, axis=1)
            _,coef,_=linear_model.lasso_path(X=temp[~whichj,:],y=np.ravel(x[~whichj,c]),alphas=lambdas)
            predictions = np.dot(temp[whichj,:],coef)
            totalerr[:,i]=np.mean((x[whichj,c:c+1]-predictions)**2,axis=0)
        return totalerr

    def cv_bestlambda(self, lambdas,x,K=10):
        n=x.shape[0]
        p=x.shape[1]
        l=lambdas.shape[0]
        k=np.linspace(1,K,num=K)
        cv=np.tile(k,n)
        cv=cv[0:n]
        dataselects = np.random.choice(cv,size=n, replace=False, p=None)
        errmean = np.ones(shape=(l,1))
        for c in range(p):
            err_unit=self.totalerr_unit(c, K, dataselects, x, lambdas)
            err_unit_mean = np.mean(err_unit, axis=1).reshape(l,1)
            errmean=np.append(errmean,err_unit_mean,axis=1)

        errmean=np.delete(errmean,0,axis=1)
        errmean=np.mean(errmean,axis=1)
        pos_min=errmean==np.min(errmean)
        bestlam=lambdas[pos_min]
        return bestlam

    def nodewiselasso(self, x,lambdatuningfactor=1):
        lambdas=self.getlambdasequence(x)
        bestlambda=self.cv_bestlambda(lambdas,x)
        bestlambda=bestlambda*lambdatuningfactor
        n = x.shape[0]
        p = x.shape[1]
        Z = np.empty(shape=(n, p))
        for i in range(p):
            temp = np.copy(x)
            temp = np.delete(temp, i, axis=1)
            fit = linear_model.Lasso(alpha=bestlambda)
            fit.fit(X=temp, y=np.ravel(x[:, i], order='C'))
            prediction =fit.predict(X=temp)
            Z[:, i] = x[:, i] - prediction

        # rescale Z#
        dz = np.dot(Z.T, x)
        dz = np.diag(dz)
        scaleZ = dz / n
        Z = np.true_divide(Z, scaleZ)

        return Z

    ##########################################



if __name__ == '__main__':
    data=np.genfromtxt("/Users/alexazhu/Desktop/test simulation data/Data4_10full.csv", delimiter=',')

    data=np.array(data, dtype=float)
    data=np.delete(data,0,axis=0)
    x=data[:,2:]
    y=data[:,1:2]

    hdi = HDILasso()
    pvalues,bhat=hdi.fit(x=x,y=y)

    print("p-values=",pvalues)

    # First 44 parameters have nonzero coefficients
    realb=np.append(np.ones(44),np.zeros(2200-44))

    fpr,tpr,thresholds = roc_curve(realb,1-pvalues,pos_label=1)
    aucscore=auc(fpr,tpr)
    print("AUC=",aucscore)
