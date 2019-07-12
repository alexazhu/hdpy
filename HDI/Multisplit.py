import numpy as np
import math
from sklearn.linear_model import Lasso,LassoCV,LassoLars,RidgeCV
from statsmodels.api import OLS
import sys


class multi_split:
    def __init__(self,B=100,fraction=0.5,model_selector=LassoCV(cv=10,n_jobs=-1,fit_intercept=False,tol= 0.001),ci=True,ci_level=0.95,exact=True,repeat_max = 20):
        # B: Times of single split
        # fraction: the ratio of I1 to I2
        # ci: calculate c.i. simultaneously
        # ci_level: confidence level
        # repeat_max: maximum repeat times

        self.B = B
        self.ci = ci
        self.ci_level = ci_level
        self.fraction = fraction

        self.selector = model_selector
        self.exact = exact
        self.repeat_max = repeat_max
        self.gamma =np.arange(start=math.ceil(0.05 * B) / B, stop=1 - 1 / B, step=1 / B,dtype=float)

    def singlesplit(self,X,y): # single split

        n,p = X.shape

        pvals_v = np.ones(p)
        lci_v = np.array([-np.Inf]*p)
        uci_v = np.array([np.Inf]*p)
        coefs = np.zeros(p)

        tryagain = True
        count = 0


        while tryagain:

            split = np.random.randint(low=1,high=n,size=self.nleft)
            xleft = X.copy()[split,:]
            yleft = y.copy()[split]

            xright = X.copy()[~split,:]
            yright = y.copy()[~split]

            sel_model = self.selector
            eps = 0.001
            K = 100

            max_lambda = np.max(np.abs(np.sum(np.dot(xleft.T,yleft)) ))/n
            lambda_path = np.round(np.exp(np.linspace(math.log10(max_lambda), math.log10(max_lambda * eps), K)),decimals=100)
            sel_model.set_params(alphas=lambda_path)
            sel_model.fit(X=xleft,y=yleft)
            sel_nonzero = np.where(sel_model.coef_!=0)[0]
            p_sel = len(sel_nonzero)

            if count>5: # Adaptive lasso
                init = RidgeCV(fit_intercept=False,cv=10).fit(xleft,yleft)
                w= abs(init.coef_)
                sel_model = LassoLars(fit_intercept=False,normalize=False,fit_path=False)
                sel_model.fit(xleft*w,y=yleft)
                sel_nonzero = np.where(sel_model.coef_!=0)[0]
                p_sel = len(sel_nonzero)


            ## Classical situation:
            ## A model with intercept is used, hence p.sel + 1 < nrow(x.right),
            ## otherwise, p-values can *not* be calculated
            if p_sel==0:
                print("Empty model selected")
                tryagain = False

            if p_sel>0 and p_sel< (self.nright-1):

                #Fitting Sample II with selected features using simple linear regression
                lm = OLS(yright,xright[:,sel_nonzero]).fit()
                sel_pval = lm.pvalues
                coefs[sel_nonzero] = lm.params


                if len(sel_pval)!=p_sel:
                    print("The classical OLS didn't return the correct number of p-values for the provided submodel.")
                    sys.exit()
                if not(np.all(sel_pval>=0) and np.all(sel_pval<=1)):
                    print("The classical OLS returned p-values below 0 or above 1.")
                    sys.exit()

                # Multi-test adjustment with Bonferroni method:
                pvals_v[sel_nonzero] = np.minimum(sel_pval*p_sel,1) #renew p-values
                tryagain=False
                # Calculate confidence intervals
                if self.ci:
                    if not (all(abs(self.gamma * self.B % 1) <= pow(10, -5))):
                        print("Duality might be violated because of choice of gamma. Use steps of length 1 / B")
                    sel_ci = np.array(lm.conf_int(alpha=self.ci_level))
                    lci_v[sel_nonzero] = sel_ci[:,0]
                    uci_v[sel_nonzero] = sel_ci[:,1]
                    pvals_adjusted = np.minimum(pvals_v * p_sel, 1)
                    return pvals_adjusted,coefs, lci_v, uci_v

            if p_sel >= (self.nright-1):#rankX less than number of low
                tryagain=True
                count=count+1
                print("Too large model selected in a sample-split")


            if count>self.repeat_max:
                print("Exceed max repeat times,sample splits resulted in too large models.")
                sys.exit()

        pvals_adjusted = np.minimum(pvals_v*p_sel,1)

        return pvals_adjusted,coefs

    def fit(self,X,y):

        n,p = X.shape

        # split the sample into two parts of approximately same size
        self.nleft = math.floor(n*self.fraction)
        self.nright = n - self.nleft

        if not (self.nleft>=1 or self.nright>=1):
            print("Not enough samples for splitting")
            sys.exit()

        pvals = np.zeros((self.B,p))
        coefs = np.zeros((self.B,p))

        if self.ci:
            lci = np.zeros((self.B,p))
            uci = np.zeros((self.B,p))

            for b in range(self.B):
                pvals[b,:],coefs[b,:],lci[b,:],uci[b,:]= self.singlesplit(X,y)
        else:
            for b in range(self.B):
                pvals[b,:],coefs[b,:]= self.singlesplit(X,y)

        self.p_nonaggr = pvals
        ## Calculate final p-values ##

        ## Control for FWER
        if not 0.05 in self.gamma:
                print("0.05 is not in gamma range due to the choice of B, the results might be incorrect.")

        quant_gamma = np.minimum(np.quantile(pvals,self.gamma,axis=0)
                                 /self.gamma.reshape(self.gamma.shape[0],1),1)
        inf_quant = np.min(quant_gamma, axis=0)

        if len(self.gamma)>1:
            penalty = 1-np.log10(np.min(self.gamma))
        else:
            penalty = 1

        pvals_pre = inf_quant*penalty
        pvals_current = np.minimum(pvals_pre, 1)

        self.d = p
        self.gamma_min = self.gamma[np.where(quant_gamma==inf_quant)[0]]
        self.pvals_corr = pvals_current
