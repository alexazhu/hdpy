import numpy as np
import math
from sklearn.linear_model import LassoCV,LassoLars,RidgeCV
from statsmodels.api import OLS
from scipy.stats import norm
from multiprocessing import Pool
import sys

class Multisplit:    
    def __init__(self, ci=False, ci_level=0.95,
                 aggr_ci=False, return_nonaggr=False,
                 B=100, fraction=0.5, repeat_max=20,
                 ):

        '''
        An implementation of Multiple Splitting methods https://doi.org/10.1198/jasa.2009.tm08647
        Following the R package hdi by Lukas Meier, Date:  2 Apr 2013, 11:52
        ==========================================================================================
        :param ci: calculate c.i. simultaneously
        :param ci_level: confidence level
        :param B: Times of single split
        :param repeat_max: maximum repeat times
        ==========================================================================================
        Attributes:
        p_nonaggr: Raw p-values that haven't been aggregated, shape of B*p
        pvals_corr: P-values aggregated controlling for family-wise error rate
        lci: Lower bounds of the confidence intervals
        uci: Upper bounds of the confidence intervals
        gamma_min: Smallest value of the gamma sequence
        ===========================================================================================
        Author: Ziyan Zhu, Date: Oct 4th,2019
        ==========================================================================================
        '''

        self.B = B
        self.ci = ci
        self.ci_level = ci_level
        self.fraction = fraction
        self.aggr_ci = aggr_ci
        self.return_nonaggr = return_nonaggr

        self.repeat_max = repeat_max
        self.gamma = np.arange(start=math.ceil(0.05 * B) / B, stop=1 - 1 / B, step=1 / B, dtype=float)

    def singlesplit(self, X, y, nleft):
        ######## Split a sample into two subsamples ########
        #print(self.gamma)
        n, p = X.shape
        nright = n - nleft

        pvals_v = np.ones(p)
        lci_v = np.array([-np.Inf] * p)
        uci_v = np.array([np.Inf] * p)
        coefs = np.zeros(p)
        ses_v = np.array([np.Inf] * p)
        df_res = 0

        tryagain = True
        count = 0
 
        while tryagain:

            ######## Randomly split the sample #######
            split = np.random.choice(np.arange(n), nleft, replace=False)  # without replacement
            xleft = X.copy()[split, :]
            yleft = y.copy()[split]

            xright = X.copy()[~split, :]
            yright = y.copy()[~split]
            

            ######## Model selection on Sample I #######
            if self.manual_lam:
                # calculate regularization path
                eps = 0.001
                K = 100
                max_lambda = np.max(np.abs(np.sum(np.dot(xleft.T, yleft)))) / n
                lambda_path = np.round(np.exp(np.linspace(math.log10(max_lambda), math.log10(max_lambda * eps), K)),
                                       decimals=100)
                self.selector.set_params(alphas=lambda_path,normalize =True)

            #print(lambda_path)
            self.selector.fit(X=xleft, y=yleft)

            sel_nonzero = (self.selector.coef_ != 0)  # location of selected variables

            #location = np.where(self.selector.coef_ != 0)
            #print(location)
            #print(self.selector.coef_)

            p_sel = sum(sel_nonzero)  # size of selected variables

            ######## Check up the selected results, make sure applicable for OLS ########

            if (p_sel + 1) >= nright:
                # rankX larger than number of row, cannot calculate p-values
                tryagain = True
                count = count + 1
                print("Too large model selected in a sample-split")

            if p_sel == 0:
                print("Empty model selected, it is OK")
                tryagain = False

            if p_sel > 0 and (p_sel + 1) < nright:

                tryagain = False
            
                ######## Fitting Sample II with reduced features using OLS ########

                lm = OLS(yright, xright[:, sel_nonzero]).fit(method="qr")

                df_res = lm.df_resid
                sel_pval = lm.pvalues

                coefs[sel_nonzero] = lm.params

                ses_v[sel_nonzero] = lm.bse

                # Sanity checks for p-values

                if len(sel_pval) != p_sel:
                    sys.exit(
                        "The statsmodels.OLS didn't return the correct number of p-values for the provided submodel.")
                if not (np.all(sel_pval >= 0) and np.all(sel_pval <= 1)):
                    sys.exit("The statsmodels.OLS returned p-values below 0 or above 1.")

                ######## Multiple testing adjustment on small sample: Bonferroni ########

                pvals_v[sel_nonzero] = np.minimum(sel_pval * p_sel, 1)  # renew p-values

                ######## Confidence intervals and other relative informations ########
                if all(pow(10, -5) < abs(self.gamma * self.B % 1)):
                    print("Duality might be violated because of choice of gamma. Use steps of length 1 / B")

                sel_ci = lm.conf_int(alpha=self.ci_level)

                lci_v[sel_nonzero] = sel_ci[:, 0]
                uci_v[sel_nonzero] = sel_ci[:, 1]

        
                

                ######## End of C.I. ########

            if count > self.repeat_max:
                print("Exceed max repeat times,sample splits resulted in too large models.")
                sys.exit()

            return pvals_v, p_sel, coefs, lci_v, uci_v, ses_v, df_res

    def fit(self, X, y，manual_lam=True,
                 model_selector=LassoCV(n_jobs=-1, fit_intercept=False, tol=1e-07,cv=10)):
        '''       
        ==========================================================================================
        :param fraction: the ratio of I1 to I2
        :param repeat_max: maximum repeat times
        :param model_selector: model for selection on the first half
        :param manul_lam: use manually calculated lambda sequences
        :param fraction: the ratio of I1 to I2
       
        ==========================================================================================
        Attributes:
        p_nonaggr: Raw p-values that haven't been aggregated, shape of B*p
        pvals_corr: P-values aggregated controlling for family-wise error rate
        lci: Lower bounds of the confidence intervals
        uci: Upper bounds of the confidence intervals
        gamma_min: Smallest value of the gamma sequence
        ===========================================================================================
        Author: Ziyan Zhu, Date: Oct 4th,2019
        ==========================================================================================
        '''

        n, p = X.shape

        ######## Split the sample into two parts of approximately same size ########
        nleft = math.floor(n * self.fraction)
        nright = n - nleft

        if not (nleft >= 1 or nright >= 1):
            sys.exit("Not enough samples for splitting")

        ######## Repeat sample splitting for B times ########
        pvals = np.zeros((self.B, p))
        coefs = np.zeros((self.B, p))
        s0 = np.zeros(self.B)

        lci = np.zeros((self.B, p))
        uci = np.zeros((self.B, p))
        ses = np.zeros((self.B, p))
        df_res = np.zeros(self.B)

        for b in range(self.B):
            pvals[b, :], s0[b], coefs[b, :], lci[b, :], uci[b, :], ses[b, :], df_res[b] = self.singlesplit(X, y, nleft)


        # for b in range(self.B):
        # pvals[b,:],coefs[b,:],lci[b,:],uci[b,:],ses[b,:],df_res[b]= self.singlesplit(X,y,nleft)

        if self.return_nonaggr:
            self.p_nonaggr = pvals

        ######## Aggregate into final p-values ########

        if not (0.05 in self.gamma):
            print("0.05 is not in gamma range due to the choice of B, the results might be incorrect.")

        if len(self.gamma) > 1:
            penalty = 1 - np.log10(np.min(self.gamma))
        else:
            penalty = 1

        quant_gamma = np.quantile(pvals, self.gamma, axis=0) / self.gamma.reshape(self.gamma.shape[0], 1)
        inf_quant = np.min(quant_gamma, axis=0)
        pvals_pre = inf_quant * penalty
        pvals_current = np.minimum(pvals_pre, 1)

        self.pvals_corr = pvals_current
        self.gamma_min = np.min(self.gamma)

        print("P-values Done")
        ######## Aggregate Confidence intervals

        if self.aggr_ci:
            self.s0 = 0
            self.d = p
            self.lci, self.uci = self.aggregate_ci(L=lci, U=uci, S0=s0, C=coefs, DF_RES=df_res, SES=ses)
        else:
            self.lci = np.median(lci, axis=1)
            self.uci = np.median(uci, axis=1)
        
        return pvals_current
         

    def does_it_cover(self, betaj, ci_info, multi_corr=False):
        alpha = 1 - self.ci_level

        ci_length = ci_info[2]
        no_inf_ci, center, ses, s0, df_res = ci_info[4:]
        # ci_Information = [lci, uci, ci_length, inf_ci, no_inf_ci, center, ses, s0, df_res]

        # the rank of the p-value in increasing order
        pval_rank = np.argsort(-np.true_divide(np.abs(betaj - center), ci_length / 2))
        # the number of ci + the inf ci we left out
        nsplit = len(pval_rank) + no_inf_ci

        gamma_b = pval_rank / nsplit

        if multi_corr:
            level = (1 - s0 * alpha * gamma_b / (1 - np.log(self.gamma_min)))
        else:
            level = (1 - alpha * gamma_b / (1 - np.log(self.gamma_min)))

        a = 1 - (1 - level) / 2

        if (gamma_b <= self.gamma_min).all():
            return True
        else:
            fac = norm.ppf(a, df_res)

            nlci = (center - fac * ses)
            nuci = (center + fac * ses)

            position = (gamma_b > self.gamma_min)
            if (position).any():
                return (nlci[position] <= betaj).all() and (nuci[position] >= betaj).all()

    def find_bisection_bounds(self, shouldcover, shouldnotcover, ci_info):
        reset_shouldnotcover = False
        if self.does_it_cover(shouldnotcover, ci_info):
            reset_shouldnotcover = True
            while self.does_it_cover(shouldnotcover, ci_info):
                old = shouldnotcover.copy()
                shouldnotcover = shouldnotcover + (shouldnotcover - shouldcover)
                shouldcover = old.copy()
        if not self.does_it_cover(shouldcover, ci_info):
            if reset_shouldnotcover:
                sys.exit("Problem")
            while self.does_it_cover(shouldcover, ci_info):
                old = shouldcover.copy()
                shouldcover = shouldcover + (shouldcover - shouldnotcover)
                shouldnotcover = old.copy()

        return shouldcover, shouldnotcover

    def check_bounds(self, shouldcover, shouldnotcover, ci_info):
        if self.does_it_cover(shouldnotcover, ci_info):
            sys.exit("Shouldnotcover bound is covered")
        if not self.does_it_cover(shouldcover, ci_info):
            sys.exit("Shouldcover is a bad covered bound.")

    def bisection_gammamin_coverage(self, outer, inner, eps_bound, ci_info):
        self.check_bounds(inner, outer, ci_info)
        eps = 1
        while eps > eps_bound:
            middle = (outer + inner) / 2
            if self.does_it_cover(middle, ci_info):
                inner = middle.copy()
            else:
                outer = middle.copy()
            eps = np.abs(inner - outer)
        solution = (inner + outer) / 2
        return solution

    def aggregate_ci(self, L, U, S0, C, DF_RES, SES):

        low = np.min(C, axis=1)
        high = np.max(C, axis=1)

        ###### find an inside point: we need to find a point that is definitely in the confidence intervals

        range_len = 10
        l_bound = np.empty(self.d)
        u_bound = np.empty(self.d)

        for i in range(self.d):
            lci = L[:, i].copy()
            uci = U[:, i].copy()

            inf_ci = np.isinf(lci) + np.isinf(uci)

            center = C[~inf_ci, i].copy()
            ses = SES[~inf_ci, i].copy()

            no_inf_ci = sum(inf_ci)

            if no_inf_ci == len(lci) or no_inf_ci >= (1 - self.gamma_min) * len(lci):
                l_bound[i] = -np.Inf
                u_bound[i] = np.Inf
                continue

            lci = lci[~inf_ci]
            uci = uci[~inf_ci]
            ci_length = uci - lci

            df_res = DF_RES[~inf_ci]
            s0 = S0[~inf_ci]

            ci_Information = [lci, uci, ci_length, inf_ci, no_inf_ci, center, ses, s0, df_res]

            test_range = np.linspace(low[i], high[i], range_len)

            # does it cover gamma min?
            cover = np.array([False] * range_len)

            for j in range(range_len):
                cover[j] = self.does_it_cover(test_range[j], ci_Information)

            while not any(cover):
                range_len = range_len * 10
                test_range = np.linspace(low[i], high[i], range_len)
                cover = np.array([False] * range_len)
                for r in range(range_len):
                    cover[r] = self.does_it_cover(test_range[r], ci_Information)
                if range_len > 10 ** 3:
                    sys.exit("Find no inside point.")

            ##### Inner point
            inner = np.min(test_range[cover])
            outer = np.min(lci)

            ##### Generate new bounds
            ninner, nouter = self.find_bisection_bounds(inner, outer, ci_Information)
            l_bound[i] = self.bisection_gammamin_coverage(nouter, ninner, 10 ** (-7), ci_Information)

            outer = np.max(uci)

            ninner, nouter = self.find_bisection_bounds(ninner, outer, ci_Information)
            u_bound[i] = self.bisection_gammamin_coverage(nouter, ninner, 10 ** (-7), ci_Information)

        return l_bound, u_bound

