import numpy as np
import math


# produce a symmetrix matrix with an triangular matrix
def copytri(m):
    uppertri = np.triu(a=m, k=1)
    lowertri = np.transpose(uppertri)
    out = np.diag(np.diag(m)) + uppertri + lowertri
    return out


class simulation:
    def __init__(self, n, p, type):
        self.type = type
        self.n = n
        self.p = p

    def get_sigma(self):
        # Generate covariance matrix for the designed matrices
        sigma = np.identity(self.p)

        if self.type == 'Toeplitz':
            sigmaT = sigma.copy()
            for i in range(self.p):
                t = self.p - 1
                index = i + 1
                while index < self.p:
                    sigmaT[i, index] = math.pow(0.9, abs(index - i))
                    index = index + 1

            return copytri(sigmaT)
        else:
            if self.type == 'Exp_decay':
                sigma_inv = np.identity(self.p)
                for i in range(self.p):
                    t = self.p - 1
                    index = i + 1
                    while index < self.p:
                        sigma_inv[i, index] = math.pow(0.4, abs(index - i) / 5)
                        index = index + 1
                sigma_inv = copytri(sigma_inv)
                sigmaD = np.linalg.inv(sigma_inv)
                return sigmaD

            elif self.type == 'Equi_corr':
                sigmaE = np.full(shape=(p, p), fill_value=0.8)
                sigmaE = np.fill_diagonal(sigmaE, val=1)
                return sigmaE

            elif self.type == 'circulant':
                for i in range(self.p):
                    sigma[i, (i + 1):(i + 5)] = 0.1
                    sigma[i, (i + self.p - 5):(i + self.p - 1)] = 0.1
                sigma = copytri(sigma)
                return sigma

            elif type == 'normal':
                return sigma
        return 0

    def get_coefs(self, s0, random_coef=False, b=None, unif_up=None, unif_lw=None):
        
        """ 
        -----------
        Parameters:
        -----------
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
    
        active_index = np.random.randint(low=1, high=self.p, size=s0, dtype='l')
        betas = np.zeros(self.p)
        if random_coef:
            betas[active_index] = np.random.randint(low=unif_lw, high=unif_up, size=s0, dtype='l')
        else:
            betas[active_index] = b

        self.coefs = betas
        return betas

    def get_data(self, verbose=False):
        sigma = self.get_sigma()
        if verbose == True:
            print("The used covariance matrix is:")
            print(sigma)
        # Designed matrices are generated ~ Np(0,sigma)
        X = np.random.multivariate_normal(mean=np.zeros(self.p), cov=sigma, size=self.n)
        noise = np.random.normal(size=self.n)
        betas = self.coefs
        Y = np.dot(X, betas) + noise
        return X, Y

