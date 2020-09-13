import numpy as np
import math

class simulation:
    def __init__(self, n, p, snr, type,seed):
        self.type = type
        self.n = n
        self.p = p
        self.snr= snr
        self.seed = seed

    def get_sigma(self):
        # Generate a designed matrix sigma

        def copytri(matrix):
            # produce a symmetrix matrix with an triangular matrix
            uppertri = np.triu(matrix, 1)
            lowertri = np.transpose(uppertri)
            out = np.diag(np.diag(matrix)) + uppertri + lowertri
            return out

        if self.type == 'Toeplitz':
            indice = np.zeros((self.p, self.p))
            temp = np.arange(self.p)
            indice[0, :] = temp.copy()
            j = 1
            while j < self.p:
                indice[j, j:] = temp[:-j].copy()
                j = j + 1
            indice = copytri(indice)
            return np.power(0.9,indice)

        if self.type == 'Exp_decay':
            indice = np.zeros((self.p, self.p))
            temp = np.arange(self.p)
            indice[0, :] = temp.copy()
            j = 1
            while j < self.p:
                indice[j, j:] = temp[:-j].copy()
                j = j + 1
            indice = copytri(indice)
            pow_indice = np.power(0.4, (indice / 5))

            return np.linalg.inv(pow_indice)

        if self.type == 'Equi_corr':
            sigmaE = np.full(shape=(self.p, self.p), fill_value=0.8)
            np.fill_diagonal(sigmaE, val=1)
            return sigmaE

        if self.type == 'Circulant':
            sigma = np.eye(self.p)
            circulate = [0.1]*5 + [0.0]*(self.p-11) + [0.1]*(5)
            for j in range(self.p):
                sigma[j,(j + 1):] = circulate[:(self.p-1-j)]
                sigma[j, :j] = circulate[(self.p-1-j):self.p]
            return sigma

    def get_coefs(self, s0=None, random_coef=False, b=1, unif_up=None, unif_lw=None,
                  save=True, path="./",filename="data"):
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
        if not None:
            s0=round(self.n*0.1)
        np.random.seed(self.seed)
        active_index = np.random.randint(low=1, high=self.p, size=s0, dtype='l')
        betas = np.zeros(self.p)
        if random_coef:
            betas[active_index] = np.random.randint(low=unif_lw, high=unif_up, size=s0, dtype='l')
        else:
            betas[active_index] = b

        if save:
            np.savetxt(path + filename + "_betas.csv", betas,
                       fmt="%.4f", delimiter=",")
            np.savetxt(path + filename + "_index.csv", active_index,
                       fmt="%.4f", delimiter=",")

        return betas, active_index

    def get_data(self, verbose=False,save=True,
                 path="./",filename="data"):
        sigma = self.get_sigma()
        if verbose:
            print("The used covariance matrix is:")
            print(sigma)
        # Designed matrices are generated ~ Np(0,sigma)
        np.random.seed(self.seed)
        X = np.random.multivariate_normal(mean=np.zeros(self.p), cov=sigma, size=self.n)
        noise = np.random.normal(loc=0,scale=1/self.snr,size=self.n)
        betas, active_index=self.get_coefs()
        Y = np.dot(X, betas) + noise
        sim_data = np.column_stack((Y, X))

        if save:
            np.savetxt(path + filename + ".csv", sim_data,
                       fmt="%.4f", delimiter=",")

        return X, Y

