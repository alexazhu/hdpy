import numpy as np
import math


def copytri(m):
    uptri = np.triu(m, 1)
    dwtri = np.transpose(uptri)
    out = np.diag(np.diag(m)) + uptri + dwtri
    return out

class simulation:

    def Sigma(self,type,n,p):
        sigma = np.identity(p)
        if type=='Toeplitz':
            sigmaT = sigma.copy()
            for i in range(p):
                t = p-1
                index = i+1
                while index < p:
                    sigmaT[i,index] = math.pow(0.9,abs(index-i))
                    index = index + 1

            return copytri(sigmaT)
        else:
            if type == 'Exp_decay':
                sigma_inv = np.identity(p)
                for i in range(p):
                    t = p - 1
                    index = i+1
                    while index < p:
                        sigma_inv[i, index] = math.pow(0.4, abs(index - i)/5)
                        index = index + 1
                sigma_inv = copytri(sigma_inv)
                sigmaD = np.linalg.inv(sigma_inv)
                return sigmaD

            elif type == 'Equi_corr':
                sigmaE = np.full(shape=(p,p),fill_value=0.8)
                sigmaE = np.fill_diagonal(sigmaE,val=1)
                return sigmaE

            elif type == 'circulant':
                for i in range(p):
                    sigma[i,(i+1):(i+5)] = 0.1
                    sigma[i,(i+p-5):(i+p-1)] = 0.1
                sigma = copytri(sigma)
                return sigma

            elif type == 'normal':
                return sigma
        return 0


    def GetB(self,p,s0,b):
        s = np.random.randint(low=1,high=p,size=s0,dtype='l')
        beta = np.zeros(p)
        for i in range(s0):
            beta[s[i]] = b
        return beta


    def GetX(self,type,n,p):
        sigma = self.Sigma(type,n,p)
        print(sigma)
        X = np.random.multivariate_normal(mean=np.zeros(p),cov=sigma,size=n)
        return X

    def GetSample(self,type,n,p,s0,b):
        x = self.GetX(type,n,p)
        noise = np.random.normal(size=n)
        beta = self.GetB(p, s0, b)
        Y = np.dot(x,beta)+noise
        return beta,x,Y

