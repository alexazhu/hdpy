__author__ = 'Ziyan Zhu'

import numpy as np
import random


class geneSimulation(object):
    def Beta(self, G1, g1, n1, G2, g2, n2, G3, g3, n3, G4, g4, n4, N_zero=2200 - 44):

        gene1 = np.array([G1])
        g1 = np.array(g1)
        n1 = np.array(n1)
        gene1 = np.append(gene1, np.repeat(g1, n1))

        gene2 = np.array([G2])
        g2 = np.array(g2)
        n2 = np.array(n2)
        gene2 = np.append(gene2, np.repeat(g2, n2))

        gene3 = np.array([G3])
        g3 = np.array(g3)
        n3 = np.array(n3)
        gene3 = np.append(gene3, np.repeat(g3, n3))

        gene4 = np.array([G4])
        g4 = np.array(g4)
        n4 = np.array(n4)
        gene4 = np.append(gene4, np.repeat(g4, n4))

        emp = np.repeat([0], N_zero)

        b = np.hstack((gene1, gene2, gene3, gene4, emp))

        return b

    def Simulation(self, b, n, p, seed):  # 200 TFs genes, each regulates 10 genes.


        b1 = self.Beta(5, 5 / np.sqrt(10), 10, -5, -5 / np.sqrt(10), 10, 3, 3 / np.sqrt(10), 10, -3, -3 / np.sqrt(10),
                       10)

        b3 = self.Beta(5, 5 / 10, 10, -5, -5 / 10, 10, 3, 3 / 10, 10, -3, -3 / 10, 10)

        b2 = self.Beta(5, [-5 / np.sqrt(10), 5 / np.sqrt(10)], [3, 7], -5, [5 / np.sqrt(10), -5 / np.sqrt(10)], [3, 7],
                       3, [-3 / np.sqrt(10), 3 / np.sqrt(10)], [3, 7], -3, [3 / np.sqrt(10), -3 / np.sqrt(10)], [3, 7])

        b4 = self.Beta(5, [-5 / 10, 5 / 10], [3, 7], -5, [5 / 10, -5 / 10], [3, 7],
                       3, [-3 / 10, 3 / 10], [3, 7], -3, [3 / 10, -3 / 10], [3, 7])

        if b == 'b1':
            b = b1
        elif b == 'b2':
            b = b2
        elif b == 'b3':
            b = b3
        elif b == 'b4':
            b = b4

        sigma = np.sqrt(np.sum(b ** 2) / 4)

        random.seed(seed)
        e = np.random.normal(loc=0, scale=sigma, size=n)

        X = np.empty(shape=(n, p))
        Xtf = np.empty(shape=(n, 200))
        Y = np.empty(shape=(n, 1))

        for i in range(n):
            Xtf[i, :] = np.random.normal(loc=0, scale=1, size=200)
            Xtemp = np.array([0])
            for j in range(200):
                gene = np.random.normal(loc=0.7 * Xtf[i, j], scale=0.51, size=10)
                temp = np.append(Xtf[i, j], gene)
                Xtemp = np.append(Xtemp, temp)
            Xtemp = Xtemp[1:]
            X[i, :] = Xtemp

        for v in range(n):
            Y[v, :] = np.dot(X[v, :], b) + e[v]

        return X, Y



from Simulation_HDI import geneSimulation

func = geneSimulation()

x, y = func.Simulation(b='b1', n=200, p=2200, seed=50)

print(y)
print(x)
