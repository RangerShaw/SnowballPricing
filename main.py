# -*- coding: utf-8 -*-
from snowball import SnowBall

import numpy as np
import pickle as pk


def EulerMilsteinMCStock(scheme, parameters):
    np.random.seed(1000)

    # time setup
    T = parameters['setup']['T']  # total time/maturity
    numSteps = parameters['setup']['numSteps']  # number of steps
    numPaths = parameters['setup']['numPaths']  # number of simulated paths
    dt = parameters['setup']['dt']

    # model parameters
    S_0 = parameters['model']['S0']  # initial value
    sigma = parameters['model']['sigma']  # initial value
    rf = parameters['model']['rf']  # initial value

    # simulation
    S = np.zeros((numSteps + 1, numPaths), dtype=float)
    S[0, :] = np.log(S_0)

    # simulations for asset price S
    for i in range(numPaths):
        for t_step in range(1, numSteps + 1):
            # the 2 stochastic drivers for variance V and asset price S and correlated
            Zs = np.random.normal(0, 1, 1)
            S[t_step, i] = S[t_step - 1, i] + (rf - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * Zs

    return np.exp(S)


def load_pickle_data(fp):
    f = open(fp, 'rb')
    return np.array(pk.load(f))


if __name__ == '__main__':
    sb = SnowBall(1.0, 0.036, 0.3, 1, 1.0, 0.75)
    fp = 'data/QuasiRand.pickle'
    sb.solve_mc(fp)
