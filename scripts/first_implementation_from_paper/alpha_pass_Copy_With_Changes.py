""" alpha_pass function for HMM
    alpha_pass.py - 24-10-2019
    Author: Kasper Meldgaard
    #--#
    function to complete the alpha pass of HMM training.
    --> also known as the forward algorithm.
    Returns the probability of a partial observation sequence (upto time t=[0...T-1])
    #--#

    :param
    model:      the current HMM model object
    obs_seq:    observation sequence object

    :return
    alpha:  the matrix describing how good the current model fit the obs_seq
    c:      row-vector of some constant to be used in beta-pass
    """

# package dependencies
import numpy as np
import logging


def alpha_pass(model, obs_seq):
    alpha = np.zeros((model.get_N(), obs_seq.get_num_obs()))  # initialize alpha matrix
    # c0 = 0
    c = np.zeros(obs_seq.get_num_obs())    # initialize c array with zeros
    # compute first row:
    for n in range(model.get_N()):
        alpha[0, n] = model.pi[n] * model.B[n, obs_seq.obs[0]]
        c[0] += alpha[0, n]

    # Scale alpha_0(i)
    c[0] = 1/c[0]
    for i in range(model.get_N()):    # CHANGED TO model.get_N() instead of obs_seq.get_num_obs()
        alpha[0, i] *= c[0]

    # compute rest of alpha/c rows
    for t in range(1, (obs_seq.get_num_obs())):     # possible remove the '-1'
        # ct = 0
        for i in range(model.get_N()):  # i in paper
            alpha[i, t] = 0
            for j in range(model.get_N()):  # denoted j in paper
                alpha[i, t] += alpha[j, t - 1] * model.A[j, i]
            alpha[i, t] *= model.B[i, obs_seq.obs[t]]
            c[t] += alpha[i, t]

        # scale alpha_t(i):
        c[t] = 1 / c[t]
        for i in range(model.get_N()):
            alpha[i, t] *= c[t]
        # end scale

    return alpha, c
