""" alpha_pass function for HMM
    alpha_pass.py - 24-10-2019
    Author: Kasper Meldgaard
    #--#
    function to complete the alpha pass of HMM training.
    --> also known as the forward algorithm.
    Returns the probability of a partial observation sequence (upto time t=[0...T-1])
    #--#
    model: the current HMM model object
    O: observation sequence
    """

# package dependencies
import numpy as np
import logging


def alpha_pass(model, O):
    alpha = np.zeros((model.get_N, len(O)))  # initialize alpha matrix
    # c0 = 0
    c = np.zeros(len(O))    # initialize c array with zeros
    # compute first row:
    for n in range(model.get_N):
        alpha[0, n] = model.pi[n] * model.B[O[0], n]
        c[0] += alpha[0, n]
    # compute rest of alpha rows
    for t in range(1, (len(O) - 1)):
        # ct = 0
        for n in range(model.get_N):  # denoted i in paper
            alpha[t, n] = 0
            for i in range(model.get_N):  # denoted j in paper
                alpha[t, n] += alpha[t - 1, i] * model.A[i, n]
            alpha[t, n] *= model.B[n, O(t)]
            ct += alpha[t, n]
        # scale alpha:
        ct = 1 / ct
        for n in range(model.get_N):
            alpha[t, n] *= ct
            # end scale

    return alpha, c
