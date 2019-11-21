"""
    Beta_pass funtion for HMM
    beta_pass.py - 11-06-2019
    Author: Maximiliano Luchsinger

    #--#
    function to complete the beta pass.
    --> also known as the backward algorithm.

    Given Model lambda and an observation sequence,
    finds an optimal state sequence --> uncover the hidden part of the HMM
    #--#
    :param
    model:      The current HMM model object
    obs_seq:    observation sequence object
    c:          row-vector to scale the values
                (solution for the underflow problem --> converging to 0 exponentially as T increases)
    :return
    beta:       beta matrix
"""

import numpy as np

def beta_pass(model, obs_seq, c):
    beta = np.zeros((model.get_N(), obs_seq.get_num_obs()))

    # set all elements of last row to scaled value c[t-1]
    # WHY? NO CLUE YET
    for n in range(model.get_N()):    # n denoted i in paper
        # Scaled values assigned to last row
        beta[n, obs_seq.get_num_obs()-1] = c[obs_seq.get_num_obs() - 1]

    # Compute rest of Beta-Pass
    for t in range(obs_seq.get_num_obs()-2, 0, -1):
        for n in range(0, model.get_N()):      # n denoted i in paper
            beta[n, t] = 0
            for i in range(0, model.get_N()):  # i denoted j in paper
                beta[n, t] = beta[n, t] + model.A[n, i] * model.B[i, obs_seq.obs[t+1]] * beta[i, t+1]

            # Scaling Beta[t,n] with same scale c as alpha[t,n]
            beta[n, t] = c[t] * beta[n, t]

    return(beta)