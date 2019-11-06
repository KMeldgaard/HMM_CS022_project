""" Beta_pass funtion for HMM
    beta_pass.py - 24-10-2019
    Author: Kasper Meldgaard"""
# Given Model lambda and an observation sequence,
# find an optimal state sequence --> uncover the hidden part of the HMM


import numpy as np

# C_t hast to be passed on from alpha pass?
def beta_pass(model, obs_seq, c):
    beta = np.zeros((model.get_N(), obs_seq.get_num_obs()))

    # set all elements of last row to = 1
    # WHY? NO CLUE YET
    for n in range(model.get_N()):    # n denoted i in paper
        # Scaled values assigned to last row
        beta[obs_seq.get_num_obs()-1, n] = c[obs_seq.get_num_obs - 1]

    # Compute rest of Beta-Pass
    for t in range(start=obs_seq.get_num_obs-2, stop=0, step=-1):
        for n in range(start=0, stop=model.get_N()):      # n denoted i in paper
            beta[t, n] = 0
            for i in range(start=0, stop=model.get_N()):  # i denoted j in paper
                beta[t, n] = beta[t, n] + model.A[n, i] * model.B[i, obs_seq.obs[t+1]] * beta[t+1, i]

            # Scaling Beta[t,n] with same scale as alpha[t,n]
            beta[t, n] = c[t] * beta[t, n]

    return(beta)