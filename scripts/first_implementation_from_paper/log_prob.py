""" Function to compute the log probability of P(O | lambda)
    log_prob.py - 24.11.2019
    Author: Maximiliano Luchsinger"""

import numpy as np

def compute_logprob(obs_seq, c):
    logProb = 0
    for i in range(0, obs_seq.get_num_obs()):
        logProb = logProb + np.log(c[i])

    logProb = -logProb
    # Log Likelihood can go from -Inf to 0
    return(logProb)