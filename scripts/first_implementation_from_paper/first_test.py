""" CS022B - SJSU - Project
    First implementation of all functions
    first_test.py - 25-11-2019
    Author: Kasper Meldgaard"""

# import all the shit
import numpy as np
from hmm import HMM
from observation import Latin_observations
from alpha_pass_Copy_With_Changes import alpha_pass
from beta_pass import beta_pass
from log_prob import compute_logprob
# from ForwardBackward import Forward_Backward

# Initialize calculation
max_iteration = 10
old_log_prob = -float('inf')
log_prob_history = np.array(old_log_prob, dtype=float)

model = HMM(2,27)
obs = Latin_observations('test_text.txt')

for iteration in range(max_iteration):
    # alpha pass
    alpha, c = alpha_pass(model, obs)

    # beta pass
    beta = beta_pass(model, obs, c)

    # gamma-pass
