""" CS022B - SJSU - Project
    First implementation of all functions
    first_test.py - 25-11-2019
    Author: Kasper Meldgaard"""

# consts:
sequence_syms = {
    'a': 0, 'o': 1, 'u': 2, 'i': 3, 'e': 4, 'y': 5, 'b': 6, 'c': 7, 'd': 8, 'f': 9, 'g': 10, 'h': 11, 'j': 12,\
    'k': 13, 'l': 14, 'm': 15, 'n': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'v': 22, 'x': 23, 'z': 24, 'w': 25,\
    ' ': 26}


# import all the shit
import numpy as np
from hmm import HMM
import pandas as pd
from matplotlib import pyplot
from observation import Latin_observations
from alpha_pass_Copy_With_Changes import alpha_pass
from beta_pass import beta_pass
from log_prob import compute_logprob
from ForwardBackward import Forward_Backward
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
# ------------------ #

# Initialize calculation
max_iteration = 100
old_log_prob = -float('inf')
log_prob_history = np.array(old_log_prob, dtype=float)

model = HMM(2, 27)
obs = Latin_observations('guide_to_wireless_intro.txt')
# obs = Latin_observations('test_text.txt')

B_res = pd.DataFrame(model.B)
A_res = pd.DataFrame(model.A)
pi_res = pd.DataFrame(model.pi)

for iteration in range(max_iteration):
    if iteration == max_iteration-1:
        print("MAX ITERATIONS DONE! (", max_iteration, ")")
    if iteration % 10 == 0:
        print("ITERATION", iteration)
    # alpha pass
    alpha, c = alpha_pass(model, obs)

    # beta pass
    beta = beta_pass(model, obs, c)

    # gamma-pass
    start_probs, trans, emis, gamma, di_gamma = \
        Forward_Backward(model.get_N(), alpha, beta, obs.obs, sequence_syms, model.A, model.B)

    # update model
    model.A = trans
    model.B = emis
    model.pi = start_probs

    # compute lob probs
    new_log_prob = compute_logprob(obs, c)

    # compare
    log_prob_history = np.append(log_prob_history, new_log_prob)
    if new_log_prob > old_log_prob:
        old_log_prob = new_log_prob
        continue
    else:
        print("STOPPED AT ITERATION:", iteration+1)
        break

"""view some results:"""
# nice pandas display
col_names = [chr(c) for c in range(ord('a'), ord('z')+1)]
col_names.append('space')
# append B result
B_res = B_res.append(pd.DataFrame(model.B))
B_res = B_res.round(5)
B_res.columns = col_names
B_res_trans = B_res.transpose()
# append A & pi result
A_res = A_res.append(pd.DataFrame(model.A))
A_res.columns = ['0', '1']
print("A matrix:\n", A_res)

pi_res = pi_res.append(pd.DataFrame(model.pi))
print("pi matrix:\n", pi_res)

# print(B_res)
print("B matrix:\n", B_res_trans)
# export to excel
B_res.to_csv('export.csv', index=None)

pyplot.plot(range(len(log_prob_history)), log_prob_history.tolist())
pyplot.show()