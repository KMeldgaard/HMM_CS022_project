""" Test script fom alpha-pass combined with model and obs objects
    alpha_test.py - 20-11-2019
    Author: Maximiliano Luchsinger"""

from hmm import HMM
from observation import Latin_observations
from alpha_pass import alpha_pass
from beta_pass import beta_pass

# setup objects
obs = Latin_observations("test_text.txt")
model = HMM(2, 27);

# attempt an alpha-pass
alpha, c = alpha_pass(model, obs)

# debug prints
print("Model:\nA:\n", model.A, "\nB:\n", model.B, "\npi:\n", model.pi)
print("Observations:\nSequence:\n", obs.obs, "\n# of obs:", obs.num_obs)
print("Alpha matrix:\n", alpha)
print("c vector:\n", c)




# ----------------------------- BETA-PASS TEST ----------------------------------------------------
beta = beta_pass(model, obs, c)
# debug prints
print("Beta matrix:\n", beta)
