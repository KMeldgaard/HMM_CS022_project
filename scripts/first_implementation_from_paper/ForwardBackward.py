"""
By Zalasyu
"""
import numpy as np


# P(O|λ) = N_Sigma_i=1(αT(i)) <= Denominator <= From Kasper's return should be that stored in variable alpha??

def Forward_Backward(N, alpha, beta, test_sequence, sequence_syms, transition, emission):
    """Expectation Step"""

    # Initialize denominator:
    P_O_given_model = np.sum(alpha)

    di_gamma = []
    # Computing di_gamma
    for t in range(len(test_sequence) - 1):  # Creates index list
        obs_idx = test_sequence[t + 1]
        Z_t = np.zeros((N, N))

        for s_i in range(N):
            for s_j in range(N):
                Z_t[s_i, s_j] = alpha[s_i, t] * transition[s_i, s_j] * emission[s_j, obs_idx] * \
                                beta[s_j, t + 1] / P_O_given_model
        di_gamma.append(Z_t)

    # Computing gamma. To adjust the model parameters to best fit the observations.
    # Gamma_t(i) = sum(gamma(i,j)
    gamma = np.zeros((N, len(test_sequence)))
    for t in range(len(test_sequence) - 1):
        for s_i in range(N):
            gamma[s_i, t] = sum([di_gamma[t][s_i, s_j] for s_j in range(N)])

        for s_j in range(N):
            gamma[s_j, len(test_sequence) - 1] = sum([di_gamma[t][s_i, s_j] for s_i in range(N)])

    """ Maximization Step """
    start_probs = np.array([gamma[s_i, 0] for s_i in range(N)])

    # re-estimate emis probabilities
    emis = np.zeros((N, len(sequence_syms)))

    for s in range(N):
        denominator = sum([gamma[s, t] for t in range(len(test_sequence))])
        for vocab_item, obs_index in sequence_syms.items():
            emis[s, obs_index] = sum(
                [gamma[s, t] for t in range(len(test_sequence)) if test_sequence[t] == obs_index]) / denominator

    # re-estimate transition probabilities
    trans = np.zeros((N, N))

    for s_i in range(N):
        denominator = sum([gamma[s_i, t] for t in range(len(test_sequence) - 1)])

        for s_j in range(N):
            trans[s_i, s_j] = sum([di_gamma[t][s_i, s_j] for t in range(len(test_sequence) - 1)]) / denominator

    return start_probs, trans, emis, gamma, di_gamma

