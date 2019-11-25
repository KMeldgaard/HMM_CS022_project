"""
By Zalasyu
"""
import numpy as np

"""
T = length of the observation sequence
N = number of states in the model
M = number of observation symbols
Q = {q0, q1, . . . , qN−1} = distinct states of the Markov process
V = {0, 1, . . . , M − 1} = set of possible observations
A = state transition probabilities
B = observation probability matrix
π = initial state distribution
O = (O0, O1, . . . , OT −1) = observation sequence.
i = S_i (present state)
j = S_j (state after transition)
"""
"""Used to understand our raw inputs."""
# transition probabilities
transition = np.array([[0.46, 0.54],
                       [0.52, 0.48]])
# Emission probabilities
emission = np.random.normal(loc=1 / 27, scale=0.5, size=(2, 27))
print(emission)
# defining states and sequence symbols
states = ['V', 'C']
states_dic = {'V': 0, 'C': 1}
sequence_syms = {
    'a': 0, 'o': 1, 'u': 2, 'i': 3, 'e': 4, 'y': 5, 'b': 6, 'c': 7, 'd': 8, 'f': 9, 'g': 10, 'h': 11, 'j': 12,
    'k': 13, 'l': 14, 'm': 15, 'n': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'v': 22, 'x': 23, 'z': 24, 'w': 25,
    ' ': 26}

sequence = ['a,', 'o', 'u', 'i', 'e', 'y', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p',
            'q', 'r', 's', 't', 'v', 'w' 'x', 'z', ' ']

# test sequence
test_sequence = 'glhgaeiproekrertgdqth'
test_sequence = [x for x in test_sequence]

# probabilities of going to end state
end_probs = [0.52, 0.48]
# probabilities of going from start state
start_probs = [0.52, 0.48]

"""
emission probabilities: numpy matrix of shape (N, |V|) (where |V| is the size of the vocabulary)
where entry (j, o) has emis_j(o).

transition probablities: numpy matrix of shape (N, N) where entry (i, j) has trans(i, j).

forward and backward probabilities: numpy matrix of shape (N, T) where entry (s, t) has forward probability
for state s and observation t
"""

#  Need Forward Algorithim => P(o1,o2 ...ot, qt = j | λ)
#       qt = j means “the t^th state in the sequence of states is state j”
"""
1. Initialization:
    α1(j) = πj*bj(o1) 1 ≤ j ≤ N
2. Recursion:
    αt(j) = N_Sigma_i=1( αt−1(i) * aij * bj(ot)) 1 ≤ j ≤ N,1 < t ≤ T
3. Termination:
    P(O|λ) = N_Sigma_i=1(αT(i))   <= Critical!!!!!!

"""

#  Need Backward Probability => P(o_t+1,o_t+2....oT | q_t=Si , λ)
# starting in state Si at time t and generating the rest of the observation sequence ot+1,..., oT.
"""
1. Initialization:
    Beta_T(i) = 1,  1 ≤ i ≤ N
2. Recursion:
    beta_t(i) = N_Sigma_j=1( aij * bj(o_t+1) * Beta_t+1(j) )
3. Termination:
    P(O|λ) = N_sigma_j=1( πj * bj(o1) * β1(j) ) 

"""


# P(O|λ) = N_Sigma_i=1(αT(i)) <= Denominator <= From Kasper's return should be that stored in variable alpha??
# But this ^^^^^^^^^^^^^^^^^^  is Dynamic with each new trans matrix made.... right??
# Also it is normalized. due to scaling block.....
# The denominator can be computed in many different ways, all producing the same result.

def Forward_Backward(N, alpha, beta, test_sequence, sequence_syms, transition, emission):
    """Expectation Step"""

    # Initialize denominator:
    P_O_given_model = np.sum(alpha)  # alpha matrix from Kasper.
    # This will do the following mathematical operation =>N_Sigma_i=1(αT(i)) to get the total probability that the
    # current Aij will output the observation sequence. Therefore, a scalar must be outputted above.

    di_gamma = []  # Zt = a list of T matrices of size (N,N) The probability of being in state s_i at time t and
    # transiting to state s_j at time t=1
    # Computing di_gamma
    for t in range(len(test_sequence) - 1):  # Creates index list
        obs_idx = sequence_syms[t + 1]
        Z_t = np.zeros((N, N))

        for s_i in range(N):
            for s_j in range(N):
                Z_t[s_i, s_j] = alpha[s_i, t] * transition[s_i, s_j] * emission[s_j, obs_idx] * \
                                beta[s_j, t + 1] / P_O_given_model
        di_gamma.append(Z_t)

    # Computing gamma. To adjust the model parameters to best fit the observations.
    # Gamma_t(i) = sum(gamma(i,j)
    gamma = np.zeros((N + 2, len(test_sequence)))
    for t in range(len(test_sequence) - 1):
        for s_i in range(N):
            gamma[s_i, t] = sum([di_gamma[t][s_i, s_j] for s_j in range(N)])

        for s_j in range(N):
            gamma[s_j, len(test_sequence) - 1] = sum([di_gamma[t][s_i, s_j] for s_i in range(N)])

    """ Maximization Step """
    start_probs = np.array([gamma[s_i, 0] for s_i in range(N)])  # Heuristic Search for global maximum. This will be
    # used if the fow&back function is called again. (aka gamma_0(j))

    # re-estimate emis probabilities
    emis = np.zeros((N, len(sequence_syms)))

    for s in range(N):
        denominator = sum([gamma[s, t] for t in range(len(test_sequence))])
        for vocab_item, obs_index in sequence_syms.items():
            emis[s, obs_index] = sum(
                [gamma[s, t] for t in range(len(test_sequence)) if test_sequence[t] == vocab_item]) / denominator

    # re-estimate transition probabilities
    trans = np.zeros((N, N))

    for s_i in range(N):
        denominator = sum([gamma[s_i, t] for t in range(len(test_sequence) - 1)])

        for s_j in range(N):
            trans[s_i, s_j] = sum([di_gamma[t][s_i, s_j] for t in range(len(test_sequence) - 1)]) / denominator

    return start_probs, trans, emis, gamma, di_gamma

