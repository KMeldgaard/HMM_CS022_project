import numpy as np
from fractions import Fraction as frac

# generating initial probabilities

# transition probabilities
transition = np.array([[0.46, 0.54],
                       [0.52, 0.48]])
# Emission probabilities
emission = np.random.normal(loc=1/27,scale=0.5, size=(2, 27))
print(emission)
# defining states and sequence symbols
states = ['V', 'C']
states_dic = {'V': 0, 'C': 1}
sequence_syms = {
    'a': 0, 'o': 1, 'u': 2, 'i': 3, 'e': 4, 'y': 5, 'b': 6, 'c': 7, 'd': 8, 'f': 9, 'g': 10, 'h': 11, 'j': 12,
    'k': 13, 'l': 14, 'm': 15, 'n': 16, 'p': 17, 'q': 18, 'r': 19, 's': 20, 't': 21, 'v': 22, 'x': 23, 'z': 24, 'w': 25}

sequence = ['a,', 'o', 'u', 'i', 'e', 'y', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p',
            'q', 'r', 's', 't', 'v', 'w' 'x', 'z']

# test sequence
test_sequence = 'glhgaeiproekrertgdqth'
test_sequence = [x for x in test_sequence]

# probabilities of going to end state
end_probs = [0.52, 0.48]
# probabilities of going from start state
start_probs = [0.52, 0.48]


# function to find forward probabilities
def forward_probs():
    # node values stored during forward algorithm
    node_values_fwd = np.zeros((len(states), len(test_sequence)))  # Create a probability matrix Foward[N,T]
    #  N = Number of states
    #  T = Length of Observation sequence.
    # Therefore, node_values_fwd creates a two (no. of states to go through for partial probabilities) by nine (Length of Observation) matrix
    for t, sequence_val in enumerate(test_sequence):  # t = time step; sequence_val = the value of the element.
        for s in range(len(states)):  # s = state
            # if first sequence value then do this
            if t == 0:
                """ Initialization: Alpha_0(j) = pi*beta_j(O_1)"""
                node_values_fwd[s, t] = start_probs[s] * emission[s, sequence_syms[sequence_val]]
                # Emission[s,sequence[sequnce_val]] = when in state 0 or 1 that is used to access into one of the
                # lists in emission list of lists, then after the comma the test sequence is converted into an
                # observation symbol which is another index notation which is then used to extract the element's
                # value in the inner list.
            else:
                """ Recursions: Summation of the three factors that are multiplied together:
                    (1) the previous forward path probability alpha_t-1(i)
                    (2) the transition probability from previous state q_i to current q_j
                    (3) the state observation likelihood of the observation symbol O_t given the current state j"""
                values = [node_values_fwd[k, t - 1] * emission[s, sequence_syms[sequence_val]] * transition[k, s] for k
                          in range(len(states))]
                """Termination with end states [add-on]: Summation of all the forward probabilities to determine P(O|model) 
        model = (A,B, Lambda) """
                node_values_fwd[s, t] = sum(values)  # Summation of the three factors and filling in the time step
                # with partial probabilities.

    # end state value

    end_state = np.multiply(node_values_fwd[:, -1], end_probs)
    end_state_val = sum(end_state)
    return node_values_fwd, end_state_val


# function to find backward probabilities
def backward_probs():
    node_values_bwd = np.zeros((len(states), len(test_sequence)))  # A path probability matrix viterbi[N,T]

    # for i, sequence_val in enumerate(test_sequence):
    for t in range(1, len(test_sequence) + 1):  # t = time step
        for s in range(len(states)):  # s = state ||| This is for each state 1 to N
            # if first sequence value then do this
            if -t == -1:
                node_values_bwd[s, -t] = end_probs[s]
            # else perform this
            else:
                """Rescursion: """
                values = [
                    node_values_bwd[k, -t + 1] * emission[k, sequence_syms[test_sequence[-t + 1]]] * transition[s, k]
                    for k in range(len(states))]
                node_values_bwd[s, -t] = sum(values)

    # start state value
    start_state = [node_values_bwd[m, 0] * emission[m, sequence_syms[test_sequence[0]]] for m in range(len(states))]
    start_state = np.multiply(start_state, start_probs)
    start_state_val = sum(start_state)
    return node_values_bwd, start_state_val


# function to find si probabilities
def si_probs(forward, backward, forward_val):
    si_probabilities = np.zeros((len(states), len(test_sequence) - 1, len(states)))

    for i in range(len(test_sequence) - 1):
        for j in range(len(states)):
            for k in range(len(states)):
                si_probabilities[j, i, k] = (forward[j, i] * backward[k, i + 1] * transition[j, k] * emission[
                    k, sequence_syms[test_sequence[i + 1]]]) \
                                            / forward_val
    return si_probabilities


# function to find gamma probabilities
def gamma_probs(forward, backward, forward_val):
    gamma_probabilities = np.zeros((len(states), len(test_sequence)))

    for i in range(len(test_sequence)):
        for j in range(len(states)):
            # gamma_probabilities[j,i] = ( forward[j,i] * backward[j,i] * emission[j,sequence_syms[test_sequence[i]]]
            # ) / forward_val
            gamma_probabilities[j, i] = (forward[j, i] * backward[j, i]) / forward_val

    return gamma_probabilities


# performing iterations until convergence

for iteration in range(2000):

    print('\nIteration No: ', iteration + 1)
    # print('\nTransition:\n ', transition)
    # print('\nEmission: \n', emission)

    # Calling probability functions to calculate all probabilities
    fwd_probs, fwd_val = forward_probs()
    bwd_probs, bwd_val = backward_probs()
    si_probabilities = si_probs(fwd_probs, bwd_probs, fwd_val)
    gamma_probabilities = gamma_probs(fwd_probs, bwd_probs, fwd_val)

    # print('Forward Probs:')
    # print(np.matrix(fwd_probs))
    #
    # print('Backward Probs:')
    # print(np.matrix(bwd_probs))
    #
    # print('Si Probs:')
    # print(si_probabilities)
    #
    # print('Gamma Probs:')
    # print(np.matrix(gamma_probabilities))

    # caclculating 'a' and 'b' matrices
    a = np.zeros((len(states), len(states)))
    b = np.zeros((len(states), len(sequence_syms)))

    # 'a' matrix
    for j in range(len(states)):
        for i in range(len(states)):
            for t in range(len(test_sequence) - 1):
                a[j, i] = a[j, i] + si_probabilities[j, t, i]

            denomenator_a = [si_probabilities[j, t_x, i_x] for t_x in range(len(test_sequence) - 1) for i_x in
                             range(len(states))]
            denomenator_a = sum(denomenator_a)

            if (denomenator_a == 0):
                a[j, i] = 0
            else:
                a[j, i] = a[j, i] / denomenator_a

    # 'b' matrix
    for j in range(len(states)):  # states
        for i in range(len(sequence)):  # seq
            indices = [idx for idx, val in enumerate(test_sequence) if val == sequence[i]]
            numerator_b = sum(gamma_probabilities[j, indices])
            denomenator_b = sum(gamma_probabilities[j, :])

            if (denomenator_b == 0):
                b[j, i] = 0
            else:
                b[j, i] = numerator_b / denomenator_b

    print('\nMatrix a:\n')
    print(np.matrix(a.round(decimals=4)))
    print('\nMatrix b:\n')
    print(np.matrix(b.round(decimals=4)))

    transition = a
    emission = b

    new_fwd_temp, new_fwd_temp_val = forward_probs()
    print('New forward probability: ', new_fwd_temp_val)
    diff = np.abs(fwd_val - new_fwd_temp_val)
    print('Difference in forward probability: ', diff)

    if diff < 0.0000001:
        break

c = 1
