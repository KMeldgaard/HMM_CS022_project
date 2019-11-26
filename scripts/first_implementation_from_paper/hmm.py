""" HMM class
    hmm.py - 28-10-2019
    Author: Kasper Meldgaard
    #--#
    HMM class holds a hidden markov model (lambda) parameters.
    -N:  Number of states in model
    -M:  Number of different observations
    +A:  State transition matrix
    +B:  Observation probability matrix
    +pi: initial state
    """

DEBUG = "DEBUG:"

# package dependencies
import numpy as np
import logging

# DEBUG: on / off
A_debug = logging.getLogger("A_debug")
B_debug = logging.getLogger("B_debug")
pi_debug = logging.getLogger("pi_debug")
level = logging.DEBUG
logging.basicConfig(level=level)
# -- comment these in/out --#
A_debug.disabled = True
B_debug.disabled = True
pi_debug.disabled = True


class HMM:
    def __init__(self, N, M):
        self.__N__ = N
        self.__M__ = M
        self.A = np.array([self.__random_row__(N, A_debug) for i in range(N)])  # add N rows of length N
        A_debug.debug("Array A:\n%s", self.A)  # debug

        self.B = np.array([self.__random_row__(M, B_debug) for i in range(N)])
        B_debug.debug("Array B:\n%s", self.B)  # debug

        self.pi = np.array(self.__random_row__(N, pi_debug))
        pi_debug.debug("pi: %s", self.pi)

        # double-check the rows
        try:
            for i in range(N):
                for e in self.A[i]:
                    check_neg_pro(e)
                for e in self.B[i]:
                    check_neg_pro(e)
        except Neg_prob_Error as ex:
            print(ex.__DESCRIPTION__)
            exit(1)


    def __random_row__(self, l, row_debug):
        """Creates a row of l numbers with value normally distributed around 1/l"""
        while True:
            # TODO fix the distribution!!!
            # TODO np.random.uniform(low=, high=0, size=)
            # row = np.random.normal(loc=1 / l, scale=0.015, \
                                   # size=(l - 1))  # normally distribute values around 1/l except last value
            row = np.random.uniform(low=0.0, high= 2 / l, size=(l-1))
            row_debug.debug("Row before last elem: %s", row)  # debug
            if np.sum(row) > 1.0:
                continue
            for elem in row.tolist():
                # print(elem)
                if elem < 0:
                    # print("continue")
                    continue
            row = np.append(row, 1 - np.sum(row))  # add the last element and ensure row sums to 1 (stochastic row)
            row_debug.debug("Row after last elem: %s", row)  # debug
            row_debug.debug("Sum of row: %s", sum(row))  # debug
            if sum(row) == 1:
                break
        return row.tolist()  # a stochastic row of lenght l

    def get_N(self):
        return self.__N__

    def get_M(self):
        return self.__M__


""" Negative probability exception
    To ensure the model is initialized correctly"""
class Neg_prob_Error(Exception):
    __DESCRIPTION__ = "A negative value was encountered!"

""" Function to check if an elem is negative"""
def check_neg_pro(elem):
    if elem < 0.0:
        raise Neg_prob_Error
    return

# test class:
model = HMM(2, 27)
print("Model:\nA:\n", model.A, "\nB:\n", model.B, "\npi:\n", model.pi)