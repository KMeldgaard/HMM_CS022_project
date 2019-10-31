""" Class to hold an observation for training a Hidden markov model
    observation.py - 30-10-2019
    Author: Kasper Meldgaard
    #
    Observations are mapped to integers to represent them acording to a lookup table.
    obs: observation sequence mapped ti integers"""

import numpy as np


class latin_observations:
    def __init__(self, input_path=""):
        file = open(input_path)
        i_str = file.read().replace("\n", ' ')
        # print(i_str)    # debud
        i_str = i_str.lower()
        # print(i_str)    # debug
        # map input characters to integers
        self.obs = np.array(list(map(lambda i: (ord(i) - 97) if ord(i) in list(range(97, 123)) else 27, i_str)))
        self.num_obs = len(self.obs)

    def get_num_obs(self):
        return self.num_obs

    def get_clear_text(self):
        # TODO
        pass

    # def __map_input__(self, input):
    #     look_up = {'a': 0, 'b': 1, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, \
    #                'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
    #                'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x':24, 'y': 25, 'z': 26, ' ': 27}
    #     if input in look_up.keys():
    #         return look_up[input]
    #     else:   # a special character like ':' or '.'
    #         return " "


o = latin_observations("test_text.txt")
print(o.obs)
