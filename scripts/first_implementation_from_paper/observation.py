""" Class to hold an observation for training a Hidden markov model
    observation.py - 30-10-2019
    Author: Kasper Meldgaard
    #
    Observations are mapped to integers to represent them acording to a lookup table.
    obs: observation sequence mapped ti integers"""

import numpy as np
import string

class Latin_observations:
    def __init__(self, input_path=""):
        try:
            file = open(input_path)
        except IOError:
            print("Input file wrong!")
            exit(1)

        i_str = file.read().replace("\n", ' ').replace(",", '').replace("(", '').replace(")", '').replace("/", '')
        i_str = i_str.strip(string.punctuation)
        i_str = i_str.strip(string.digits)
        # print(i_str)    # debug
        i_str = i_str.lower()
        # print(i_str)    # debug
        # map input characters to integers
        self.obs = np.array(list(map(lambda i: (ord(i) - ord('a')) if ord(i) in list(range(ord('a'), \
                                                                                           (ord('z') + 1))) else 26,
                                     i_str)))
        self.num_obs = len(self.obs)

    def get_num_obs(self):
        return self.num_obs

    def __len__(self):  # same as get_num_obs()
        return len(self.obs)

    def get_clear_text(self):
        # TODO
        pass


# test
o = Latin_observations("guide_to_wireless_intro.txt")
# print(o.obs)
