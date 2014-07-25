"""
Sequential form of games (in the sense of Benhard Von Stengel et al.).
"""

# Author: DOHMATOB Elvis <gmdopp@gmail.com>

import numpy as np


class Player(object):
    def __init__(self, seqs, info):
        self.seqs = seqs
        self.info = info
        self.E = None
        self.e = None

    def compute_constraints(self):
        """
        Compute constraints on admissible behavioural strategies.

        Theses contraints are of the form (E, e), where E is an i-by-s
        matrix, being 1 + the number of information sets (h) for the
        player, and s being the number of sequences (sigma) playable by
        the player.

        Parameters
        ----------
        seqs: list of lists
           Set of sequences playable by the player.

        info: list of tuples
           Each entry is a pair (sigma_h, c_h), where:
               h is the information set
               sigma_h: is the (unique) sequence leading to the information set
                        from the root node
               c_h: the "moves at h"

        Returns
        -------
        E: 2D array
        e: 1D array

        """

        self.E = np.zeros((len(self.info) + 1, len(self.seqs)))
        self.e = np.zeros(self.E.shape[0])
        self.e[0] = 1
        self.E[0, 0] = 1
        for h, (sigma_h, c_h) in enumerate(self.info):
            self.E[h + 1, self.seqs.index(sigma_h)] = -1
            for c in c_h:
                self.E[h + 1, self.seqs.index(sigma_h + [c])] = 1

        return self.E, self.e


if __name__ == "__main__":
    # player 1
    player1 = Player([[], ['L'], ['R'], ['L', 'S'], ['L', 'T']],
                     [([], ['L', 'R']), (['L'], ['S', 'T'])])
    player1.compute_constraints()
    print "E =", player1.E
    print "e =", player1.e
    print

    # player 2
    player2 = Player([[], ['l'], ['r']], [([], ['l', 'r'])])
    player2.compute_constraints()
    print "E =", player2.E
    print "e =", player2.e
    print
