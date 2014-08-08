# Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

import itertools
import re
from nose.tools import assert_true, assert_equal
import numpy as np
import pylab as pl
import networkx as nx


class Kurn3112GameTree(object):
    """Implementation of kuhn(3, 1, 1, 2) poker game tree."""

    def __init__(self):
        self.tree = nx.DiGraph()
        self.edge_labels = {}

        # tree
        self.build_tree()
        self.tree_data = dict(self.tree.nodes(data=True))

        # player moves
        self.moves = {}
        for player in xrange(3):
            self.moves[player] = self.player_moves(player)

        # information sets
        self.info = {}
        self.build_info_sets()

        # player sequences
        self.sequences = {}
        for player in xrange(3):
            self.sequences[player] = self.player_sequences(player)

        # constraints on realization plan space for each player
        self.constraints = {}
        self.build_strategy_constraints()

        # payoff matrix
        # XXX A should be a scipy.sparse.csr_matrix!!!
        self.A = np.zeros((len(self.sequences[1]), len(self.sequences[2])))
        self.payoff_matrix()

    def _add_labelled_edge(self, x, y, move, **kwargs):
        """Adds a labelled edge to the tree."""
        self.tree.add_node(y, **kwargs)
        self.tree.add_edge(x, y)
        self.edge_labels[(x, y)] = move

    def is_leaf(self, node):
        """Checks whether given node is leaf / terminal."""
        return self.tree.out_degree(node) == 0.

    def player_moves(self, player):
        """The possible moves of given player in the game tree."""
        if player == 0.:
            return map(lambda x: ''.join(x),
                       list(itertools.permutations('123', 2)))
        else:
            moves = 'crfk'
            if player == 1.:
                moves = moves.upper()
            return sorted(moves)

    def build_tree(self):
        """Builds the tree (nx.DiGraph object)."""
        self.tree = nx.DiGraph()
        self.tree.add_node('/', player=None)
        self.edge_labels = {}

        # build tree
        proba = "1/6"
        for perm in itertools.permutations('123', 2):
            self._add_labelled_edge('/', '/.%s' % ''.join(perm), proba,
                                    player=0, proba=eval("1. * %s" % proba))
            for a in 'CR':
                self._add_labelled_edge('/.%s' % ''.join(perm), '/.%s.%s' % (
                        ''.join(perm), a), a, player=1)
                if a == 'C':  # check
                    for b in 'cr':
                        dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                        self._add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                           dst, b, player=2)
                        if b == "c":
                            self.tree.add_node(dst, player=2,
                                               payoff=cmp(*perm))
                        else:  # raise
                            for x in 'FK':
                                dst = '/.%s.%s.%s.%s' % (
                                    ''.join(perm), a, b, x)
                                self._add_labelled_edge(
                                    '/.%s.%s.%s' % (''.join(perm), a, b),
                                    dst, x, player=1)
                                if x == "F":  # fold
                                    self.tree.add_node(dst, player=1,
                                                       payoff=-1)
                                else:  # call
                                    self.tree.add_node(dst, player=1,
                                               payoff=2 * cmp(*perm))
                else:  # raise
                    for b in "fk":
                        dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                        self._add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                           dst, b, player=2)
                        if b == "f":  # fold
                            self.tree.add_node(dst, player=2, payoff=1.)
                        else:  # call
                            self.tree.add_node(dst, player=2,
                                               payoff=2 * cmp(*perm))

        return self

    def level_first_traversal(self, root="/"):
        yield root
        visited = set()
        cur_level = [root]
        while cur_level:
            for v in cur_level:
                visited.add(v)
            next_level = set()
            for v in cur_level:
                for w in self.tree[v]:
                    if w not in visited:
                        yield w
                        next_level.add(w)
            cur_level = next_level

    def project_seq_onto_player(self, seq, player):
        """Ignores all but the moves of a player, along given sequence."""
        if isinstance(seq, str):
            seq = seq.split(".")
        return [x for x in seq if x in self.moves[player]]

    def all_subsequences(self, seq):
        player = self.owner(seq)
        if isinstance(seq, str):
            seq = seq.split(".")
        return [".".join(seq[:i + 1]) for i in xrange(len(seq)) if
                self.owner(".".join(seq[:i + 1])) == player]

    def player_sequences(self, player):
        """The possible sequences of given player in the game tree."""
        seqs = [[]]
        for value in self.info[player].values():
            sigma_h = value['sigma']
            c_h = value['relabelled_moves'].values()
            for c in c_h:
                seq = list(sigma_h) + [c]
                assert not seq in seqs
                seqs.append(seq)
        return sorted(seqs)

    def owner(self, node):
        """Returns player to which node belongs."""
        if node == "/":
            return 0
        elif node[-1] in '123cfkr':
            return 1
        elif node[-1] in 'CFKR':
            return 2
        else:
            assert 0

    def node_knowledge(self, node):
        """Returns all the information available at a node."""
        if node == "/":
            return None, (), tuple(self.moves[0])
        player = self.owner(node)
        signal = re.match('^/\.%s' % ("(.)" if player == 1 else ".(.)"),
                          node).group(1)
        sigma = self.project_seq_onto_player(
            node, self.owner(node))
        moves = sorted([succ.split('.')[-1]
                        for succ in self.tree.successors(node)])
        return signal, tuple(sigma), tuple(moves)

    def prev_node_played(self, sigma):
        player = self.owner(sigma)
        if isinstance(sigma, str):
            sigma = sigma.split(".")
        if len(sigma) < 2:
            return None
        for i in xrange(len(sigma) - 2, -1, -1):
            sub = ".".join(sigma[:i + 1])
            move = sigma[i + 1]
            if self.owner(sub) == player:
                return sub, move
        else:
            return None

    def sigma_to_node(self, node):
        prev  = self.prev_node_played(node)
        if prev is None:
            return []
        prev, move = prev
        player = self.owner(node)
        assert self.owner(prev) == player
        key = self.node_knowledge(prev)
        assert key in self.info[player]
        infoset = self.info[player][key]
        return infoset['sigma'] + [infoset['relabelled_moves'][move]]

    def build_info_sets(self):
        """Generate information sets for each player."""
        self.info = {0: {}, 1: {}, 2: {}}
        for node in self.level_first_traversal():
            player = self.owner(node)
            if self.is_leaf(node):
                continue
            key = h, sigma, moves = self.node_knowledge(node)
            if player == 0.:
                moves = sorted([succ.split('.')[-1]
                                for succ in self.tree.successors(node)])
            else:
                sigma = self.sigma_to_node(node)
            if key not in self.info[player]:
                # h = len(self.info[player])
                moves = dict(zip(moves, [(move, h) for move in moves])
                             )  # relabelled moves
                self.info[player][key] = dict(nodes=[node], sigma=sigma,
                                              h=h, relabelled_moves=moves)
            else:
                assert not node in self.info[player][key]["nodes"]
                self.info[player][key]["nodes"].append(node)
        return self.info

    def build_strategy_constraints(self):
        """Builds constraint matrices for the realization plans of each player.
        """
        # XXX Each E and e should a scipy.sparse.csr_matrix object!!!
        for player in xrange(1, 3):
            E = np.zeros((len(self.info[player]) + 1,
                          len(self.sequences[player])))
            e = np.zeros(E.shape[0])
            e[0] = 1.
            E[0, 0] = 1.
            for h, stuff in enumerate(self.info[player].itervalues()):
                sigma_h = stuff['sigma']
                c_h = stuff["relabelled_moves"].values()
                E[h + 1, self.sequences[player].index(sigma_h)] = -1
                for c in c_h:
                    E[h + 1, self.sequences[player].index(sigma_h + [c])] = 1
            self.constraints[player] = E, e
        return self.constraints

    def payoff_matrix(self):
        """Computes payoff matrix for two-person zero-sum sequential game."""
        data_lookup = dict(self.tree.nodes(data=True))
        for leaf, data in [(k, v) for k, v in data_lookup.iteritems()
                           if self.is_leaf(k)]:
            x, y = list(leaf.split(".")[1])
            tau = [(c, x)
                   for c in self.project_seq_onto_player(leaf, 1)]
            sigma = [(c, y)
                   for c in self.project_seq_onto_player(leaf, 2)]
            luck = np.prod([data_lookup[n]['proba']
                            for n, d in data_lookup.iteritems()
                            if 'proba' in d and leaf.startswith(n)])
            self.A[self.sequences[1].index(tau),
                   self.sequences[2].index(sigma)] += data['payoff'] * luck
        return self.A


def test_build_information_sets():
    k = Kurn3112GameTree()
    assert_equal(sorted([n for player in xrange(3)
                         for v in k.info[player].itervalues()
                         for n in v['nodes']]),
                 sorted([n for n in k.tree.nodes() if not k.is_leaf(n)]))


def test_player_sequences():
    k = Kurn3112GameTree()
    for player in xrange(3):
        for x in k.info[player].itervalues():
            sigma_h = x['sigma']
            assert_true(list(sigma_h) in k.sequences[player],
                        msg=(player, sigma_h, k.sequences[player]))
            c_h = x['moves']

if __name__ == "__main__":
    kuhn = Kurn3112GameTree()
    pos = nx.graphviz_layout(kuhn.tree, prog='dot')

    # leaf (terminal) nodes
    nx.draw_networkx_nodes(kuhn.tree, pos,
                           nodelist=[n for n in kuhn.tree.nodes()
                                     if kuhn.is_leaf(n)], node_shape='s')

    # decision nodes
    nx.draw_networkx_nodes(kuhn.tree, pos,
                           nodelist=[n for n in kuhn.tree.nodes()
                                     if not kuhn.is_leaf(n)], node_shape='o')

    # labelled edges
    nx.draw_networkx_edges(kuhn.tree, pos, arrows=False)
    nx.draw_networkx_edge_labels(kuhn.tree, pos, edge_labels=kuhn.edge_labels)
    pl.show()


def level_first_traversal(root="/"):
    visited = set()
    cur_level = [root]
    while cur_level:
        for v in cur_level:
            visited.add(v)
        next_level = set()
        # levelGraph = {v: set() for v in cur_level}
        yield v
        for v in cur_level:
            for w in G[v]:
                if w not in visited:
                    # levelGraph[v].add(w)
                    yield w
                    next_level.add(w)
        # for me, children in levelGraph.iteritems():
        #     yield me
        #     for child in children:
        #         yield child
        # yield levelGraph
        cur_level = next_level
