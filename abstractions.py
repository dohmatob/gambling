"""
References
----------
[1] Bernhard von Stengel, "Efficient Computation of Behavior Strategies"

"""
# Author: DOHMATOB Elvis Dopgima

import re
import networkx as nx
import numpy as np
from nose.tools import assert_equal, assert_true, assert_false


class Game(object):
    player_choices = None

    def __init__(self):
        self.tree = nx.DiGraph()
        self.edge_labels = {}
        self.infosets = {}
        self.sequences = {}
        self.constraints = {}
        self.payoff_matrix = None
        self.build_tree()
        self.build_infosets()
        self.build_sequences()
        self.build_strategy_constraints()
        self.build_payoff_matrix()

    def build_tree(self):
        raise NotImplementedError

    def add_labelled_edge(self, x, choice, **kwargs):
        """Adds a labelled edge to the tree."""
        y = ".".join([x, choice])
        self.tree.add_node(y, **kwargs)
        self.tree.add_edge(x, y)
        self.edge_labels[(x, y)] = choice
        return y

    def is_leaf(self, node):
        """Checks whether given node is leaf / terminal."""
        return self.tree.out_degree(node) == 0.

    def project_onto_player(self, node, player):
        """Ignores all but the choices of a player, along given node."""
        if isinstance(node, str):
            node = node.split(".")
        return [x for x in node if x in self.player_choices[player]]

    def current_player(self, node):
        """Returns player to which node belongs."""
        if self.is_root(node):
            return 0
        node = node.split('.')
        if node[-1] not in self.player_choices[1]:
            return 1
        elif node[-1] not in self.player_choices[2]:
            return 2
        else:
            assert 0

    def is_root(self, node):
        return node == "/"

    def previous_player(self, node):
        """Player whose move leads to this node."""
        if self.is_root(node):
            return None
        node = node.split('.')
        for k, v in self.player_choices.iteritems():
            if node[-1] in v:
                return k
        else:
            assert 0

    def info_at_node(self, node):
        """Returns all the information available at a node."""
        player = self.current_player(node)
        sigma = self.project_onto_player(node, player)
        choices = sorted([succ.split('.')[-1]
                          for succ in self.tree.successors(node)])
        if player == 0:
            signal = None
        else:
            m = re.match('^/\.%s' % ("(.)" if player == 1 else ".(.)"),
                         node)
            assert m, "Node %s not in tree!" % node
            signal = m.group(1)
        return signal, tuple(sigma), tuple(choices)

    def level_first_traversal(self, root="/", player=None):
        _skip = lambda node: False if player is None else (
            self.previous_player(node) != player)
        if not _skip(root):
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
                        if not _skip(w):
                            yield w
                        next_level.add(w)
            cur_level = next_level

    def leafs_iter(self, data=True):
        if data:
            for node, data in self.tree.nodes_iter(data=True):
                if self.is_leaf(node):
                    yield node, data
        else:
            for node in self.tree.nodes_iter(data=False):
                if self.is_leaf(node):
                    yield node

    def build_infosets(self):
        """Generate information sets for each player."""
        self.infosets.clear()
        for node in self.level_first_traversal():
            player = self.current_player(node)
            if not player in self.infosets:
                self.infosets[player] = {}
            if self.is_leaf(node):
                continue
            info = self.info_at_node(node)
            if info not in self.infosets[player]:
                self.infosets[player][info] = [node]
            else:
                assert not node in self.infosets[player][info]
                self.infosets[player][info].append(node)
        return self.infosets

    def chop(self, node):
        """Return list of all nodes played along this path,
        by previous player."""
        if isinstance(node, str):
            node = node.split('.')
        prev = self.previous_player(".".join(node))
        return [node[:j] for j in xrange(1, len(node) + 1)
                if self.previous_player(".".join(node[:j])) == prev]

    def node2seq(self, node):
        """Returns sequence of information-set-relabelled moves made
        by previous player along this path."""
        if node is None:
            return []
        else:
            return [(self.info_at_node('.'.join(item[:-1])), item[-1])
                    for item in self.chop(node)]

    def last_node(self, node, player):
        """Last node played by given player, before this point."""
        if self.is_root(node):
            return None
        if self.previous_player(node) == player:
            return node
        else:
            return self.last_node(".".join(node.split('.')[:-1]), player)

    def build_sequences(self):
        """
        Each sequence for a player is of the form (i_1, a_1)(i_2,, a_2)...,
        wher each a_j is an action at the information set i_j
        """
        self.sequences.clear()
        for player in self.infosets.keys():
            self.sequences[player] = [[]]
        for node in self.level_first_traversal():
            if self.is_root(node):
                continue
            prev = self.previous_player(node)
            seq = self.node2seq(node)
            if seq not in self.sequences[prev]:
                self.sequences[prev].append(seq)
        for s in self.sequences.values():
            s = sorted(s, cmp=lambda a, b: len(a) - len(b)
                       if len(a) != len(b) else cmp(a, b))
        return self.sequences

    def build_strategy_constraints(self):
        """Generates matrices for the equality constraints on each player's
        admissible realization plans.

        The constraints for player i are a pair_i, e_i), representing,
        read as "E_ix=e_i". E_i has as many columns as player i has sequences,
        and as many rows as there information sets for player i, plus 1.
        """
        self.constraints.clear()
        for player in xrange(1, 3):
            row = np.zeros(len(self.sequences[player]))
            row[0] = 1.
            E = [row]
            for i, sigma in enumerate(self.sequences[player]):
                mem = {}
                for j, tau in enumerate(self.sequences[player]):
                    if tau and tau[:-1] == sigma:
                        h, _ = tau[-1]
                        if h not in mem:
                            mem[h] = []
                        mem[h].append(j)
                for where in mem.values():
                    row = np.zeros(len(self.sequences[player]))
                    row[i] = -1.
                    row[where] = 1.
                    E.append(row)
            e = np.zeros(len(self.infosets[player]) + 1)
            e[0] = 1.
            self.constraints[player] = np.array(E), e
        return self.constraints

    def build_payoff_matrix(self):
        """Builds payoff matrix from player 1's perspective.

        The rows (resp. columns) are labelled with player 1's (resp. 2's)
        sequences.
        """
        self.payoff_matrix = np.zeros((len(self.sequences[1]),
                                       len(self.sequences[2])))
        for leaf, data in self.leafs_iter(data=True):
            i = self.sequences[1].index(self.node2seq(self.last_node(leaf, 1)))
            j = self.sequences[2].index(self.node2seq(self.last_node(leaf, 2)))
            self.payoff_matrix[i, j] += data['payoff'] * data['proba']
        return self.payoff_matrix

    def draw(self):
        pos = nx.graphviz_layout(self.tree, prog='dot')

        # leaf (terminal) nodes
        nx.draw_networkx_nodes(self.tree, pos,
                               nodelist=[n for n in self.tree.nodes()
                                         if self.is_leaf(n)], node_shape='s')

        # decision nodes
        nx.draw_networkx_nodes(self.tree, pos,
                               nodelist=[n for n in self.tree.nodes()
                                         if not self.is_leaf(n)],
                               node_shape='o')

        # labelled edges
        nx.draw_networkx_edges(self.tree, pos, arrows=False)
        nx.draw_networkx_edge_labels(self.tree, pos,
                                     edge_labels=self.edge_labels)


class Kuhn3112(Game):
    """
    Proof-of-Concept for sequence-form representation (@ la BvS) poker.
    """

    player_choices = {0: ['12', '13', '21', '23', '31', '32'],
                      1: list('CFKR'), 2: list('cfkr')}

    def cmp_cards(self, a, b):
        return cmp(int(a), int(b))

    def build_tree(self):
        """Builds the tree (nx.DiGraph object)."""
        for perm, proba in zip(['12', '13', '21', '23', '31', '32'],
                               [1. / 6] * 6):
            self.add_labelled_edge('/', ''.join(perm),
                                   player=0)
            for a in 'CR':
                self.add_labelled_edge('/.%s' % ''.join(perm), a, player=1)
                if a == 'C':  # check
                    for b in 'cr':
                        child = self.add_labelled_edge(
                            '/.%s.%s' % (''.join(perm), a), b, player=2)
                        if b == "c":
                            self.tree.add_node(child, player=2, proba=proba,
                                               payoff=self.cmp_cards(*perm))
                        else:  # raise
                            for x in 'FK':
                                child = self.add_labelled_edge(
                                    '/.%s.%s.%s' % (''.join(perm), a, b),
                                    x, player=1)
                                if x == "F":  # fold
                                    self.tree.add_node(child, player=1,
                                                       payoff=-1, proba=proba)
                                else:  # call
                                    self.tree.add_node(
                                        child, player=1, proba=proba,
                                        payoff=2 * self.cmp_cards(*perm))
                else:  # raise
                    for b in "fk":
                        child = self.add_labelled_edge(
                            '/.%s.%s' % (''.join(perm), a), b, player=2)
                        if b == "f":  # fold
                            self.tree.add_node(child, player=2, payoff=1.,
                                               proba=proba)
                        else:  # call
                            self.tree.add_node(
                                child, player=2, proba=proba,
                                payoff=2 * self.cmp_cards(*perm))


class BvSFig21(Game):
    '''Example in Reference 1, fig 2.1.'''
    player_choices = {0: ['1', '2'], 1: list('lr'), 2: list('cd')}

    def build_tree(self):
        a = self.add_labelled_edge("/", '1', proba=1. / 3)
        proba = 1. / 3
        self.add_labelled_edge(a, "l", payoff=0., proba=proba)
        b = self.add_labelled_edge(a, "r")
        self.add_labelled_edge(b, 'c', payoff=3., proba=proba)
        self.add_labelled_edge(b, "d", payoff=-3., proba=proba)
        a = self.add_labelled_edge("/", '2')
        proba = 2. / 3
        b = self.add_labelled_edge(a, "l")
        self.add_labelled_edge(b, 'c', payoff=-3., proba=proba)
        self.add_labelled_edge(b, "d", payoff=6., proba=proba)
        self.add_labelled_edge(a, "r", payoff=3. / 2, proba=proba)


def test_build_tree():
    g = Kuhn3112()
    assert_equal(len(g.tree), 55)


def test_is_leaf():
    g = Kuhn3112()
    assert_true(g.is_leaf("/.32.R.f"))
    assert_true(g.is_leaf("/.32.C.r.K"))
    assert_false(g.is_leaf("/.32.R"))


def test_project_onto_player():
    g = Kuhn3112()
    assert_equal(g.project_onto_player("/.23.C.r", 1), ['C'])
    assert_equal(g.project_onto_player("/.23.C.r", 2), ['r'])


def test_current_player():
    g = Kuhn3112()
    assert_equal(g.current_player("/.31"), 1)


def test_info_at_node():
    g = Kuhn3112()
    card, sigma, choices = g.info_at_node("/.12.C.r.K")
    assert_equal(card, '2')
    assert_equal(sigma, ('r',))
    assert_equal(choices, ())


def test_level_first_traversal():
    g = Kuhn3112()
    nodes = list(g.level_first_traversal())
    assert_equal(nodes[0], "/")
    assert_equal(nodes[-1], "/.32.C.r.K")


def test_build_infosets():
    g = Kuhn3112()
    g.build_infosets()
    assert_equal(len(g.infosets[0]), 1)
    assert_equal(len(g.infosets[1]), 6)
    assert_equal(len(g.infosets[2]), 6)
    assert_equal(sorted([n for player in xrange(3)
                         for nodes in g.infosets[player].itervalues()
                         for n in nodes]),
                 sorted([n for n in g.tree.nodes() if not g.is_leaf(n)]))


def test_build_sequences():
    g = Kuhn3112()
    g.build_sequences()
    assert_equal(len(g.sequences[0]), 7)
    for player in xrange(1, 3):
        assert_equal(len(g.sequences[player]), 13, msg=len(g.sequences[1]))


def test_build_payoff_matrix():
    k = Kuhn3112()
    k.build_payoff_matrix()
    assert_equal((k.payoff_matrix != 0.).sum(), 30.)


def test_leafs_iter():
    k = Kuhn3112()
    assert_equal(len(list(k.leafs_iter())), 30)

if __name__ == "__main__":
    import pylab as pl
    from sequential_games import compute_ne
    if 1:
        game = Kuhn3112()
    else:
        game = BvSFig21()
    E, e = game.constraints[1]
    F, f = game.constraints[2]
    A = game.payoff_matrix
    x, y, values = compute_ne(A, E, F, e, f, tol=0, max_iter=100)
    print
    print "Nash Equilibrium:"
    print "x* = ", x
    print "y* =", y
    pl.semilogx(values)
    value = values[-1]
    pl.axhline(value, linestyle="--",
               label="value of the game: %5.2e" % value)
    pl.xlabel("k")
    pl.ylabel("value of game after k iterations")
    pl.legend(loc="best")
    pl.title("NE computation in sequence-form (game = %s)" % (
            game.__class__.__name__))
    pl.show()
