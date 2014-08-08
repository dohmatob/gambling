import itertools
import re
import networkx as nx
from nose.tools import assert_true, assert_equal, assert_false
import numpy as np


class Game(object):
    """Implementation of KUHN(3, 1, 1, 2) 1-card poker."""

    def __init__(self):
        self.tree = nx.DiGraph()
        self.edge_labels = {}
        self.infosets = {}
        self.sequences = {}
        self.constraints = {}
        self.player_choices = {0: list(itertools.permutations('123', 2)),
                               1: 'CFKR', 2: 'cfkr'}
        self.build_tree()
        self.build_infosets()

    def _add_labelled_edge(self, x, y, move, **kwargs):
        """Adds a labelled edge to the tree."""
        self.tree.add_node(y, **kwargs)
        self.tree.add_edge(x, y)
        self.edge_labels[(x, y)] = move

    def build_tree(self):
        """Builds the tree (nx.DiGraph object)."""
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

    def is_leaf(self, node):
        """Checks whether given node is leaf / terminal."""
        return self.tree.out_degree(node) == 0.

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

    def project_onto_player(self, node, player):
        """Ignores all but the choices of a player, along given node."""
        if isinstance(node, str):
            node = node.split(".")
        return [x for x in node if x in self.player_choices[player]]

    def player_to_play(self, node):
        """Returns player to which node belongs."""
        if node == "/":
            return 0
        elif node[-1] in '123cfkr':
            return 1
        elif node[-1] in 'CFKR':
            return 2
        else:
            assert 0

    def info_at_node(self, node):
        """Returns all the information available at a node."""
        player = self.player_to_play(node)
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

    def build_infosets(self):
        """Generate information sets for each player."""
        self.infosets.clear()
        for node in self.level_first_traversal():
            player = self.player_to_play(node)
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

    def project_onto_player_advanced(self, node, player):
        if isinstance(node, str):
            node = node.split(".")
        train = [node[:i + 1] for i in xrange(len(node)) if
                 node[i] in self.player_choices[player]]
        return [(self.info_at_node(".".join(x[:-1])), x[-1])
                for x in train]

    def build_sequences(self):
        """
        Each sequence for a player is of the form (i_1, a_1)(i_2,, a_2)...,
        wher each a_j is an action at the information set i_j
        """
        self.sequences.clear()
        for player in xrange(3):
            self.sequences[player] = [tuple()]
            self.sequences[player] += [(i, c) for i in self.infosets[player]
                                       for c in i[-1]]
        return self.sequences

    def build_strategy_constraints(self):
        """Builds constraint matrices for the realization plans of each player.
        """
        # XXX Each E and e should a scipy.sparse.csr_matrix object!!!
        self.constraints.clear()
        for player in xrange(1, 3):
            tmp = []
            E = np.zeros((len(self.infosets[player]) + 1,
                          len(self.sequences[player])))
            E[0, 0] = 1.
            e = np.zeros(E.shape[0])
            e[0] = 1.
            for x, q in enumerate(self.sequences[1]):
                if not q:
                    continue
                i, _ = q
                ii = self.infosets[player].keys().index(i)
                assert not (q, ii) in tmp
                tmp.append((q, ii))
                if not ii:
                    continue
                assert ii, q
                E[ii, x] = -1
                h = self.infosets[1][i][0]
                proj = self.project_onto_player_advanced(h, 1)
                for y, q_ in enumerate(self.sequences[1]):
                    if not q_ or q_ == q:
                        continue
                    i_, _ = q_
                    h_ = self.infosets[1][i_][0]
                    proj_ = self.project_onto_player_advanced(h_, 1)
                    if proj_:
                        E[ii, y] = 1
                    elif len(proj_) > len(proj) and proj_[:-1] == proj:
                        E[ii, tmp] = 1.
            assert 0
        for player in xrange(1, 3):
            E = np.zeros((len(self.infosets[player]) + 1,
                          len(self.sequences[player])))
            e = np.zeros(E.shape[0])
            e[0] = 1.
            E[0, 0] = 1.
            for h, stuff in enumerate(self.infosets[player].keys()):
                sigma_h = stuff['sigma']
                c_h = stuff["relabelled_moves"].values()
                E[h + 1, self.sequences[player].index(sigma_h)] = -1
                for c in c_h:
                    E[h + 1, self.sequences[player].index(sigma_h + [c])] = 1
            self.constraints[player] = E, e
        return self.constraints


def test_build_tree():
    g = Game()
    assert_equal(len(g.tree), 55)


def test_is_leaf():
    g = Game()
    assert_true(g.is_leaf("/.32.R.f"))
    assert_true(g.is_leaf("/.32.C.r.K"))
    assert_false(g.is_leaf("/.32.R"))


def test_project_onto_player():
    g = Game()
    assert_equal(g.project_onto_player("/.23.C.r", 1), ['C'])
    assert_equal(g.project_onto_player("/.23.C.r", 2), ['r'])


def test_player_to_play():
    g = Game().build_tree()
    assert_equal(g.player_to_play("/.31"), 1)


def test_info_at_node():
    g = Game()
    card, sigma, choices = g.info_at_node("/.12.C.r.K")
    assert_equal(card, '2')
    assert_equal(sigma, ('r',))
    assert_equal(choices, ())


def test_level_first_traversal():
    g = Game()
    nodes = list(g.level_first_traversal())
    assert_equal(nodes[0], "/")
    assert_equal(nodes[-1], "/.32.C.r.K")


def test_build_infosets():
    g = Game()
    g.build_infosets()
    assert_equal(len(g.infosets[0]), 1)
    assert_equal(len(g.infosets[1]), 6)
    assert_equal(len(g.infosets[2]), 6)
    assert_equal(sorted([n for player in xrange(3)
                         for nodes in g.infosets[player].itervalues()
                         for n in nodes]),
                 sorted([n for n in g.tree.nodes() if not g.is_leaf(n)]))


def test_build_sequences():
    g = Game()
    g.build_sequences()
    assert_equal(len(g.sequences[0]), 6)
    for player in xrange(1, 3):
        assert_equal(len(g.sequences[player]), 12, msg=len(g.sequences[1]))


if __name__ == "__main__":
    import pylab as pl
    g = Game()
    g.build_tree().draw()
    print g.build_sequences()
    g.build_strategy_constraints()
    pl.show()
