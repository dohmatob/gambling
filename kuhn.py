import re
import networkx as nx
import numpy as np
from nose.tools import assert_equal, assert_true, assert_false

CARD_RANDKINS = {'3': 1, '2': 2, '1': 3}


class Poker(object):
    def __init__(self):
        self.player_choices = {0: ['12', '13', '21', '23', '31', '32'],
                               1: list('CFKR'), 2: list('cfkr')}
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

    def _add_labelled_edge(self, x, y, move, **kwargs):
        """Adds a labelled edge to the tree."""
        self.tree.add_node(y, **kwargs)
        self.tree.add_edge(x, y)
        self.edge_labels[(x, y)] = move

    def cmp_cards(self, a, b):
        return cmp(CARD_RANDKINS[b], CARD_RANDKINS[a])

    def build_tree(self):
        """Builds the tree (nx.DiGraph object)."""
        for perm, proba in zip(self.player_choices[0],
                               [1. / 6] * 6):
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
                            self.tree.add_node(dst, player=2, proba=proba,
                                               payoff=self.cmp_cards(*perm))
                        else:  # raise
                            for x in 'FK':
                                dst = '/.%s.%s.%s.%s' % (
                                    ''.join(perm), a, b, x)
                                self._add_labelled_edge(
                                    '/.%s.%s.%s' % (''.join(perm), a, b),
                                    dst, x, player=1)
                                if x == "F":  # fold
                                    self.tree.add_node(dst, player=1,
                                                       payoff=-1, proba=proba)
                                else:  # call
                                    self.tree.add_node(
                                        dst, player=1, proba=proba,
                                        payoff=2 * self.cmp_cards(*perm))
                else:  # raise
                    for b in "fk":
                        dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                        self._add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                           dst, b, player=2)
                        if b == "f":  # fold
                            self.tree.add_node(
                                dst, player=2, payoff=1., proba=proba)
                        else:  # call
                            self.tree.add_node(
                                dst, player=2, proba=proba,
                                payoff=2 * self.cmp_cards(*perm))

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

    def project_onto_player_advanced(self, node, player):
        if isinstance(node, str):
            node = node.split(".")
        train = [node[:i + 1] for i in xrange(len(node)) if
                 node[i] in self.player_choices[player]]
        return [(self.info_at_node(".".join(x[:-1])), x[-1])
                for x in train]

    def chop(self, node):
        if isinstance(node, str):
            node = node.split('.')
        prev = self.previous_player(".".join(node))
        return [node[:j] for j in xrange(1, len(node) + 1)
                if self.previous_player(".".join(node[:j])) == prev]

    def node2seq(self, node):
        return [(self.info_at_node('.'.join(item[:-1])), item[-1])
                for item in self.chop(node)]

    def last_node(self, node, player):
        """Last node played given player, before this point."""
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
        # for s in self.sequences.values():
        #     s.sort()
        return self.sequences

    def build_strategy_constraints(self):
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


def test_build_tree():
    g = Poker()
    assert_equal(len(g.tree), 55)


def test_is_leaf():
    g = Poker()
    assert_true(g.is_leaf("/.32.R.f"))
    assert_true(g.is_leaf("/.32.C.r.K"))
    assert_false(g.is_leaf("/.32.R"))


def test_project_onto_player():
    g = Poker()
    assert_equal(g.project_onto_player("/.23.C.r", 1), ['C'])
    assert_equal(g.project_onto_player("/.23.C.r", 2), ['r'])


def test_current_player():
    g = Poker()
    assert_equal(g.current_player("/.31"), 1)


def test_info_at_node():
    g = Poker()
    card, sigma, choices = g.info_at_node("/.12.C.r.K")
    assert_equal(card, '2')
    assert_equal(sigma, ('r',))
    assert_equal(choices, ())


def test_level_first_traversal():
    g = Poker()
    nodes = list(g.level_first_traversal())
    assert_equal(nodes[0], "/")
    assert_equal(nodes[-1], "/.32.C.r.K")


def test_build_infosets():
    g = Poker()
    g.build_infosets()
    assert_equal(len(g.infosets[0]), 1)
    assert_equal(len(g.infosets[1]), 6)
    assert_equal(len(g.infosets[2]), 6)
    assert_equal(sorted([n for player in xrange(3)
                         for nodes in g.infosets[player].itervalues()
                         for n in nodes]),
                 sorted([n for n in g.tree.nodes() if not g.is_leaf(n)]))


def test_build_sequences():
    g = Poker()
    g.build_sequences()
    assert_equal(len(g.sequences[0]), 7)
    for player in xrange(1, 3):
        assert_equal(len(g.sequences[player]), 13, msg=len(g.sequences[1]))

if __name__ == "__main__":
    import pylab as pl
    from sequential_games import compute_ne
    kuhn = Poker()
    E, e = kuhn.constraints[1]
    F, f = kuhn.constraints[2]
    A = kuhn.payoff_matrix
    x, y, values, _ = compute_ne(A, E, F, e, f, tol=0, max_iter=100)
    print
    print "Nash Equilibrium:"
    print "x* = ", x
    print "y* =", y
    pl.semilogx(values)
    pl.axhline(values[-1], linestyle="--", label="value of game")
    pl.xlabel("k")
    pl.ylabel("value of game after k iterations")
    pl.legend(loc="best")
    pl.title("NE computation in sequence-form Kuhn poker")
    pl.show()
