# Author: DOHMATOB Elvis

import re
from nose.tools import assert_equal
import networkx as nx


class Game(object):
    PLAYER_CHOICES = None

    def __init__(self):
        self.tree = nx.DiGraph()
        self.edge_labels = {}
        self.infosets = {}
        self.sequences = {}
        self.constraints = {}
        self.init_misc()
        self.build_tree()
        self.build_sequences()

    def init_misc(self):
        self.last_node_played_patterns = {}
        for player, choices in self.PLAYER_CHOICES.iteritems():
            choices = "|".join(choices)
            self.last_node_played_patterns[player] = re.compile(
                '^(.*)(\([%s],\d\))[^%s]*$' % (choices, choices))

    def is_leaf(self, node):
        """Checks whether given node is leaf / terminal."""
        return not self.tree.node[node]["info"]["choices"]

    def is_root(self, node):
        return node == "(/,0)"

    def player_to_play(self, node):
        """Returns player to play at given node."""
        return int(re.match("^.*?(\d)\\)$", node).group(1))

    def previous_player(self, node):
        pred = self.tree.pred[node]
        return self.player_to_play(pred.keys()[0]) if pred else None

    def last_node_played(self, node, player=None):
        """Returns last node played by player to play at given node."""
        if player is None:
            player = self.player_to_play(node)
        hit = self.last_node_played_patterns[player].match(node)
        return None if hit is None else hit.groups()

    def project_onto_player(self, node, player=None):
        """Returns the encoding of the given node, only using characters
        from the player to play at the node.
        """
        last = self.last_node_played(node, player=player)
        return "" if last is None else self.project_onto_player(
            last[0], player=player) + last[1]

    def blur(self, chance_choice, player):
        """Returns the part of the given chance choice only observable
        by the given player.

        Method should be implemented by subclasses.
        """
        raise NotImplementedError

    def compute_info_at_node(self, node, choices):
        """Returns info at given node.

        Nodes with the same info belong to the same "information set".

        Returns
        -------
        x : list of char
            Sequence of chance choices observed upto this node.

        y : string
            Sequence of choices made from root, to this node, made by player
            to play at this node.
        z : list of char
            Choices available at this node.
        """
        player = self.player_to_play(node)
        chance = self.project_onto_player(node, player=0)
        chance = [self.blur(x[1], player)
                  for x in re.split(",\d\)", chance)[:-1]]
        mine = self.project_onto_player(node, player=player)
        return tuple(chance), mine, tuple(choices)

    def build_tree(self):
        raise NotImplementedError

    def store_node_info(self, player, node, choices):
        if not node in self.tree.node:
            self.tree.add_node(node)
        info = self.compute_info_at_node(node, choices)
        if not player in self.infosets:
            self.infosets[player] = {}
        if not info in self.infosets:
            self.infosets[player][info] = []
        self.infosets[player][info].append(node)
        self.tree.node[node]["info"] = dict(chance=info[0], sigma=info[1],
                                            choices=info[2])

    def add_labelled_edge(self, src, move, choices=[], **kwargs):
        """Adds a labelled edge to the tree.

        Parameters
        ----------
        src : string
            Source node.

        move : pair (choice, next_player)
            `choice` is the choice made by player at `src` and `next_player`
            is the next player to play.

        choices : None
            Choices available at this node.

        kwargs : value-pair dict-like
            Additional data to store at destination node.
        """
        choice, player = move
        dst = "%s.(%s,%s)" % (src, move[0], move[1])
        next_player = self.player_to_play(dst)
        assert_equal(player, next_player)
        self.tree.add_node(dst, player=player, **kwargs)
        self.tree.add_edge(src, dst)
        self.edge_labels[(src, dst)] = choice
        self.store_node_info(player, dst, choices)
        return dst

    def level_first_traversal(self, player=None):
        root = "(/,0)"
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

    def chop(self, node):
        """Return list of all nodes played along this path,
        by previous player."""
        prev = self.previous_player(node)
        pieces = node.split(".")
        for j in xrange(len(pieces)):
            child = pieces[:j + 1]
            if self.previous_player(".".join(child)) == prev:
                yield child

    def node2seq(self, node):
        """Returns sequence of information-set-relabelled moves made
        by previous player along this path."""
        if node is None:
            return []
        else:
            return [(self.tree.node['.'.join(item[:-1])]["info"], item[-1])
                    for item in self.chop(node)]

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
    PLAYER_CHOICES = {0: ['u', 'v', 'w', 'x', 'y', 'z'],
                      1: list('CFKR'),
                      2: list('cfkr')}
    BOOK = ['12', '13', '21', '23', '31', '32']

    def chance_word_to_char(self, word):
        return self.PLAYER_CHOICES[0][self.BOOK.index(word)]

    def chance_char_to_word(self, char):
        return self.BOOK[self.PLAYER_CHOICES[0].index(char)]

    def blur(self, choice, player):
        return self.chance_char_to_word(choice)[player - 1]

    def cmp_cards(self, a, b):
        return cmp(int(a), int(b))

    def build_tree(self):
        """Builds the tree (nx.DiGraph object)."""
        self.store_node_info(0, "(/,0)", self.PLAYER_CHOICES[0])
        for perm, proba in zip(self.PLAYER_CHOICES[0], [1. / 6] * 6):
            perm_word = self.chance_char_to_word(perm)
            x = self.add_labelled_edge("(/,0)", (perm, 1), choices=["C", "R"])

            for a in 'CR':
                if a == 'C':  # check
                    y = self.add_labelled_edge(x, (a, 2), choices=["r", "c"])
                    for b in 'cr':
                        if b == "c":  # check
                            self.add_labelled_edge(
                                y, (b, 1), proba=proba, payoff=self.cmp_cards(
                                    *perm_word))
                        else:  # raise
                            z = self.add_labelled_edge(
                                y, (b, 1), choices=['F', 'K'])
                            for w in 'FK':
                                if w == "F":  # fold
                                    self.add_labelled_edge(
                                        z, (w, 2), proba=proba,
                                        payoff=self.cmp_cards(*perm_word))
                                else:  # call
                                    self.add_labelled_edge(
                                        z, (w, 2), proba=proba,
                                        payoff=2 * self.cmp_cards(*perm_word))
                else:  # raise
                    y = self.add_labelled_edge(x, (a, 2), choices=["f", "k"])
                    for b in "fk":
                        if b == "f":  # fold
                            self.add_labelled_edge(
                                y, (b, 1), proba=proba, payoff=1.)
                        else:  # call
                            self.add_labelled_edge(
                                y, (b, 1), payoff=2 * self.cmp_cards(
                                    *perm_word))


def test_player_to_play():
    g = Kuhn3112()
    assert_equal(g.player_to_play("(/,0)"), 0)
    assert_equal(g.player_to_play("(/,0)(a,1)"), 1)
    assert_equal(g.player_to_play("(/,0)(e,2)(k, 0)(b, 2)"), 2)


def test_blur():
    g = Kuhn3112()
    assert_equal(g.blur("v", 2), "3")
    assert_equal(g.blur("v", 1), "1")


def test_chance_word_to_char():
    g = Kuhn3112()
    assert_equal(g.chance_word_to_char('31'), 'y')


def test_chance_char_to_word():
    g = Kuhn3112()
    assert_equal(g.chance_char_to_word("u"), "12")


def test_last_node_played():
    g = Kuhn3112()
    node = "(/,0)(u,1)(C,0)(v,1)(R,2)(k,1)"
    last, choice = g.last_node_played(node)
    assert_equal(last, '(/,0)(u,1)(C,0)(v,1)')
    assert_equal(choice, '(R,2)')


def test_project_onto_player():
    g = Kuhn3112()
    node = "(/,0)(u,1)(C,0)(v,1)(R,2)(k,1)"
    assert_equal(g.project_onto_player(node), '(C,0)(R,2)')


def test_compute_info_at_node():
    g = Kuhn3112()
    node = "(/,0)(u,1)(C,0)(y,1)(R,2)(k,1)"
    x, y, z = g.compute_info_at_node(node, ['F'])
    assert_equal(x, ('1', '3'))
    assert_equal(y, '(C,0)(R,2)')
    assert_equal(z, ('F',))
