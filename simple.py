# Author: DOHMATOB Elvis

import re
from nose.tools import assert_equal, assert_true
import networkx as nx
import numpy as np


class Game(object):
    """Sequence-form representaion of two-person zero-sum games with perfect
    recall and imcomplete information (like Poker, love, war, diplomacy, etc.)

    Terminology
    -----------

    Game Tree (normal-form definition)
    ==================================
    There are 3 players, namely: the chance player (player 0, aka nature),
    Alice (player 0), and Bob (player 2), all three treated symmetrically.
    Each player p has a set of moves (aka, choices, actions, etc.) denoted
    by an alphabet A_p. For clarity of the discussion, we assume that the
    set of moves of any two distinct players are disjoint.

    The game tree T (from Alice's perspective) is defined as follows.
    T has a set V(T) of nodes (aka vertices) and a set E(T) of edges.
    Each node v belongs to a single player, p(v), called "the player to act
    at v". At node v, the player p(v) has a set C(v) \subset A_p of possible
    moves. For each player p \in [0, 1, 2], the set of all nodes at which p
    acts is called the nodes of p, denoted V_p(T). Thus v(0), v(1), and v(2)
    form a partition of V(T).

    There are two kinds of nodes: the leafs L(T), at which the game must
    end and decision nodes D(T), at which the player to act must make a move.
    L(T) and D(T) are a partition of the node V(T). For example in Poker,
    a leaf node is reached at countdown, or when a player folds, or when the
    player runs out of money.

    There is a special node root(T) defined by

        root(T) := (c_0, p_0),

    where c_0 := / (/ is  a special symbol) and p_0 \in [0, 1, 2] is the player
    who begins the game, also known as the player to act at root(T).
    Typically, p_0 = 0 (i.e the game begins with chance).

    In accordance with the rules of the game, every other node
    y \in V(T)\{root(T)} is of the form

         y = v.(c, p(v))

    Where v \in D is another node and c \in C(v). The point "." between v and
    (c, p(v)) is a special marker. Thanks to the "perfect recall" assumption,
    exactly one v \in D satisfies the above equation and so

        ant(y) = v

    defines a single-valued function from V(T)\{root} to D(T). Thus each
    not-root node y is of the form

        y = ant(y).(c, p(ant(y)),

    with c \in C(ant(y)).

    We put a directed edge e(y) from ant(y) to y, and label it with the move c.
    The set of edges E(T) is simply the set of all such edges e(y).

    Thus a general non-root node is of the (unique!) form

        y = (c_0, p_0).(c_1, p_1).(c_2, p_2)...(c_l(y), p_l(y)),

    where l(y) > 0 is length of the path from root(T) to y, p_j is defined
    recursively (on l(y)) to equal p(y), the player to act at y. Thus each node
    y encodes the history of a play (ongoing or terminated). Also y is the the
    root of a unique subgame which begins at y.

    It is clear that T = (V(T), E(T)) as defined, is indeed a tree.

    Payoffs and Rationailty
    =======================
    Associated with T is a payoff function phi such that at each leaf node
    v \in L(T), player Alice (player 1) gets phi(v)$ and Bob (player 2)
    gets -phi(v)$; at v Alice losses, draws, or wins acording as phi(v) < 0,
    phi(v) = 0, or phi(v) > 0 respectively. The function phi uniquely
    determined by the sequence of moves made by the chance player. For
    example in Poker, phi simply scores each player's hand at countdown
    (assuming nobody has folded yet, etc.).

    A player is said to be rational if their aimed "goal" is to constraint
    the game to land on a leaf node at which their reward is as big as
    possible; otherwise they're considered irrational. We'll assume Alice
    and Bob are both rational. In particular, this assumption implies that
    Bob and Alice play non-cooperatively!

    The "chance" player (0), Imcomplete Information, and Information Sets
    =====================================================================
    Each time the "chance" player (aka player 0, aka nature) plays, she does so
    by generating a signal s from a fixed (and publicly known) probability
    distribution gamma_0 on A_0; two quantities zeta1(s) and zeta2(s) are
    computed from s and revealed exclusively to Alice and Bob respectively.
    This marks the begining of a round. It is assumed that both Alice and Bob
    know a public procedure which can be used to recover s from its parts
    zeta1(s) and zeta2(s). As an example (Kuhn's Poker), take

        A_0 := {'12', '13', '21', '23', '31', '32'},
        zeta1(s) := s[0], and
        zeta2(s) := s[1], for all s \in A_0.

    The 'projectors' zeta1 and zeta2 may change from round, as is the case
    in games like Poker. Usually in Poker, zeta1(s) = zeta2(s) = s for
    community cards; for non-community cards zeta1(s) are the cards given to
    Alice face-down and zeta2(s) are the cards given to Bob face-down, and
    it is practically impossible to recover s from only zeta1(s) or zeta2(s),
    behond guessing.

    Because Alice (respective Bob) cannot always determine a move s \in A_0
    made by the chance player (as she may only know part of s, not all of
    it), she can't always tell on which node she is on the game tree. This
    indeterminacy is referred to as imcomplete information, and calls for
    strategic speculation by both players. For each player p \in [1, 2],
    the is a partitioning I_p \subset \powerset(V_p(T)) V_p(T) of p into
    equivalence classes known as information sets. Each information set of p
    (i.e each element of I_p) contains a subset elements of V_p(T), which are
    all mutually indistinguishable to p. Of course if each element of each
    I_p is a singleton, then the game reduces to a game with complete
    information.

    """

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
        self.build_constraints()
        self.build_payoff_matrix()

    def init_misc(self):
        """Miscellaneous initializations."""
        self.last_node_played_patterns = {}
        for player, choices in self.PLAYER_CHOICES.iteritems():
            choices = "|".join(choices)
            self.last_node_played_patterns[player] = re.compile(
                '^(.*)(\([%s],\d\))[^%s]*$' % (choices, choices))

    def is_leaf(self, node):
        """Checks whether given node is leaf / terminal."""
        return not "info" in self.tree.node[node]

    def is_root(self, node):
        """checks whether given node is root."""
        return node == "(/,0)"

    def player_to_play(self, node):
        """Returns player to play at given node."""
        return int(re.match("^.*?(\d)\\)$", node).group(1))

    def previous_player(self, node):
        """Returns player who plays play leads to given node."""
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
        """Builds the game tree as an nx.DiGraph object.

        Abstract method.
        """
        raise NotImplementedError

    def set_node_info(self, player, node, choices):
        """Computes info at a node given node, and stores it in the tree
        structure for the game."""

        # create node if inexsitent
        if not node in self.tree.node:
            self.tree.add_node(node)
        self.tree.node[node]["player"] = player

        # return if no choices (i.e if leaf)
        if not choices:
            return

        # create and store node info (i.e information set to which it belongs)
        info = self.compute_info_at_node(node, choices)
        if not player in self.infosets:
            self.infosets[player] = {}
        if not info in self.infosets[player]:
            self.infosets[player][info] = []
        self.infosets[player][info].append(node)
        self.tree.node[node]["info"] = dict(chance=info[0], sigma=info[1],
                                            choices=info[2])

    def add_labelled_edge(self, src, choice, next_player, choices=None,
                          proba=1., **kwargs):
        """Adds a labelled edge to the tree.

        Parameters
        ----------
        src : string
            Source (current) node.

        next_player : int
            Player to move at a node `src`.

        choice : char
            The choice made by player at node `src`.

        choices : list, optional (default None)
            Choices available at destination (final) node.

        proba : float in the interval (0, 1]
            If player is chance player, then `proba` is the probability with
            which it is making the choice.

        **kwargs : value-pair dict-like
            Additional data to store at destination node.
        """
        dst = "%s.(%s,%s)" % (src, choice, next_player)
        assert 0. < proba <= 1.
        proba *= self.tree.node[src].get("proba", 1.)
        self.tree.add_node(dst, proba=proba, **kwargs)
        self.tree.add_edge(src, dst)
        self.edge_labels[(src, dst)] = choice
        if choices is None:
            choices = []
        self.set_node_info(next_player, dst, choices)
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

    def chop(self, node, player=None):
        """Return list of all nodes played along this path,
        by given player (defaults to None)"""
        if player is None:
            player = self.previous_player(node)
        pieces = node.split(".")
        for j in xrange(len(pieces)):
            child = pieces[:j + 1]
            if self.previous_player(".".join(child)) == player:
                yield child

    def node2seq(self, node):
        """Returns sequence of information-set-relabelled moves made
        by previous player along this path."""
        if node is None:
            return []
        else:
            return [(self.tree.node['.'.join(item[:-1])]["info"], item[-1])
                    for item in self.chop(node)]

    def leafs_iter(self, data=True):
        """Iterate over leafs of game tree."""
        if data:
            for node, data in self.tree.nodes_iter(data=True):
                if self.is_leaf(node):
                    yield node, data
        else:
            for node in self.tree.nodes_iter(data=False):
                if self.is_leaf(node):
                    yield node

    def last_node(self, node, player):
        """Last node played by given player, before this point."""
        if self.is_root(node):
            return None
        if self.previous_player(node) == player:
            return node
        else:
            return self.last_node(".".join(node.split('.')[:-1]), player)

    def build_sequences(self):
        """Each sequence for a player is of the form (i_1, a_1)(i_2,, a_2)...,
        where each a_j is an action at the information set i_j.
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
        for sequences in self.sequences.itervalues():
            sequences.sort()
        return self.sequences

    def build_constraints(self):
        """Generates matrices for the equality constraints on each player's
        admissible realization plans.

        The constraints for player p are a pair E_p, e_p corresponding to a
        set of linear constraints read as "E_p x = e_p". E_p has as many
        columns as player p has sequences, and as many rows as there
        information sets for player p, plus 1.
        """
        self.constraints.clear()

        # loop over players
        for player in [1, 2]:
            row = np.zeros(len(self.sequences[player]))
            row[0] = 1.
            E = [row]
            # loop over sequences for player
            for i, sigma in enumerate(self.sequences[player]):
                mem = {}
                # loop over all sequences for which the sequence tau is a
                # preix
                for j, tau in enumerate(self.sequences[player]):
                    if tau and tau[:-1] == sigma:
                        # sigma is the (unique) antecedant of tau
                        h, _ = tau[-1]
                        h_ = tuple(h.values())  # handle unhashable dict type
                        if h_ not in mem:
                            mem[h_] = []
                        mem[h_].append(j)
                # fill row: linear constraint (corresponds to Bayes rule)
                for where in mem.values():
                    row = np.zeros(len(self.sequences[player]))
                    row[i] = -1.
                    row[where] = 1.
                    E.append(row)
            # right handside
            e = np.zeros(len(self.infosets[player]) + 1)
            e[0] = 1.
            self.constraints[player] = np.array(E), e
        return self.constraints

    def build_payoff_matrix(self):
        """Builds payoff matrix from player 1's perspective.

        The rows (resp. columns) are labelled with player 1's (resp. 2's)
        sequences.

        TODO
        ----
        Use tricks in equation (38) of "Smoothing Techniques for Computing Nash
        Equilibria of Sequential Games" http://repository.cmu.edu/cgi
        /viewcontent.cgi?article=2442&context=compsci to compute the payoff
        matrix as a block diagonal matrix whose blocs are sums of Kronecker
        products of sparse matrices. This can be done by appropriately
        permuting the list of sequences of each (non-chance) player.
        """
        self.payoff_matrix = np.zeros((len(self.sequences[1]),
                                       len(self.sequences[2])))
        for leaf, data in self.leafs_iter(data=True):
            i = self.sequences[1].index(self.node2seq(self.last_node(leaf, 1)))
            j = self.sequences[2].index(self.node2seq(self.last_node(leaf, 2)))
            self.payoff_matrix[i, j] += data['payoff'] * data['proba']
        return self.payoff_matrix

    def draw(self):
        """Draw game tree."""
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
    """Kuhn's 3-card Poker: http://en.wikipedia.org/wiki/Kuhn_poker"""
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
        """Builds the game tree for Kuhn's Poker, as an nx.DiGraph object."""
        self.set_node_info(0, "(/,0)", self.PLAYER_CHOICES[0])
        for perm, proba in zip(self.PLAYER_CHOICES[0], [1. / 6] * 6):
            perm_word = self.chance_char_to_word(perm)
            x = self.add_labelled_edge("(/,0)", perm, 1,
                                       choices=["(C,2)", "(R,2)"], proba=proba)

            for a in 'CR':
                if a == 'C':  # check
                    y = self.add_labelled_edge(
                        x, a, 2, choices=["(r,1)", "(c,1)"])
                    for b in 'cr':
                        if b == "c":  # check
                            self.add_labelled_edge(
                                y, b, 1, payoff=self.cmp_cards(*perm_word))
                        else:  # raise
                            z = self.add_labelled_edge(
                                y, b, 1, choices=['(F,*)', '(K,*)'])
                            for w in 'FK':
                                if w == "F":  # fold
                                    self.add_labelled_edge(
                                        z, w, "*", payoff=-1)
                                else:  # call
                                    self.add_labelled_edge(
                                        z, w, "*",
                                        payoff=2 * self.cmp_cards(*perm_word))
                else:  # raise
                    y = self.add_labelled_edge(
                        x, a, 2, choices=["(f,*)", "(k,*)"])
                    for b in "fk":
                        if b == "f":  # fold
                            self.add_labelled_edge(y, b, "*", payoff=1.)
                        else:  # call
                            self.add_labelled_edge(
                                y, b, "*",
                                payoff=2 * self.cmp_cards(*perm_word))


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


def test_all_leafs_have_proba_datum():
    g = Kuhn3112()
    for _, leaf_data in g.leafs_iter():
        assert_true("proba" in leaf_data)

if __name__ == "__main__":
    import pylab as pl
    from sequential_games import compute_ne
    game = Kuhn3112()
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
    pl.figure()
    game.draw()
    pl.show()
