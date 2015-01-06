# Author: DOHMATOB Elvis

import re
from nose.tools import assert_equal, assert_true
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from primal_dual import primal_dual_ne, primal_dual_sg_ne


class Game(object):
    """Sequence-form representaion of two-person zero-sum games with perfect
    recall and imcomplete information (like Poker, love, war, diplomacy, etc.)

    Terminology
    -----------

    Game Tree (normal-form definition)
    ==================================
    There are 3 players, namely: the "chance" player (player 0, aka nature),
    Alice (player 0), and Bob (player 2), and the "mute" (player 3) all four
    treated symmetrically. Each player p has a set of moves (aka, choices,
    actions, etc.) denoted by an alphabet A_p. For clarity of the discussion,
    we assume that the set of moves of any two distinct players are disjoint.
    Furthermore, player 3 (mute) has no moves, i.e A_3 = {}, hence her name.

    The game tree T (from Alice's perspective) is defined as follows.
    T has a set V(T) of nodes (aka vertices) and a set E(T) of edges.
    Each node v belongs to a single player, p(v), called "the player to act
    at v"; p(v) := 3 if v is a leaf node. At node v, the player p(v) has a set
    C(v) \subset A_p of possible moves. For each player p \in [0, 1, 2],
    the set of all nodes at which p acts is called the nodes of p,
    denoted V_p(T). Thus v(0), v(1), and v(2) form a partition of V(T).

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
    BOOK = None
    PLAYER_COLORS = "gbr"

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

    def chance_word_to_char(self, word):
        """Convert chance choice from long to short form."""
        return self.PLAYER_CHOICES[0][self.BOOK.index(word)]

    def chance_char_to_word(self, char):
        """Converts chance choice from short to long form."""
        return self.BOOK[self.PLAYER_CHOICES[0].index(char)]

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

    def unpack_choice(self, choice):
        """Splits a choice into the a pair (char, int) pair, namely (c, p)
        where `c` is the character representing the choice and `p` is the
        player to act next."""
        choice, next_player = re.match("^\((.),(\d)\)$", choice).groups()
        return choice, int(next_player)

    def add_labelled_edge(self, src, choice, next_player, choices=None,
                          proba=1., **kwargs):
        """Adds a labelled edge to the tree.

        Parameters
        ----------
        src : string
            Source (current) node.

        next_player : int
            Player to move at a node `src`.

        choice : string of the for "(%c,%i)"
            The choice made by player at node `src`.

        choices : list, optional (default None)
            Choices available at destination (final) node.

        proba : float in the interval (0, 1]
            If player is chance player, then `proba` is the probability with
            which it is making the choice.

        **kwargs : value-pair dict-like
            Additional data to store at destination node.
        """
        choice_ = "(%s,%i)" % (choice, next_player)
        if not choice_ in self.tree.node[src]["info"]["choices"]:
            raise RuntimeError("%s is not a choice at %s" % (choice_, src))
        dst = "%s.%s" % (src, choice_)
        assert 0. < proba <= 1.
        proba *= self.tree.node[src].get("proba", 1.)
        self.tree.add_node(dst, proba=proba, **kwargs)
        self.tree.add_edge(src, dst)
        self.edge_labels[(src, dst)] = (choice if self.player_to_play(src) > 0
                                        else self.chance_char_to_word(choice))
        if choices is None:
            choices = []
        self.set_node_info(next_player, dst, choices)
        return dst

    def level_first_traversal(self, player=None):
        """Level-first traversal of the game tree."""
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
            # right handside, e
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
        matrix as a block diagonal matrix whose blocks are sums of Kronecker
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

    def draw(self, figsize=None):
        """Draw game tree."""
        plt.figure(figsize=figsize)
        pos = nx.graphviz_layout(self.tree, prog='dot')

        # leaf (terminal) nodes
        leaf_nodes = [node for node in self.tree.nodes() if self.is_leaf(node)]
        nx.draw_networkx_nodes(self.tree, pos, nodelist=leaf_nodes,
                               enode_shape='s', node_color="k")

        # decision nodes
        dec_nodes = [node
                     for node in self.tree.nodes() if not self.is_leaf(node)]
        nx.draw_networkx_nodes(
            self.tree, pos, nodelist=dec_nodes, node_shape='o',
            node_color=[self.PLAYER_COLORS[self.tree.node[node]["player"]]
                        for node in dec_nodes])

        # labelled edges
        nx.draw_networkx_edges(self.tree, pos, arrows=False)
        nx.draw_networkx_edge_labels(self.tree, pos,
                                     edge_labels=self.edge_labels)

        # plt.axis("off")


class Kuhn3112(Game):
    """Kuhn's 3-card Poker: http://en.wikipedia.org/wiki/Kuhn_poker"""
    PLAYER_CHOICES = {0: ['u', 'v', 'w', 'x', 'y', 'z'],
                      1: list('CFKR'),
                      2: list('cfkr')}
    BOOK = ['12', '13', '21', '23', '31', '32']

    def blur(self, choice, player):
        """Projects chance player's choice into given player's space (i.e
        reveals only the part of the choice which should be visible to
        the given player (cf. hole / private cards in Poker).
        """
        return self.chance_char_to_word(choice)[player - 1]

    def unblur(self, part, player):
        """Computes chance choice which has given projection unto given
        player."""
        if player == 0:
            raise ValueError("Must be non-chance player!")
        for i, x in enumerate(self.BOOK):
            if player == 1 and x.startswith(part):
                return self.PLAYER_CHOICES[0][i]
            elif player == 2 and x.endswith(part):
                return self.PLAYER_CHOICES[0][i]
        raise RuntimeError("Shouldn't be here")

    def cmp_cards(self, a, b):
        """Compares two cards lexicographically."""
        return cmp(int(a), int(b))

    def build_tree(self):
        """Builds the game tree for Kuhn's Poker, as an nx.DiGraph object."""
        self.set_node_info(
            0, "(/,0)", ["(%s,1)" % choice
                         for choice in self.PLAYER_CHOICES[0]])
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
                                y, b, 1, choices=['(F,3)', '(K,3)'])
                            for w in 'FK':
                                if w == "F":  # fold
                                    self.add_labelled_edge(
                                        z, w, 3, payoff=-1)
                                else:  # call
                                    self.add_labelled_edge(
                                        z, w, 3,
                                        payoff=2 * self.cmp_cards(*perm_word))
                else:  # raise
                    y = self.add_labelled_edge(
                        x, a, 2, choices=["(f,3)", "(k,3)"])
                    for b in "fk":
                        if b == "f":  # fold
                            self.add_labelled_edge(y, b, 3, payoff=1.)
                        else:  # call
                            self.add_labelled_edge(
                                y, b, 3,
                                payoff=2 * self.cmp_cards(*perm_word))


class SimplifiedPoker(Game):
    """A simplified 2-card Poker."""
    PLAYER_CHOICES = {0: ['u', 'v', 'w', 'x'],
                      1: list('BF'), 2: list('bf')}

    BOOK = ['KK', 'KA', 'AK', 'AA']

    def blur(self, choice, player):
        """Projects chance player's choice into given player's space (i.e
        reveals only the part of the choice which should be visible to
        the given player (cf. hole / private cards in Poker).
        """
        return self.chance_char_to_word(choice)[player - 1]

    def unblur(self, part, player):
        """Computes chance choice which has given projection unto given
        player."""
        if player == 0:
            raise ValueError("Must be non-chance player!")
        for i, x in enumerate(self.BOOK):
            if player == 1 and x.startswith(part):
                return self.PLAYER_CHOICES[0][i]
            elif player == 2 and x.endswith(part):
                return self.PLAYER_CHOICES[0][i]
        raise RuntimeError("Shouldn't be here!")

    def cmp_cards(self, a, b):
        """Compares two cards lexicographically."""
        return cmp(int(a), int(b))

    def build_tree(self):
        self.set_node_info(
            0, "(/,0)", ["(%s,1)" % choice
                         for choice in self.PLAYER_CHOICES[0]])
        for perm, proba in zip(self.PLAYER_CHOICES[0], [1. / 4] * 4):
            perm_word = self.chance_char_to_word(perm)
            child = self.add_labelled_edge(
                "(/,0)", perm, 1, proba=proba, choices=["(B,2)", "(F,3)"])
            for x in 'BF':
                if x == "F":
                    self.add_labelled_edge(child, x, 3, payoff=-1.)
                else:
                    a = self.add_labelled_edge(
                        child, x, 2, payoff=-1., choices=["(b,3)", "(f,3)"])
                    for y in 'bf':
                        if y == "f":
                            payoff = 1
                        elif perm_word[0] != perm_word[1]:
                            payoff = 4. if perm_word[0] == "A" else -4.
                        else:
                            payoff = 0.
                        self.add_labelled_edge(a, y, 3, payoff=payoff)


class Player(object):
    """Generic player

    Parameters
    ----------
    name : string
        Name of player.

    player : int in [0, 1]
        Player number.

    Attributes
    ----------
    location : string
        Current node / state of game tree, from player's perspective
        (Note that there is uncertainty due to imcomplete information.)
    """
    def __init__(self, name, player, game):
        self.name = name
        self.player = player
        self.game = game
        self.location = "(/,0)"

    def __str__(self):
        return self.name

    def clear(self):
        self.location = "(/,0)"

    def choice(self, choices):
        """Invoked to make a choice from given list of choices.

        XXX Use local rng!

        Parameters
        ----------
        start : string
            Node from which choice is to be made

        choices : list of characters from universal choice alphabet
            Choices available at node `start`

        Returns
        -------
        c : string
            The made choice.
        """
        return np.random.choice(choices, size=1)[0]

    def observe(self, player, choice):
        """Observe given player make a move, and memorize it."""
        if player == 0:
            # unblur chance move by extrapolating unto any node in current
            # information set
            c, i = re.match("\((.),(\d)\)", choice).groups()
            i = int(i)
            c = self.game.unblur(c, self.player)
            choice = "(%c,%i)" % (c, i)
        self.location = "%s.%s" % (self.location, choice)


class _ChancePlayer(Player):
    """Nature / chance player (private!)."""
    pass


class NashPlayer(Player):
    """Player using NE solution concept.

       See Benhard von Stengel 1995, etc.
    """
    def __init__(self, name, player, game):
        super(NashPlayer, self).__init__(name, player, game)
        self.game = game
        self.sequences = self.game.sequences[self.player]
        self.compute_optimal_rplan()
        self.location = "(/,0)"

    def compute_optimal_rplan(self):
        """Compute optimal realization plan, which is our own part of
        the Nash-Equlibrium for the sequence-form representation of the
        game. This computation is done offline.
        """

        E, e = self.game.constraints[1]
        F, f = self.game.constraints[2]
        A = self.game.payoff_matrix
        x, y, _ = primal_dual_ne(A, E, F, e, f)
        self.rplan = np.array([x, y][self.player - 1])

    def choice(self, choices):
        """Makes a choice at give node, according to our optimal realization
        plan pre-computed offline.

        Parameters
        ----------
        start : string
            Node from which choice is to be made

        choices : list of characters from universal choice alphabet
            Choices available at node `start`

        Returns
        -------
        c : string
            The made choice.
        """
        # get all nodes possible "next" nodes (in sequence-form)
        menu = map(self.game.node2seq, ['.'.join([self.location, choice])
                                        for choice in choices])

        # get the probabilities with which these nodes are played next
        weights = self.rplan[map(self.sequences.index, menu)]

        # ... and then make the next move according to this distribution
        choice = choices[np.random.choice(
                range(len(menu)), p=weights / weights.sum(), size=1)[0]]
        self.observe(self.player, choice)
        return choice


class Duel(object):
    """Two-person zero-sum sequential game duel.

    Parameters
    ----------
    game : `Game` instance
        Instance of game to be played.

    players : list of 2 `Player` objects
        The contending players.
    """
    def __init__(self, game, players, verbose=1):
        self.game = game
        self.players = players
        self.verbose = verbose
        self.nature = _ChancePlayer("nature", 0, game)

    def play(self, root="(/,0)"):
        """Recursively makes players play subgame rooted at node `start`.

        Paremeters
        ----------
        players : List of 2 `Player` instances
            The 2 players competing.

        root : string
            Root node of subgame about to be played.

        Returns
        -------
        term : string
            Terminal node at which the game has ended.

        payoff: float
            Payoff to players[0] (players[0] gets -payoff since the game
            is zero-sum).
        """
        if self.game.is_root(root):
            print "Oracle: Entering new game..."
            for player in self.players:
                player.clear()

        # end subgame if root is a leaf
        if self.game.is_leaf(root):
            payoff = self.game.tree.node[root]["payoff"]
            if self.verbose:
                print "Oracle: Terminated at leaf node %s with payoff %g." % (
                    root, payoff)
            print
            return root, payoff

        # get player to start subgame
        p = self.game.player_to_play(root)
        player = self.players[p - 1] if p > 0 else self.nature

        # retrieve available choices
        choices = self.game.tree.node[root]['info']['choices']

        # let player make a choice
        if self.verbose:
            if p > 0:
                print "True location: %s" % root
                print "%s's assumed location: %s" % (
                    player, player.location)
        choice = player.choice(list(choices))
        assert choice in choices
        if self.verbose:
            print "%s: %s" % (player.name, choice)

        # let other player observe what has just been played
        if p == 0:
            # blur
            c, next_player = self.game.unpack_choice(choice)
            for i in xrange(2):
                c_ = self.game.blur(c, i + 1)
                self.players[i].observe(0, "(%s,%i)" % (c_, next_player))

        else:
            other_player = 1 + p % 2
            self.players[other_player - 1].observe(other_player, choice)

        # play subsubgame
        root = ".".join([root, choice])
        return self.play(root=root)


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


def test_all_leafs_have_proba_datum():
    for game_cls in [Kuhn3112, SimplifiedPoker]:
        game = game_cls()
        for _, leaf_data in game.leafs_iter():
            assert_true("proba" in leaf_data)


def test_play():
    for game_cls in [Kuhn3112, SimplifiedPoker]:
        game = game_cls()
        players = [Player('alice', 1, game), Player('bob', 2, game)]
        duel = Duel(game, players)
        for _ in xrange(10):
            term, payoff = duel.play()

            # check term is a leaf
            assert_true(game.is_leaf(term))

            # check game over
            term_, payoff_ = duel.play(root=term)
            assert_equal(term_, term)
            assert_equal(payoff_, payoff)


def test_nash_player():
    game = SimplifiedPoker()
    nash = NashPlayer("alice", 1, game)
    opponents = [NashPlayer("bob", 2, game), Player("olaf", 2, game)]
    for opponent in opponents:
        duel = Duel(game, [nash, opponent], verbose=0)
        mean_payoff = np.mean([duel.play()[1]
                               for _ in xrange(1000)])
        if isinstance(opponent, NashPlayer):
            np.testing.assert_almost_equal(mean_payoff, -.25,
                                           decimal=1)
        else:
            assert_true(mean_payoff >= -.25)


if __name__ == "__main__":
    for game_cls in [SimplifiedPoker, Kuhn3112, "simplex"][1:]:
        if game_cls == "simplex":
            game = None
            name = game_cls
            A = np.random.uniform(-1, 1, size=(1000, 1000))
            x, y, values, dgaps = primal_dual_sg_ne(A)
        else:
            game = game_cls()
            name = game.__class__.__name__
            E1, e1 = game.constraints[1]
            E2, e2 = game.constraints[2]
            A = game.payoff_matrix
            args = (A, E1, E2, e1, e2)
            x, y, values, dgaps = primal_dual_ne(A, E1, E2, e1, e2)
        print
        print "Nash Equilibrium:"
        print "x* = ", x
        print "y* =", y

        # plot evoluation of game value
        plt.figure(figsize=(13.5, 10))
        plt.semilogx(values, linewidth=4)
        value = values[-1]
        plt.axhline(-1. / 18. if isinstance(game, Kuhn3112) else value,
                    linestyle="--",
                    label="true value of the game: $%s$" % (
                        "-1 / 18" if isinstance(game, Kuhn3112) else
                        ("-1 / 4" if isinstance(game, SimplifiedPoker)
                         else "%.2e" % value)), linewidth=4, color="k")
        plt.xlabel("$k$", fontsize=25)
        plt.ylabel("value of game after $k$ iterations", fontsize=25)
        plt.legend(loc="best", prop=dict(size=25))
        # plt.title("%s: Sequence-form NE computation" % (
        #         game.__class__.__name__))
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig("%s_NE.pdf" % name)

        # plot evolution of gual gap
        plt.figure(figsize=(13.5, 10))
        plt.loglog(1. / np.arange(1, len(dgaps) + 1),
                   label="$\\mathcal{O}(1/k)$", linewidth=4, color="k")
        plt.loglog(np.abs(dgaps), label="Primal-Dual", linewidth=4)
        plt.xlabel("$k$", fontsize=25)
        plt.ylabel(
            "Duality gap $|e_1^Tp^{(k)} + e_2^Tq^{(k)}|$",
            fontsize=25)
        plt.legend(loc="best", prop=dict(size=25))
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.savefig("%s_dgap.pdf" % name)

        # draw game tree
        if not game is None:
            game.draw(figsize=(13.5, 7))
            plt.savefig("%s_gt.pdf" % name)

        # tmpfig = plt.figure(figsize=(13.5, 10))
        # plt.matshow(game.payoff_matrix, fignum=tmpfig.number)
        # plt.axis("off")
        # plt.savefig("%s_payoff.pdf" % game.__class__.__name__)

        # plt.figure()
        # game.draw()
        # plt.title("Game tree (T) for %s" % game.__class__.__name__)

    # plt.show()
