import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def _matrix2texstr(A, name="A", exclude=None):
    if exclude is None:
        exclude = []
    vals = dict((val, []) for val in np.unique(A))
    for val in exclude:
        del vals[val]
    for v, u in vals.items():
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] == v:
                    u.append((i, j))
    out = []
    for v, u in vals.items():
        out.append("$" + " = ".join(["%s(%i,%i)" % (name, i, j)
                                     for (i, j) in u] + ["\\textbf{%g}$" % v]))
    return ", ".join(out)


class Game(object):

    PLAYER_CHOICES = None
    BOOK = None
    PLAYER_COLORS = "gbr"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
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

    def player_choices(self):
        return self.player_choices()

    def init_misc(self):
        """Miscellaneous initializations."""
        self.PLAYER_CHOICES = self.player_choices()
        self.last_node_played_patterns = {}
        for player, choices in self.PLAYER_CHOICES.iteritems():
            choices = "|".join(choices)
            self.last_node_played_patterns[player] = re.compile(
                '^(.*)(\([%s],\d\))[^%s]*$' % (choices, choices))

    def is_leaf(self, node):
        """Checks whether given node is leaf / terminal."""
        return "info" not in self.tree.node[node]

    def is_root(self, node):
        """checks whether given node is root."""
        return node == "(/,0)"

    def player_to_play(self, node):
        """Returns player to play at given node."""
        return int(re.match("^.*?(\d)\\)$", node).group(1))

    def previous_player(self, node):
        """Returns player whose move leads to given node."""
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
        if node not in self.tree.node:
            self.tree.add_node(node)
        self.tree.node[node]["player"] = player

        # return if no choices (i.e if leaf)
        if not choices:
            return

        # create and store node info (i.e information set to which it belongs)
        info = self.compute_info_at_node(node, choices)
        if player not in self.infosets:
            self.infosets[player] = {}
        if info not in self.infosets[player]:
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
        if choice_ not in self.tree.node[src]["info"]["choices"]:
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

        def _skip(node):
            return False if player is None else (
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

    def matrices2texstrs(self):
        return [
            _matrix2texstr(self.constraints[1][0], exclude=[0.], name="E_1"),
            _matrix2texstr(self.constraints[1][0], exclude=[0.], name="E_2"),
            _matrix2texstr(self.payoff_matrix, exclude=[0.], name="A")]
