# Author: DOHMATOB Elvis

import re
from nose.tools import assert_equal, assert_true
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sequence_form import Game
from primal_dual import primal_dual_ne, primal_dual_sg_ne


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
        x, y, _, _, _, _, _ = primal_dual_ne(A, E, F, e, f)
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
    # matplotlib confs
    matplotlib.rcParams['text.latex.preamble'] = ['\\boldmath']
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('text', usetex=True)
    plt.rc('font', size=50)  # make the power in sci notation bigger

    rng = check_random_state(42)
    max_iter = 10000
    game_clses = ["simplex", SimplifiedPoker, Kuhn3112]
    for cnt, game_cls in enumerate(game_clses):
        if game_cls == "simplex":
            game = None
            name = game_cls
            A = rng.uniform(-1, 1, size=(100, 100))
            x, y, p, q, init, values, dgaps = primal_dual_sg_ne(
                A, max_iter=max_iter, strict=False)
        else:
            game = game_cls()
            name = game.__class__.__name__
            E1, e1 = game.constraints[1]
            E2, e2 = game.constraints[2]
            A = game.payoff_matrix
            args = (A, E1, E2, e1, e2)
            x, y, p, q, init, values, dgaps = primal_dual_ne(A, E1, E2, e1, e2,
                                                          max_iter=max_iter)

        # estimate constant in bound (4.1) of Yunlong He's Theorem 3.5
        aux = np.concatenate((y, p, x, q))
        cst = 2. * np.dot(aux, aux) / init["lambd"]

        # misc
        value = values[-1]
        print
        print "Nash Equilibrium:"
        print "x* = ", x
        print "y* =", y
        print "< x*, Ay* > =", value

        # plot evoluation of game value
        plt.figure(figsize=(13.5, 10))
        ax = plt.subplot("111")
        plt.grid("on")
        ax.semilogx(values, linewidth=4)
        value_ = "%.0e" % value
        m, e = re.search("(.+?)e(.+)", value_).groups()
        m = float(m)
        m = "" if m >= 0 else "-"
        e = re.sub("\-0*", "-", e)
        kwargs = {}
        plt.xlabel("\\textbf{$k$ (iteration count)}", fontsize=50)
        if cnt == 0:
            kwargs["label"] = "$\\langle x^*,Ay^*\\rangle$"
            ax.set_ylabel(("\\textbf{$\\langle x^{(k)}, Ay^{(k)} "
                           "\\rangle$"), fontsize=50)
        ax.axhline(
            -1. / 18. if isinstance(game, Kuhn3112) else value,
            linestyle="--", dashes=(30, 10), color="b",
            linewidth=4, **kwargs)
        plt.legend(loc="best", prop=dict(size=45), handlelength=1.5)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0., 0.))
        ax.tick_params(axis='both', which='major', labelsize=50)
        plt.tight_layout()
        plt.savefig("%s_NE.pdf" % name)

        # plot evolution of gual gap
        plt.figure(figsize=(13.5, 10))
        plt.grid("on")
        plt.loglog(cst / np.arange(1, len(dgaps) + 1), linestyle="--",
                   dashes=(30, 10), label="$\\mathcal{O}(1/k)$", linewidth=4,
                   color="b")
        plt.loglog(np.abs(dgaps), label="\\textbf{Algorithm 1}", linewidth=4)
        if cnt == 0:
            plt.ylabel("\\textbf{$||\\tilde{v}^{a}_k||$}", fontsize=50)
            plt.legend(loc="best", prop=dict(size=45), handlelength=1.5)
        plt.tick_params(axis='both', which='major', labelsize=50)
        plt.tight_layout()
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

    plt.show()
