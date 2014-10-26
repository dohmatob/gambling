# Author: DOHMATOB Elvis Dopgima

import itertools
from nose.tools import assert_equal, assert_false, assert_true
import networkx as nx
import numpy as np

### Useful primitives #########################################################
move_id = lambda move, p: move.upper() if p == 1 else move.lower()
whose = lambda move: 1 if move.upper() == move else 2
opponent = lambda p: 1 + p % 2  # opponent of player
norminal = lambda t: 1 + t % 2  # norminal player on round t
copy_a  = lambda a: [a[0], a[1], a[2].copy(), a[3].copy()]  # a copy of a
ok = lambda a, limit: a[0] <= limit and np.all(a[3] >= 0.)  # is state a ok ?
join = lambda s: ".".join([x for x in s if x])  # joins a series of commands
is_leaf = lambda node: tree.out_degree(node) == 0.  # is this node a leaf
is_root = lambda node: node == "/"  # is this node the root ?


def node_color(node):
    """Get node color (i.e player to whom node belongs)."""
    if is_root(node):
        return "b"
    node = node.split(".")[-1]
    if node in ["JJ", "JQ", "QJ", "QQ"]:
        return "r"
    elif whose(node) == 1:
        return "g"
    else:
        return "r"


def chance(t):
    """Generator for choices of the chance player, on round t."""
    if t > 0:
        yield ''
    else:
        for perm in itertools.product("JQ", "JQ"):
            yield "".join(perm)


def do_call(a, p):
    """Executes a "call" for player p."""
    a[1] += a[0]
    a[2][p - 1] += a[0]
    a[3][p - 1] -= a[0]
    return a


def do_raise(a, p):
    """Executes a "raise" for player p."""
    a[0] *= 2
    return do_call(a, p)


def u(a, cmd):
    """Executes a command (cmd) and updates the band info `a`."""
    if not cmd:
        return a
    if cmd[0].lower() == "f":
        return a
    p = whose(cmd[0])
    if cmd[0].lower() == "k":
        do_call(a, p)
    elif cmd[0].lower() == "r":
        do_raise(a, p)
    else:
        pass
    return u(a, cmd[1:])


def leafs(t, T, a, L):
    """Recursively generates all leaf nodes in game tree.

    The function call leafs(t, T, a, L)  generates leafs nodes that belong
    round t onwards, where a is the bank on round t.

    Parameters
    ----------
    t: int
        Round counter.

    T: int
        Maximum number of rounds (game exits if t <= T).

    a: list of floats and lists
       Current banking info.
       a[0]: bet size (initialized to 1 at the beginning of each round)
       a[1]: running pot
       a[2]: total amount chipped into the pot thus far, by each player
       a[2][0]: total amount chipped into the pot thus far, by player 1
       a[2][1]: total amount chipped into the pot thus far, by player 2
       a[1]: running capital of each player
       a[1][0]: running capital of player 1
       a[1][1]: running capital of player 2

    Returns
    -------
    An iterator of strings of caracters seperated by "."
    (to demarkate succesive moves)
    """
    if t < T and ok(a, L):
        p = norminal(t)  # norminal player for this round
        q = opponent(p)  # the other player

        # chance player always opens a round by disclosing some (possibly void)
        # partial (or fully) observable signals (hole, cards, community cards,
        # etc.)
        for perm in chance(t):
            if t == 0:
                perm = join(["/", perm])

            # "fold" (always possible)
            yield join([perm, move_id("f", p)])

            # "check" (if possible!) and continue to next round
            c = move_id("c", p)
            a[0] = 1.
            for x in leafs(t + 1, T, a, L):
                yield join([perm, c, x])

            # "call"
            k = move_id("k", p)
            token = join([move_id("r", q), move_id("r", p)])
            for o, last_sip in zip([p, q], [move_id("r", q), '']):
                for end in [move_id('f', o), move_id('k', o)]:
                    rep = ""
                    x = join([perm, k, rep, last_sip, end])
                    a_ = u(copy_a(a), x)

                    # alternate a sequence of "raise" actions & then exit with
                    # a "fold" or "call", and perhaps move on to next round
                    while ok(a_, L):
                        yield x
                        for y in leafs(t + 1, T, a_, L):
                            yield join([x, y])
                        rep += token
                        x = join([perm, k, rep, end])
                        a_ = u(copy_a(a), x)


def nodes_on_path(path):
    """Generates all nodes on a given path."""
    path = path.split(".")
    for j in xrange(len(path)):
        yield join(path[:j + 1])


def edges(t, T, a, L):
    """Generates all edges in game tree."""
    for leaf in leafs(t, T, a, L):
        pred = None
        for node in nodes_on_path(leaf):
            if pred is not None:
                e = pred, node
                print "%s -> %s" % e
                yield e
            pred = node


def test_move_id():
    assert_equal(move_id("r", 1), "R")
    assert_equal(move_id("r", 2), "r")


def test_whose():
    assert_equal(whose("K"), 1)
    assert_equal(whose("c"), 2)


def test_opponent():
    assert_equal(opponent(1), 2)
    assert_equal(opponent(2), 1)


def test_norminal():
    assert_equal(norminal(90), 1)
    assert_equal(norminal(871), 2)


def test_copy_a():
    a = [1, 0, np.zeros(2), np.array([10, 100])]
    a_ = copy_a(a)
    a_[3] += 45.
    assert_equal(a[3][0], 10.)
    assert_equal(a[0], a[0])


def test_u():
    a = [1, 0, np.zeros(2), np.array([10, 100])]
    u(a, "rRrRk")
    assert_equal(a[0], 16)
    assert_equal(a[1], 46)
    assert_equal(a[2].sum(), a[1])

    a = [1, 0, np.zeros(2), np.array([50, 100])]
    u(a, "rRrRrK")
    assert_equal(a[0], 32)
    assert_equal(a[1], 2 + 4 + 8 + 16 + 32 + 32)
    assert_equal(a[2].sum(), a[1])


def test_chance():
    assert_equal(list(chance(0)), ['JJ', 'JQ', 'QJ', 'QQ'])
    assert_equal(list(chance(1)), [''])


def test_ok():
    assert_false(ok([1., 4., np.array([4., 0]), np.array([-4, 10])], np.inf))
    assert_false(ok([4., 14., np.array([4., 0]), np.array([4, 10])], 2))
    assert_true(ok([4., 14., np.array([4., 0]), np.array([4, 10])], 4))


def test_one_round_leafs():
    a = [1., 0., np.zeros(2), np.array([2, 2])]
    l = list(leafs(0, 1, a, 2.))
    assert_true('/.QJ.F' in l)
    assert_true('/.JJ.F' in l)
    assert_false('./JJ.c.F' in l)


def test_three_rounds_leafs():
    a = [1., 0., np.zeros(2), np.array([10, 10])]
    l = list(leafs(0, 3, a, 2.))
    for perm in ['JJ']:
        assert_true('/.%s.C.f' % perm in l)
        assert_true('/.%s.C.f' % perm in l)
        assert_true('/.%s.C.k.F' % perm in l)
        assert_true('/.%s.C.k.K.F' % perm in l)
        assert_true('/.%s.C.k.K.K.f' % perm in l)
        assert_false('/.%s.C.r.R.f' % perm in l)


def test_nodes_on_path():
    leaf = "/.C.r.R.r.R.f"
    ancestors = list(nodes_on_path(leaf))
    for node in ['/', '/.C.r', '/.C.r.R', '/.C.r.R.r', leaf]:
        assert_true(node in ancestors)


def test_edges():
    a = [1, 0., np.array([0, 0]), np.array([10000, 10000])]
    tree = nx.DiGraph()
    tree.add_edges_from(edges(0, 1, a, 10))
    for x, y in tree.edges_iter():
        assert_equal(x.split('.'), y.split('.')[:-1])
    for node in tree.nodes_iter():
        for x in node.split("."):
            if len(x) == 2:
                assert_true(x in chance(0))
            else:
                assert_equal(len(x), 1)
        assert_true(tree.out_degree(node) < 5)

if __name__ == "__main__":
    import pylab as pl

    a = [1, 0., np.array([0, 0]), np.array([1, 2])]
    tree = nx.DiGraph()
    tree.add_edges_from(edges(0, 8, a, 3))
    for x, y in tree.edges_iter():
        assert y.split(".")[:-1] == x.split(".")
    for node in tree.nodes_iter():
        assert tree.out_degree(node) < 6, tree.successors(node)

    pos = nx.graphviz_layout(tree)
    leaf_nodes = [n for n in tree.nodes() if is_leaf(n)]
    decision_nodes = [n for n in tree.nodes() if is_leaf(n)]
    nx.draw_networkx_nodes(tree, pos,
                           node_color=map(node_color, tree.nodes()),
                           node_size=30
                           )

    nx.draw_networkx_edges(tree, pos, arrows=False)
    pl.axis("off")
    pl.tight_layout()
    pl.show()
