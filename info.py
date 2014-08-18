import itertools
import networkx as nx
from nose.tools import assert_true, assert_equal


def kuhn3112_gametree():
    """Function to generate KUHN(3, 1, 1, 2) poker game tree."""
    T = nx.DiGraph()
    T.add_node('/', player=None)
    edge_labels = {}

    def _add_labelled_edge(x, y, move, **kwargs):
        T.add_node(y, **kwargs)
        T.add_edge(x, y)
        edge_labels[(x, y)] = move

    proba = "1/6"
    for perm in itertools.permutations('123', 2):
        _add_labelled_edge('/', '/.%s' % ''.join(perm), proba, player=0,
                           proba=eval("1. * %s" % proba))
        for a in 'CR':
            _add_labelled_edge('/.%s' % ''.join(perm), '/.%s.%s' % (
                    ''.join(perm), a), a, player=1)
            if a == 'C':  # check
                for b in 'cr':
                    dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                    _add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                       dst, b, player=2)
                    if b == "c":
                        T.add_node(dst, player=2, payoff=cmp(*perm))
                    else:  # raise
                        for x in 'FK':
                            dst = '/.%s.%s.%s.%s' % (''.join(perm), a, b, x)
                            _add_labelled_edge(
                                '/.%s.%s.%s' % (''.join(perm), a, b),
                                dst, x, player=1)
                            if x == "F":  # fold
                                T.add_node(dst, player=1, payoff=-1)
                            else:  # call
                                T.add_node(dst, player=1,
                                           payoff=2 * cmp(*perm))
            else:  # raise
                for b in "fk":
                    dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                    _add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                       dst, b, player=2)
                    if b == "f":  # fold
                        T.add_node(dst, player=2, payoff=1.)
                    else:  # call
                        T.add_node(dst, player=2, payoff=2 * cmp(*perm))

    return T, edge_labels


def player_moves(player):
    """Tet possible moves of given player in the game tree (T)."""
    if player == 0.:
        return map(lambda x: ''.join(x),
                   list(itertools.permutations('123', 2)))
    else:
        moves = 'crfk'
        if player == 1.:
            moves = moves.upper()
        return sorted(moves)


def project_seq_onto_player(seq, moves):
    """Ignores all but the moves of a player, along given sequence."""
    if isinstance(seq, str):
        seq = seq.split(".")
    view = [x for x in seq if x in moves]
    return view


def test_player_moves():
    assert_equal(player_moves(1), sorted(['C', 'R', 'F', 'K']))
    assert_equal(player_moves(2), sorted(['c', 'r', 'f', 'k']))
    assert_equal(player_moves(0), sorted(['12', '13', '21', '23', '31', '32']))


def test_project_seq_onto_player():
    assert_equal(project_seq_onto_player('/23.C.r.F', 'CFKR'), ['C', 'F'])


def test_add_labelle_edge():
    T = Tree()
    
