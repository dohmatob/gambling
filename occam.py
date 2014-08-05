# Author: DOHMATOB Elvis Dopgima <gmdopp@gmail.com>

import itertools
import numpy as np
import pylab as pl
import networkx as nx


def kuhn3112_gametree():
    """Function to generate KUHN(3, 1, 1, 2) poker game tree."""
    G = nx.DiGraph()
    G.add_node('/', player=None)
    edge_labels = {}

    def _add_labelled_edge(x, y, move, **kwargs):
        G.add_node(y, **kwargs)
        G.add_edge(x, y)
        edge_labels[(x, y)] = move

    for perm in itertools.permutations('123', 2):
        _add_labelled_edge('/', '/.%s' % ''.join(perm), '1/6', player=0)
        for a in 'CR':
            _add_labelled_edge('/.%s' % ''.join(perm), '/.%s.%s' % (
                    ''.join(perm), a), a, player=1)
            if a == 'C':  # check
                for b in 'cr':
                    dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                    _add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                       dst, b, player=2)
                    if b == "c":
                        G.add_node(dst, player=2, payoff=cmp(*perm))
                    else:  # raise
                        for x in 'FK':
                            dst = '/.%s.%s.%s.%s' % (''.join(perm), a, b, x)
                            _add_labelled_edge(
                                '/.%s.%s.%s' % (''.join(perm), a, b),
                                dst, x, player=1)
                            if x == "F":  # fold
                                G.add_node(dst, player=1, payoff=-1)
                            else:  # call
                                G.add_node(dst, player=1,
                                           payoff=2 * cmp(*perm))
            else:  # raise
                for b in "fk":
                    dst = '/.%s.%s.%s' % (''.join(perm), a, b)
                    _add_labelled_edge('/.%s.%s' % (''.join(perm), a),
                                       dst, b, player=2)
                    if b == "f":  # fold
                        G.add_node(dst, player=2, payoff=1.)
                    else:  # call
                        G.add_node(dst, player=2, payoff=2 * cmp(*perm))

    return G, edge_labels


def is_leaf(G, node):
    """Checks whether given node is leaf / terminal."""
    return G.out_degree(node) == 0.


def player_moves(player):
    """Get possible moves of given player in the game tree (G)."""
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


def player_sequences(G, player, moves=None):
    """Get possible sequences of given player in the game tree (G)."""
    seqs = [[]]
    if moves is None:
        moves = player_moves(player)
    for node, data in G.nodes(data=True):
        if data['player'] == player:
            item = project_seq_onto_player(node, moves)
            if not item in seqs:
                seqs.append(item)
    return sorted(seqs)


def payoff_matrix(G):
    """Computes payoff matrix for two-person zero-sum sequential game."""
    moves1 = player_moves(1)
    moves2 = player_moves(2)
    seqs1 = player_sequences(G, 1, moves=moves1)
    seqs2 = player_sequences(G, 2, moves=moves2)
    A = np.zeros((len(seqs1), len(seqs2)))
    for leaf, data in [x for x in G.nodes(data=True) if is_leaf(G, x[0])]:
        sigma = project_seq_onto_player(leaf, moves2)
        tau = project_seq_onto_player(leaf, moves1)
        A[seqs2.index(sigma), seqs1.index(tau)] += data['payoff']
    return A

G, edge_labels = kuhn3112_gametree()
pos = nx.graphviz_layout(G, prog='dot')

# leaf (terminal) nodes
nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes(
            ) if is_leaf(G, n)], node_shape='s')

# decision nodes
nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes(
            ) if not is_leaf(G, n)], node_shape='o')

# labelled edges
nx.draw_networkx_edges(G, pos, arrows=False)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
pl.show()
