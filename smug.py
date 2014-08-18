import itertools
import pylab as pl
import networkx as nx

T = nx.DiGraph()
edge_labels = {}


def is_leaf(T, node):
    """Checks whether given node is leaf / terminal."""
    return T.out_degree(node) == 0.


def _add_labelled_edge(x, y, **kwargs):
    move = y.split(".")[-1]
    T.add_node(y, **kwargs)
    T.add_edge(x, y)
    edge_labels[(x, y)] = move

proba = "1/6"
for z in itertools.permutations('123', 2):
    _add_labelled_edge('/', '/.%s' % ''.join(z), proba=proba)

pos = nx.graphviz_layout(T, prog='dot')


# BFS"
info = {1: {}, 2: {}}
source = "/"
info[1] = {}

for node in T.nodes_iter():
    if node == "/" or is_leaf(T, node):
        continue
    if node[-1] in '123cfkr':
        player = 1
    else:
        player = 2
    signal = re.match('^/\.(.)', node).group(1)
    skeleton = project_seq_onto_player(node,
                                       player_moves(player))
    moves = sorted([succ[-1] for succ in T.successors(node)])
    key = signal, tuple(skeleton), tuple(moves)
    if not key in info[player]:
        info[player][key] = dict(nodes=[])
    info[player][key]['nodes'].append(node)

for z in '123':
    print [node for node in nodes if node.startswith('/.%s' % z)]

# leaf (terminal) nodes
nx.draw_networkx_nodes(T, pos, nodelist=[n for n in T.nodes(
            ) if is_leaf(T, n)], node_shape='s')

# decision nodes
nx.draw_networkx_nodes(T, pos, nodelist=[n for n in T.nodes(
            ) if not is_leaf(T, n)], node_shape='o')

# labelled edges
nx.draw_networkx_edges(T, pos, arrows=False)
nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels)
pl.show()
