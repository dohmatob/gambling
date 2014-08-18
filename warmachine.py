from copy import deepcopy
import re
import networkx as nx
import numpy as np

other_player = lambda p: 1 + p % 2
player = lambda rnd: 1 + rnd % 2
move_id = lambda move, p: move.upper() if p == 1 else move.lower()
whose = lambda m: 1 if m == m.upper() else 2


def clone(obj):
    """Quick and dirty technology for cloning an object instance."""
    c = object.__new__(obj.__class__)
    c.__dict__ = deepcopy(obj.__dict__)
    return c


class Treasury(object):
    def __init__(self, capitals, bet_size=1, limit=np.inf):
        self.limit = limit
        self.bet_size = bet_size
        self.pot = 0.
        self.potted = np.zeros(2)
        self.accounts = np.array(capitals)

    def do_call(self, p):
        self.pot += self.bet_size
        self.potted[p - 1] += self.bet_size
        self.accounts[p - 1] -= self.bet_size
        # if np.any(self.accounts < 0.):
        #     raise RuntimeError("Some players have negative accounts!")
        assert self.potted.sum() == self.pot

    def do_raise(self, p):
        self.bet_size *= 2
        # if self.bet_size > self.limit:
        #     raise RuntimeError(
        #         "bet_size exceeded limmit of %g$" % self.limit)
        self.do_call(p)

    def __dict__(self):
        return dict((k, getattr(self, k))
                    for k in ["bet_size", "pot", "potted", "accounts"])

    def __repr__(self):
        return str(self.__dict__())

    def ok(self, limit):
        if np.any(self.accounts <= 0.):
            return False
        elif self.bet_size > limit:
            return False
        else:
            return True

    def copy(self):
        t = Treasury(self.accounts.copy(), bet_size=self.bet_size)
        t.potted = self.potted.copy()
        t.pot = self.pot
        return t


def g(treasury, cmd):
    treasury = treasury.copy()
    if np.any(treasury.accounts <= 0) or treasury.bet_size > treasury.limit:
        return None
    elif not cmd or cmd[0].lower() == "f":
        return treasury
    elif cmd[0].lower() == "k":
        treasury.do_call(whose(cmd[0]))
        return g(treasury, cmd[1:])
    elif cmd[0].lower() == "r":
        treasury.do_raise(whose(cmd[0]))
        return g(treasury, cmd[1:])
    else:
        assert 0


def process_rnd(s):
    assert isinstance(s, str)

    if not s:
        return 0.

    if s.lower() in ["c", "f"]:
        return 0.

    if s[0] == s[0].lower():
        m = re.match("^k((?:Rr)*)[KF]$", s)
        if m is None:
            m = re.match("^k((?:Rr)*R)[kf]$", s)
            assert m
    else:
        m = re.match("^K((?:rR)*)[kf]$", s)
        if m is None:
            m = re.match("^K((?:rR)*r)[KF]$", s)
            assert m
    a = len(m.group(1))

    res = 2 ** (a + 1) - 1.
    if s.lower().endswith('k'):
        res += 2 ** a
    return res


def process_game(s):
    if isinstance(s, str):
        s = s.split(".")
    if not s:
        return 0.
    return process_rnd(s[0]) + process_game(s[1:])


def gen_leafs(treasury, rnd, max_rnds, limit, parent=""):
    # at terminal node ?
    if parent.lower().endswith("f"):
        yield ""
        return
    if rnd >= max_rnds:
        return
    if not treasury:
        return
    if not treasury.ok(limit):
        return

    p = player(rnd)
    q = other_player(p)
    yield move_id("f", p)
    c = move_id("c", p)
    for s in gen_leafs(treasury, rnd + 1, max_rnds, limit,
                       parent=parent + c):
        yield c + ("." + s if s else "")
    k = move_id("k", p)
    for j in xrange(1 + limit // 2):
        rep = move_id("r", q) + move_id("r", p)
        rep *= j
        for end in "fk":
            end = move_id(end, q)
            res = k + rep + end
            for s in gen_leafs(g(treasury, res), rnd + 1, max_rnds, limit,
                               parent=parent + res):
                yield res + ("." + s if s else "")
    for j in xrange(1 + (limit - 1) // 2):
        rep = move_id("r", q) + move_id("r", p)
        rep *= j
        rep += move_id("r", q)
        for end in "fk":
            end = move_id(end, p)
            res = k + rep + end
            for s in gen_leafs(g(treasury, res), rnd + 1, max_rnds, limit,
                               parent=parent + res):
                yield res + ("." + s if s else "")


if __name__ == "__main__":
    import pylab as pl
    # import numpy as np
    max_rnds = 1
    # f = lambda limit: len(list(gen_leafs(0, max_rnds, limit)))
    # g = lambda limit: limit ** max_rnds
    # limits = np.arange(30)
    nodes = []
    treasury = Treasury([1000, 1000])
    for s in gen_leafs(treasury, 0, max_rnds, 2):
        nodes.append("/" + s.replace(".", ""))
        print 'process_game("%s") = %5.2e' % (s, process_game(s))
    tree = nx.DiGraph()
    for node in nodes:
        for j in xrange(1, len(node)):
            a, b = node[:j], node[:j + 1]
            a, b = ".".join(list(a)), ".".join(list(b))
            e = a, b
            if e in tree.edges():
                continue
            print "\t%s -> %s" % (a, b)
            tree.add_edge(a, b)

    is_leaf = lambda node: tree.out_degree(node) == 0.
    is_root = lambda node: node == "/"
    pos = nx.graphviz_layout(tree,
                             # prog="dot"
                             )
    leaf_nodes = [n for n in tree.nodes() if is_leaf(n)]
    decision_nodes = [n for n in tree.nodes() if is_leaf(n)]
    _node_color = lambda node: "b" if is_root(node) else (
        "g" if node[-1].lower() == node[-1] else "r")
    # nx.draw_networkx_nodes(tree, pos,
    #                        # node_size=30,
    #                        nodelist=leaf_nodes,
    #                        node_shape='s',
    #                        # node_color=map(_node_color, leaf_nodes)
    #                        )
    # nx.draw_networkx_nodes(tree, pos,
    #                        # node_size=30,
    #                        node_list=decision_nodes,
    #                        node_shape='o',
    #                        # node_color=map(_node_color, decision_nodes)
    #                        )
    nx.draw_networkx_nodes(tree, pos,
                           node_color=map(_node_color, tree.nodes()),
                           node_size=30
                           )
    nx.draw_networkx_edges(tree, pos, arrows=False)

    # pl.semilogy([f(l) for l in limits], label="true")
    # pl.semilogy([g(l) for l in limits], label="estimated")
    # pl.legend(loc="best")
    # pl.xlabel("limit")
    pl.axis("off")
    pl.tight_layout()
    pl.show()
