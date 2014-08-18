from copy import deepcopy
import numpy as np

move_id = lambda move, p: move.upper() if p == 1 else move.lower()
other_player = lambda p: 1 + p % 2
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


def g(treasury, cmd):
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
