import numpy as np

move_id = lambda move, p: move.upper() if p == 1 else move.lower()
norminal = lambda t: 1 + t % 2
opponent = lambda p: 1 + p % 2
whose = lambda move: 1 if move == move.upper() else 2


def chance(t):
    if t == 0:
        for perm in ['JJ', 'JQ', 'QJ', 'QQ']:
            yield perm
    else:
        yield ""


def ok(a, L):
    """Checks that the bank is still sane."""
    return a and a[0] <= L and np.all(a[3] >= .0)


def do_call(a, p):
    """Executes a "call" action."""
    a[1] += a[0]
    a[2][p - 1] += a[0]
    a[3][p - 1] -= a[0]
    return a


def do_raise(a, p):
    """Executes a "raise" action."""
    a[0] *= 2.
    return do_call(a, p)


def copy_a(a):
    """Copies / clones a banks HD."""
    return [a[0], a[1], a[2].copy(), a[3].copy()]


def u(a, cmd):
    """Executes a cmd and updates the bank info."""
    cmd = cmd.split(".")[-1]
    if cmd.lower() in ['', 'f']:
        return a
    elif cmd[0].lower() == "k":
        do_call(a, whose(cmd[0]))
    elif cmd[0].lower() == "r":
        do_raise(a, whose(cmd[0]))
    return u(a, cmd[1:])


def gamma(t, T, a, L):
    """Generates all (admissible) nodes of the game tree."""
    if t < T and ok(a, L):
        p = norminal(t)
        q = opponent(p)
        for perm in chance(t):
            yield perm + move_id("f", p)
            c = move_id("c", p)
            a[0] = 1
            for x in gamma(t + 1, T, a, L):
                yield perm + c + "." + x
            k = move_id("k", p)
            token = move_id("r", q) + move_id("r", p)
            for o, last_sip in zip([p, q], ['', move_id("r", q)]):
                for end in [move_id('f', o), move_id('k', o)]:
                    rep = ""
                    x = perm + k + rep + last_sip + end
                    a_ = u(copy_a(a), x)
                    while ok(a_, L):
                        yield x
                        for y in gamma(t + 1, T, a_, L):
                            yield x + "." + y
                        rep += token
                        x = perm + k + rep + end
                        a_ = u(copy_a(a), x)


if __name__ == "__main__":
    a = [1, 0., np.array([0, 0]), np.array([10, 1])]
    for x in gamma(0, 4, a, 5):
        print x
