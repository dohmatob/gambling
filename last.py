import itertools
import numpy as np
from nose.tools import assert_equal, assert_true, assert_false

norminal = lambda t: 1 + t % 2
opponent = lambda p: 1 + p % 2
move_id = lambda move, p: move.upper() if p == 1 else move.lower()
singletons = lambda x: itertools.combinations(x, 1)
whose = lambda m: 1 if m == m.upper() else 2


def chance(t):
    if t == 0:
        for perm in itertools.product("JQ", "JQ"):
            yield perm


def ok(a, L):
    return a and a[0] <= L and np.all(a[3] >= .0)


def do_call(a, p):
    a[1] += a[0]
    a[2][p - 1] += a[0]
    a[3][p - 1] -= a[0]
    return a


def do_raise(a, p):
    a[0] *= 2.
    return do_call(a, p)


def copy_a(a):
    return [a[0], a[1], a[2].copy(), a[3].copy()]


def u(a, cmd):
    if cmd.lower() in ['', 'f']:
        return a
    elif cmd[0].lower() == "k":
        do_call(a, whose(cmd[0]))
        return u(a, cmd[1:])
    elif cmd[0].lower() == "r":
        do_raise(a, whose(cmd[0]))
        return u(a, cmd[1:])
    else:
        assert 0


def test_u():
    a = [1, 0., np.array([0, 0]), np.array([100, 100])]
    u(a, "rRrRrRf")
    assert_equal(a[2].sum(), a[1])
    assert_equal(a[0], 64.)
    a = [1, 0., np.array([0, 0]), np.array([100, 100])]
    u(a, "rRrRrRk")
    assert_equal(a[1], 126 + 64)
    assert_equal(a[0], 64.)


def test_opponent():
    assert_equal(opponent(1), 2)
    assert_equal(opponent(2), 1)


def alpha1(t, T, a, L, rep=""):
    if t < T and ok(a, L):
        p = norminal(t)
        q = opponent(p)
        token = move_id("r", q) + move_id("r", p)
        while True:
            for end in singletons([move_id('f', q), move_id('k', q)]):
                end = end[0]
                x = rep + end
                a_ = u(copy_a(a), x)
                if ok(a_, L):
                    return set().union(
                        itertools.product([x], gamma(t + 1, T, a_, L)),
                        alpha1(t, T, a, L, rep=rep + token))
    return ""


def alpha2(t, T, a, L, rep=""):
    if t < T and ok(a, L):
        p = norminal(t)
        q = opponent(p)
        token = move_id("r", q) + move_id("r", p)
        for end in singletons([move_id('f', p), move_id('k', p)]):
            end = end[0]
            x = rep + move_id("r", q) + end
            a_ = u(copy_a(a), x)
            if ok(a_, L):
                return set().union(
                    itertools.product([x], gamma(t + 1, T, a_, L)),
                    alpha2(t, T, a, L, rep + token))
    return ""


def gamma(t, T, a, L, default=None):
    if t >= T or not ok(a, L):
        return ""
    p = norminal(t)
    print alpha1(t, T, a, L)
    return itertools.product(
        chance(t),
        set().union(move_id('f', p),
                    itertools.product(move_id('c', p),
                                      gamma(t + 1, T, a, L)),
                    itertools.product(move_id('k', p),
                                      set().union(alpha1(t, T, a, L),
                                                  alpha2(t, T, a, L)))))


if __name__ == "__main__":
    a = [1, 0., np.array([0, 0]), np.array([100, 100])]
    for x in alpha1(0, 1, a, 1):
        print x
