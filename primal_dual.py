# Primal-Dual algorithm for computing Nash equilibria in two-person
# zero-sum games.
#
# author: DOHMATOB Elvis

import warnings
from math import sqrt
import numpy as np
from scipy import linalg


def power(A, p, max_iter=100, **kwargs):
    """Power iteration to estimate spectral radius of a linear operator."""
    b = np.random.randn(p)
    for _ in xrange(max_iter):
        b = A(b, **kwargs)
        d = sqrt(np.dot(b, b))
        if d == 0:
            return d
        b /= d
    lambd = np.dot(b, A(b, **kwargs))
    lambd /= np.dot(b, b)
    return lambd


def evil(L, r, z, proj_C=lambda v: np.maximum(v, 0.), L_norm=None,
         max_iter=100, verbose=1, tol=1e-4):
    """Projects the point z onto the intersection of the polyhedron "Lx=r"
    and the convex set C.

    References
    ----------
    P. Combettes et al, "Dualization of Signal Recovery Problems", p.17
    """
    if L_norm is None:
        L_norm = linalg.svdvals(L)[0]
    gamma = .99 / L_norm ** 2
    v = np.zeros(L.shape[0])
    x = np.zeros(L.shape[1])
    salt = 1e-4
    for k in xrange(max_iter):
        old_x = x.copy()
        x = proj_C(z - L.T.dot(v)) + salt
        v += gamma * (L.dot(x) - r + salt)
        salt *= .5  # we want: \sum_{k}{salt_k} < \infty
        error = sqrt(((x - old_x) ** 2).sum())
        if verbose:
            print "\tIteration %03i/%03i: x=%s, error=%5.2e" % (
                k + 1, max_iter, x, error)
        if error < tol and 1:
            if verbose:
                print "\tConverged after %i iterations." % (k + 1)
            break
    return x


def primal_dual_ne(A, E1, E2, e1, e2, proj_C1=lambda x: np.maximum(x, 0.),
                   proj_C2=lambda y: np.maximum(y, 0.), init=None,
                   tol=1e-8, max_iter=1000, inertia=None, rho=None,
                   good_init=False):
    """Primal-Dual algorithm for computing Nash equlibrium for two-person
    zero-sum game with payoff matrix A and contraint sets

        Qj := {z in Cj | Ejz = ej}

    The formal problem is:

        minimize maximize <x, Ay>
        y in Q2  x in Q1
    """
    # sanitize rho and inertia
    inertia = inertia if not inertia is None else 0
    rho = rho if not rho is None else 1.

    if not 0 <= inertia < 1. / 3:
        raise ValueError(
            "inertia param must be in the interval [0, 1 / 3); got %g" % (
                inertia))
    if not 0 < rho < 2.:
        raise ValueError(
            "rho param must be in the interval (0, 1); got %g" % (
                rho))

    # note that there is no convergence theory for inertia combined with
    # over-relaxation; so we'll avoid it
    if inertia:
        warnings.warn(
            "Disabling over-relaxation, since inertia = %g > 0." % inertia)
        rho = 1.  # disable over-relaxation

    n1, n2 = A.shape
    k1, k2 = E1.shape[0], E2.shape[0]

    # misc
    if init is None:
        init = {}
    L = init.get("L")

    # compute spectral norm of linear operator K
    if L is None:
        def K(z):
            y = z[:n2]
            p = z[n2:]
            tmp = A.dot(y) - E1.T.dot(p), E2.dot(y)
            return np.append(A.T.dot(tmp[0]) + E2.T.dot(tmp[1]),
                             -E1.dot(tmp[0]))

        L = sqrt(power(K, n2 + k1))
    print "||K|| = %g" % L

    # initialize params
    sigma = tau = .99 / L
    p = np.zeros(k1)
    q = np.zeros(k2)
    if "xy" in init:
        x, y = init.get("xy")
    else:
        x = np.zeros(n1)
        y = np.zeros(n2)
        ytilde = np.zeros_like(y)
        ptilde = np.zeros_like(p)
        if good_init:
            x = evil(E1, e1, x, verbose=0)
            y = evil(E2, e2, y, verbose=0)
            ytilde = y.copy()
            ptilde = y.copy()

    # main loop
    old_x = old_y = old_q = old_p = None
    values = []
    dgaps = []
    for k in xrange(max_iter):
        # prepare inertia
        a, b, c, d = y.copy(), p.copy(), x.copy(), q.copy()
        if k and inertia:
            a += inertia * (y - old_y)
            b += inertia * (p - old_p)
            c += inertia * (x - old_x)
            d += inertia * (q - old_q)

        # backup variables
        old_y = y.copy()
        old_p = p.copy()
        old_x = x.copy()
        old_q = q.copy()

        # update y and p
        y = A.T.dot(c) + E2.T.dot(d)
        y *= -sigma
        y += a
        y = proj_C2(y)
        p = e1 - E1.dot(c)
        p *= -sigma
        p += b

        # update ytilde and ptilde
        ytilde = 2 * y - a
        ptilde = 2 * p - b

        # update x and q
        x = A.dot(ytilde) - E1.T.dot(ptilde)
        x *= tau
        x += c
        x = proj_C1(x)
        q = E2.dot(ytilde) - e2
        q *= tau
        q += d

        # over-relaxation
        if rho != 1.:
            y = rho * y + (1. - rho) * old_y
            p = rho * p + (1. - rho) * old_p
            x = rho * x + (1. - rho) * old_x
            q = rho * q + (1. - rho) * old_q

        # check convergence
        value = x.T.dot(A.dot(y))
        values.append(value)
        dgap = np.dot(e1, p) + np.dot(e2, q)  # see ref [1], page 235, line 5
        dgaps.append(dgap)
        print ("Primal-Dual iter %i/%i: value of game = %g,"
               " dual gap = %.2e") % (k + 1, max_iter, value, dgap)

        # check convergence; note that dgap is can be a small negative number,
        # negative perhaps because currents are not yet feasible
        if abs(dgap) < tol:
            print "\tConverged after %i iterations." % (k + 1)
            break

    return x, y, values, dgaps


def primal_dual_sg_ne(A, **kwargs):
    n1, n2 = A.shape
    E1 = np.ones((1, n1))
    E2 = np.ones((1, n2))
    e1 = e2 = 1.
    x0 = 1. * np.ones(A.shape[0]) / A.shape[0]
    y0 = 1. * np.ones(A.shape[1]) / A.shape[1]
    return primal_dual_ne(A, E1, E2, e1, e2, init=dict(xy=(x0, y0)),
                          **kwargs)


def array2tex(x, bars=False, brace="(", form=None, col_names=None,
              row_names=None, omit_zero=False):
    def _omit_zero(a):
        return " " if omit_zero and float(a) == 0. else a

    from math import floor
    if not col_names is None:
        bars = True
        col_names = " & ".join([str(cn) for cn in col_names])
    if row_names is None and not col_names is None:
        row_names = [""] * len(x)
    if form is None:
        form = "%s"
    if brace == "(":
        closing_brace = ")"
    elif brace == "[":
        closing_brace = "]"
    cline = "c" * x.shape[1]
    if bars:
        cline = "|%s|" % ("|".join(cline))
    if col_names:
        cline  = "c%s" % cline
    out = "%s\\begin{array}{%s}\n" % ("\\left" + brace if brace else "", cline)
    if not col_names is None:
        out += "%s\\\\%s\n" % (col_names, "\\hline" if bars else "")
    out += ("\\\\%s\n" % ("\\hline" if bars else "")).join(
        [" & ".join([_omit_zero(str(int(a)) if floor(a) == a else
                                ("%s" % form) % a) for a in y]) for y in x])
    if not row_names is None:
        tmp = out.split("\n")
        out = "\n".join(tmp[:2] + ["%s & %s" % (row_name, o)
                                   for row_name, o in zip(row_names, tmp[2:])])
    elif not col_names is None:
        out = "\n".join(["& %s" % o for o in out.split("\n")])
    out += "\n\\end{array}%s\n" % (
        "\\right%s" % closing_brace if brace else "")
    return out


def scientific2tex(x):
    if not isinstance(x, basestring):
        x = "%.2e" % x
    tmp = x.split("e")
    assert len(tmp) == 2
    return "%s \\times 10^{%d}" % (tmp[0], int(tmp[1]))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # build the game
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    E1 = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    e1 = np.eye(E1.shape[0])[0]
    E2 = np.array([[1., 0., 0.], [-1., 1., 1.]])
    e2 = np.eye(E2.shape[0])[0]

    # solve the game
    x, y, values = primal_dual_ne(A, E1, E2, e1, e2)
    print
    print "Nash Equilibrium:"
    print "x* = ", x
    print "y* =", y
    print "E1x^* - e1 = %s" % (E1.dot(x) - e1)
    print "E2y^* - e2 = %s" % (E2.dot(y) - e2)
    plt.semilogx(values)
    plt.axhline(values[-1], linestyle="--")
    plt.xlabel("iteration (k)")
    plt.ylabel("value of game after kth iterations")
    plt.show()
