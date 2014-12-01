# Primal-Dual algorithm for computing Nash equilibria in two-person
# zero-sum games.
#
# author: DOHMATOB Elvis

from math import sqrt
import numpy as np


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


def primal_dual_ne(A, E1, E2, e1, e2, L=None, max_iter=10000, tol=1e-4):
    """Primal-Dual algorithm for computing Nash equlibrium for two-person
    zero-sum game with payoff matrix A and contraint sets

        Qj := {z | z = (z)_+, Ejz = ej

    The formal problem is:

        minimize maximize <x, Ay>
        y in Q2  x in Q1
    """
    n1, n2 = A.shape
    p1, p2 = E1.shape[0], E2.shape[0]

    # compute spectral norm of linear operator K
    if L is None:
        def K(z):
            y = z[:n2]
            v = z[n2:]
            tmp = A.dot(y) - E1.T.dot(v), E2.dot(y)
            return np.append(A.T.dot(tmp[0]) + E2.T.dot(tmp[1]),
                             -E1.dot(tmp[0]))

        L = sqrt(power(K, n2 + p1))
    print "||K|| = %g" % L

    # initialize params
    sigma = tau = .99 / L
    x = np.zeros(n1)
    u = np.zeros(p2)
    y = np.zeros(n2)
    v = np.zeros(p1)
    ytilde = np.zeros_like(y)
    vtilde = np.zeros_like(v)

    # main loop
    values = []
    for k in xrange(max_iter):
        # backup variables
        old_y = y.copy()
        old_v = v.copy()
        old_x = x.copy()
        old_u = u.copy()

        # update x and u
        x += tau * (A.dot(ytilde) - E1.T.dot(vtilde))
        x = np.maximum(x, 0.)
        u += tau * (E2.dot(ytilde) - e2)

        # update y and v
        y -= sigma * (A.T.dot(x) + E2.T.dot(u))
        y = np.maximum(y, 0.)
        v -= sigma * (e1 - E1.dot(x))

        # update ytilde and vtilde
        ytilde = 2 * y - old_y
        vtilde = 2 * v - old_v

        # check convergence
        value = x.T.dot(A.dot(y))
        values.append(value)
        error = sqrt(((((x - old_x) ** 2).sum() + (
            u - old_u) ** 2).sum()) / tau + (((y - old_y) ** 2).sum() + (
                (v - old_v) ** 2).sum() / sigma))
        print ("Primal-Dual iter %i/%i: value of game = %g,"
               " elliptic error = %.2e") % (k + 1, max_iter, value, error)
        if error < tol:
            print "\tConverged after %i iterations." % (k + 1)
            break
    return x, y, values


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
