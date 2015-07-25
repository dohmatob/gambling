"""
Primal-Dual algorithm for computing Nash equilibria in two-person
zero-sum games.

"""
# author: Elvis Dohmatob <gmdopp@gmail.com>

from math import sqrt
import numpy as np
from scipy import linalg

# norm of the concatenation of vectors
_norm = lambda *args: sqrt(np.sum(np.concatenate(args) ** 2))


def _check_subgradient(f, x, g, n=10):
    """Checks that g is a sub-gradient of f at x."""
    from sklearn.utils import check_random_state
    rng = check_random_state(42)
    p = len(x)
    for z in rng.randn(n, p):
        if f(z) < f(x) + np.dot(g, z - x): return False
    return True


def primal_dual_ne(A, E1, E2, e1, e2, proj_C1=lambda x: np.maximum(x, 0.),
                   proj_C2=lambda y: np.maximum(y, 0.), init=None,
                   epsilon=1e-4, max_iter=10000, callback=None):
    """Primal-Dual algorithm for computing Nash equlibrium for two-person
    zero-sum game with payoff matrix A and contraint sets

        Qj := {z in Cj | Ejz = ej}

    The formal problem is:

        minimmize maximize <x, Ay>
         x in Q1   y in Q2

    Notes
    -----
    \epsilon really stands for \rho in the paper.

    Returns
    -------
    """
    # misc
    if init is None: init = {}
    n1, n2 = A.shape
    l1, l2 = E1.shape[0], E2.shape[0]
    zeros = np.zeros((l2, l1))
    if "norm_K" in init: norm_K = init["norm_K"]
    else:
        # XXX use power iteration to compute ||K||^2
        K = np.vstack((np.hstack((A, -E1.T)), np.hstack((E2, zeros))))
        norm_K = linalg.norm(K, 2)
    sigma = init.get("sigma", 1.)
    lambd = init.get("lambd", .9 * sigma / norm_K)
    y = init.get("y", np.zeros(n2))
    p = init.get("p", np.zeros(l1))
    x = init.get("x", np.zeros(n1))
    q = init.get("q", np.zeros(l2))
    delta_y_sum = np.zeros_like(y)
    delta_p_sum = np.zeros_like(p)
    delta_x_sum = np.zeros_like(x)
    delta_q_sum = np.zeros_like(q)
    values = []
    gaps = []

    # main loop
    for k in range(max_iter):
        # invoke callback
        if callback and callback(locals()): break

        # save previous iterates
        old_y = y.copy()
        old_p = p.copy()
        old_x = x.copy()

        # y update
        y -= lambd * (A.T.dot(x) + E2.T.dot(q))
        y = proj_C2(y)

        # p update
        p -= lambd * (e1 - E1.dot(x))

        # x updata
        x += lambd * (A.dot(y) - E1.T.dot(p))
        x = proj_C1(x)
        delta_x = x - old_x

        # q update
        delta_q = lambd * (E2.dot(y) - e2)
        q += delta_q

        # u update (again)
        y -= lambd * (A.T.dot(delta_x) + E2.T.dot(delta_q))

        # p update (again)
        p += lambd * E1.dot(delta_x)

        # compute game value and primal-dual gap at current iterates
        value = x.dot(A.dot(y))
        values.append(value)
        delta_y_sum += y - old_y
        delta_p_sum += p - old_p
        delta_x_sum += delta_x
        delta_q_sum += delta_q
        gap = _norm(delta_y_sum, delta_p_sum, delta_x_sum,
                    delta_q_sum) / lambd / (k + 1.)
        gaps.append(gap)

        # check convergence
        print ("Iter %03i/%03i: game value = %g, primal-dual gap (perturbed) ="
               " %.2e") % (k + 1, max_iter, value, gap)
        if gap < epsilon:
            print "Converged (gap < gap epsilon = %.2e)." % epsilon
            break

    # misc
    init["norm_K"] = norm_K
    init["sigma"] = sigma
    init["lambd"] = lambd
    init["y"] = y.copy()
    init["p"] = p.copy()
    init["x"] = x.copy()
    init["q"] = q.copy()

    return x, y, p, q, init, values, gaps


def primal_dual_sg_ne(A, epsilon=1e-4, strict=True, **kwargs):
    n1, n2 = A.shape
    E1 = np.ones((1, n1))
    E2 = np.ones((1, n2))
    e1 = 1.
    e2 = 1.

    init = kwargs.pop("init") if "init" in kwargs else {}
    if not "x" in init: init["x"] = (1. / n1) * np.ones(n1)
    if not "y" in init: init["y"] = (1. / n2) * np.ones(n2)
    kwargs["init"] = init

    gaps = []

    def cb(variables):
        gap = A.dot(variables["y"]).max() - A.T.dot(variables["x"]).min()
        gap = abs(gap)  # maybe gap < 0 in case we aren't on the feasible set
        print gap
        gaps.append(gap)
        return strict and gap < epsilon

    x, y, p, q, init, values, gaps_ = primal_dual_ne(
        A, E1, E2, e1, e2, callback=cb, epsilon=epsilon, **kwargs)

    if strict: return x, y, values, gaps
    else: return x, y, p, q, init, values, gaps_
