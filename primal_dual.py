"""
A simple algorithm for computing Nash-equilibria in incomplete
information games.

NIPS OPT2016 workshop.

"""
# author: Elvis Dohmatob <gmdopp@gmail.com>

import numbers
from math import sqrt
import numpy as np
from scipy import linalg

# norm of the concatenation of vectors
_norm = lambda *args: sqrt(np.sum(np.concatenate(args) ** 2))


class K(object):
    def __init__(self, A, E1, E2, L=None):
        self.A = A
        self.E1 = E1
        self.E2 = E2
        self.n1, self.n2 = A.shape
        self.l1, self.l2 = E1.shape[0], E2.shape[0]
        zeros = np.zeros((self.l2, self.l1))
        self.mat = np.vstack((np.hstack((A, -E1.T)), np.hstack((E2, zeros))))
        self.L = linalg.norm(self.mat, 2) if L is None else L

    def __call__(self, yp):
        return self.mat.dot(yp)

    @property
    def T(self):
        return K(self.A.T, -self.E2, -self.E1, L=self.L)


def _check_subgradient(f, x, g, n=10):
    """Checks that g is a sub-gradient of f at x."""
    from sklearn.utils import check_random_state
    rng = check_random_state(42)
    p = len(x)
    for z in rng.randn(n, p):
        z += x
        print f(z), f(x) + np.dot(g, z - x)
        if f(z) < f(x) + np.dot(g, z - x):
            return False
    return True


class _GSPEnergy(object):
    """This object is useful for checking
    v \in \partial [Phi_1(., ., x_, q_) + Phi_2(y_, p_, ., .)](y_, p_, x_, q_)
    """
    def __init__(self, K, e1, e2, y_, p_, x_, q_):
        self.K = K
        self.e1 = np.array([e1]) if isinstance(e1, numbers.Number) else e1
        self.e2 = np.array([e2]) if isinstance(e2, numbers.Number) else e2
        self.y_ = y_
        self.p_ = p_
        self.x_ = x_
        self.q_ = q_
        self.l1_ = len(self.e1)
        self.l2_ = len(self.e2)
        self.n2_ = K.shape[0] - self.l1_
        self.n1_ = K.shape[1] - self.l2_
        self.yp_ = np.concatenate((y_, p_))
        self.xq_ = np.concatenate((x_, q_))

    def __call__(self, ypxq):
        yp = ypxq[:self.K.shape[1]]
        y, p = yp[:self.n2_], yp[self.n2_:]
        y = np.maximum(0., y)
        yp = np.concatenate((y, p))
        xq = ypxq[self.K.shape[1]:]
        x, q = xq[:self.n1_], xq[self.n1_:]
        x = np.maximum(0., x)
        xq = np.concatenate((x, q))
        if not np.all(y >= 0.) or not np.all(x >= 0.):
            return np.inf
        a = self.xq_.dot(self.K.dot(yp)) + self.e1.dot(p)
        b = -xq.dot(self.K.dot(self.yp_)) + self.e2.dot(q)
        return a + b


def primal_dual_ne(A, E1, E2, e1, e2, proj_C1=lambda x: np.maximum(x, 0.),
                   proj_C2=lambda y: np.maximum(y, 0.), init=None,
                   epsilon=1e-4, max_iter=10000, check_ergodic=True,
                   callback=None):
    """Projection-free primal-Dual algorithm for computing Nash equlibrium
    for two-person zero-sum game with payoff matrix A and contraint sets

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
    if init is None:
        init = {}
    n1, n2 = A.shape
    l1, l2 = E1.shape[0], E2.shape[0]
    zeros = np.zeros((l2, l1))
    if "norm_K" in init:
        norm_K = init["norm_K"]
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
    k = None
    deltas_y = []
    deltas_p = []
    deltas_x = []
    deltas_q = []
    y_avg = np.zeros_like(y)
    x_avg = np.zeros_like(x)
    k = 1
    for k in range(max_iter):
        # invoke callback
        if callback and callback(locals()):
            break

        # save previous iterates
        old_y = y.copy()
        old_p = p.copy()
        old_x = x.copy()

        # y update
        y -= lambd * (A.T.dot(x) + E2.T.dot(q))
        y = proj_C2(y)

        # p update
        p -= lambd * (e1 - E1.dot(x))
        print e1 - E1.dot(x)

        # x updata
        x += lambd * (A.dot(y) - E1.T.dot(p))
        x = proj_C1(x)
        delta_x = x - old_x
        x_avg += x
        if check_ergodic:
            deltas_x.append(delta_x)

        # q update
        delta_q = lambd * (E2.dot(y) - e2)
        q += delta_q
        if check_ergodic:
            deltas_q.append(delta_q)

        # u update (again)
        y -= lambd * (A.T.dot(delta_x) + E2.T.dot(delta_q))
        delta_y = y - old_y
        y_avg += y
        if check_ergodic:
            deltas_y.append(delta_y)

        # p update (again)
        p += lambd * E1.dot(delta_x)
        delta_p = p - old_p
        if check_ergodic:
            deltas_p.append(delta_p)

        # compute game value and primal-dual gap at current iterates
        value = x.dot(A.dot(y))
        values.append(value)
        delta_y_sum += delta_y
        delta_p_sum += delta_p
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
    y = y_avg / float(k)
    x = x_avg / float(k)
    y = proj_C2(y)
    x = proj_C1(x)
    init["norm_K"] = norm_K
    init["sigma"] = sigma
    init["lambd"] = lambd
    init["y"] = y.copy()
    init["p"] = p.copy()
    init["x"] = x.copy()
    init["q"] = q.copy()

    # Check that we produced a small pertubation vector in the subgradient
    # of the GSP energy at the claimed (approx) equilibrium point.
    if check_ergodic:
        gsp_energy = _GSPEnergy(K, e1, e2, y, p, x, q)
        print "Checking subgradient optimality condition for some iterate..."
        for k, deltas in enumerate(zip(deltas_y, deltas_p, deltas_x,
                                       deltas_q)):
            v = np.concatenate(deltas)
            if _check_subgradient(gsp_energy, np.concatenate((y, p, x, q)), v,
                                  23) and _norm(v) <= epsilon:
                print "OK (k = %i)." % k
                break
        else:
            print "Ergodicity check failed."

    return x, y, p, q, init, values, gaps


def primal_dual_sg_ne(A, epsilon=1e-4, strict=True, **kwargs):
    """Specialization on classical matrix games on simplexes."""
    # misc
    n1, n2 = A.shape
    E1 = np.ones((1, n1))
    E2 = np.ones((1, n2))
    e1 = 1.
    e2 = 1.
    init = kwargs.pop("init") if "init" in kwargs else {}
    if "x" not in init:
        init["x"] = (1. / n1) * np.ones(n1)
    if "y" not in init:
        init["y"] = (1. / n2) * np.ones(n2)
    kwargs["init"] = init
    gaps = []

    def cb(variables):
        gap = A.dot(variables["y"]).max() - A.T.dot(variables["x"]).min()
        gap = abs(gap)  # maybe gap < 0 in case we aren't on the feasible set
        gaps.append(gap)
        return strict and gap < epsilon

    x, y, p, q, init, values, gaps_ = primal_dual_ne(
        A, E1, E2, e1, e2, callback=cb, epsilon=epsilon, **kwargs)

    if strict:
        return x, y, values, gaps
    else:
        return x, y, p, q, init, values, gaps_
