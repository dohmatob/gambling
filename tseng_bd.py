from math import sqrt
import numpy as np
from scipy import linalg

_norm = lambda *args: sqrt(np.sum(np.concatenate(args) ** 2))


def tseng_bd(K, prox_g1, prox_g2, init=None, max_iter=1000,
             tol=1e-6, callback=None):
    if init is None: init = {}
    norm_K = init.get("norm_K", None)
    init["norm_K"] = norm_K
    sigma = init.get("sigma", 1.)
    init["sigma"] = sigma
    q, p = K.shape
    if norm_K is None: norm_K = linalg.norm(K, 2)
    lambd = init.get("lambd", .9 * sigma / norm_K)
    init["lambd"] = lambd
    w = init.get("w", np.zeros(p))
    z = init.get("z", np.zeros(q))

    # main loop
    gaps = []
    delta_w_sum = np.zeros_like(w)
    delta_z_sum = np.zeros_like(z)
    for k in range(max_iter):
        # update vars
        old_w = w.copy()
        old_z = z.copy()
        w -= lambd * K.T.dot(z)
        w = prox_g1(w, lambd)
        z += lambd * K.dot(w)
        z = prox_g2(z, lambd)
        w -= lambd * K.T.dot(z - old_z)

        # compute gap
        delta_w_sum += w - old_w
        delta_z_sum += z - old_z
        gap = _norm(delta_w_sum, delta_z_sum) / lambd / (k + 1.)
        gaps.append(gap)

        # invoke callback
        if callback: callback(locals())
        print "Iter %03i/%03i: gap=%.2e" % (k + 1, max_iter, gap)
        if gap < tol:
            print "Converged (gap tol = %.2e)." % tol
            break

    # return goodies
    init["w"] = w.copy()
    init["z"] = z.copy()
    return w, z, init, np.array(gaps)


def primal_dual_ne(A, E1, E2, e1, e2, proj_C1=lambda x: np.maximum(x, 0.),
                   proj_C2=lambda y: np.maximum(y, 0.), init=None, tol=1e-10,
                   max_iter=10000):
    """Primal-Dual algorithm for computing Nash equlibrium for two-person
    zero-sum game with payoff matrix A and contraint sets

        Qj := {z in Cj | Ejz = ej}

    The formal problem is:

        minimize maximize <x, Ay>
        y in Q2  x in Q1
    """
    n1, n2 = A.shape
    k1, k2 = E1.shape[0], E2.shape[0]
    zeros = np.zeros((k2, k1))
    K = np.vstack((np.hstack((A, -E1.T)), np.hstack((E2, zeros))))
    values = []

    def prox_g1(w, lambd):
        w[:n2] = proj_C2(w[:n2])
        w[n2:] -= lambd * e1
        return w

    def prox_g2(z, lambd):
        z[:n1] = proj_C1(z[:n1])
        z[n1:] -= lambd * e2
        return z

    def callback(variables):
        y = variables["w"][:n2]
        x = variables["z"][:n1]
        value = x.dot(A.dot(y))
        values.append(value)

    w, z, init, gaps = tseng_bd(K, prox_g1, prox_g2, max_iter=max_iter,
                                init=init, tol=tol, callback=callback)
    y, p = w[:n2], w[n2:]
    x, q = z[:n1], z[n1:]
    return x, y, p, q, init, values, gaps


def primal_dual_sg_ne(A, **kwargs):
    n1, n2 = A.shape
    E1 = np.ones((1, n1))
    E2 = np.ones((1, n2))
    e1 = e2 = 1.
    return primal_dual_ne(A, E1, E2, e1, e2, **kwargs)
