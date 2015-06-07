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
