from math import sqrt
import numpy as np
from scipy import linalg


def devil(L, r, z, max_iter=100, verbose=0, tol=1e-4):
    q, n = L.shape
    l = linalg.svdvals(np.vstack((np.eye(n), L)))[0]
    sigma = tau = .99 / l
    assert sigma * tau * l * l < 1.
    x = np.zeros(n)
    xbar = x.copy()
    v = np.zeros(n)
    zeta = np.zeros(q)
    for k in xrange(max_iter):
        old_x = x.copy()
        old_v = v.copy()
        old_zeta = zeta.copy()

        v += sigma * (xbar - z)
        v /= (1. + sigma)
        zeta += sigma * (L.dot(xbar) - r)
        x -= tau * (v + L.T.dot(zeta))
        x = np.maximum(x, 0.)
        xbar = 2 * x - old_x

        error = .5 * (((x - old_x) ** 2).sum() / tau + (
                ((v - old_v) ** 2).sum() + (
                    (zeta - old_zeta) ** 2).sum()) / sigma)
        if verbose:
            print "\tIteration: %03i/%03i: error=%g" % (k + 1, max_iter, error)

        if error < tol:
            if verbose:
                print "\tConverged after %i iterations." % (k + 1)
            break

    return x


def general_condat(L, grad_F, beta, prox_G, prox_Hstar, norm_L=None,
                   max_iter=100, verbose=0, tol=1e-4):

    if norm_L is None:
        norm_L = linalg.svdvals(L)[0]
    sigma = .99 / norm_L
    tau = .9 / (sigma * norm_L * norm_L + .5 * beta)
    assert 1. / tau - sigma * norm_L * norm_L >= beta * .5
    print norm_L, beta, sigma, tau
    x = np.zeros(A.shape[1])
    y = np.zeros(A.shape[0])
    ybar = y.copy()
    delta = 2. - .5 * beta / (1. / tau - sigma * norm_L * norm_L)
    assert 1. <= delta <= 2.
    rho = .99 * delta
    for _ in xrange(max_iter):
        old_x = x.copy()
        old_y = y.copy()

        xbar = x - tau * (grad_F(x) + A.T.dot(y))
        xbar = prox_G(xbar, tau)
        ybar = y + sigma * A.dot(2 * xbar - x)
        ybar = prox_Hstar(ybar, sigma)
        x = rho * xbar + (1. - rho) * old_x
        y = rho * ybar + (1. - rho) * old_y
        if verbose:
            print x, y


def evil(L, r, z, proj_C=lambda v: np.maximum(v, 0.), gamma=None, max_iter=100,
         verbose=1, tol=1e-4):
    """Projects the point z onto the intersection of the hyperplane "Ex=r" and
    the convex set C.
    """
    if gamma is None:
        gamma = 2. / linalg.svdvals(L)[0] ** 2
    v = np.zeros(L.shape[0])
    x = np.zeros(L.shape[1])
    for k in xrange(max_iter):
        old_x = x.copy()
        x = proj_C(z - L.T.dot(v))
        v += gamma * (L.dot(x) - r)
        error = ((x - old_x) ** 2).sum()
        if verbose:
            print ("\tIteration: %03i/%03i: x^(k-1) = %s; x^(k) = %s; "
                   "||x^(k) - x^(k-1)||^2 = %s" % (k + 1, max_iter, old_x,
                                                   x, error))
        if error < tol:
            if verbose:
                print "\tConverged after %i iterations." % (k + 1)
            break
    return x


def compute_ne(A, P, Q, a, b, max_iter=1000, callback=None, tol=1e-4,
               verbose=1):
    m, n = A.shape
    assert P.shape[1] == n
    assert Q.shape[1] == m
    l = linalg.svdvals(A)[0]
    sigma = tau = .99 / l
    assert sigma * tau * l * l < 1.
    x = np.zeros(n)
    xbar = x.copy()
    y = np.zeros(m)
    eps = 1e-2
    for k in xrange(max_iter):
        old_x = x.copy()
        old_y = y.copy()

        y += sigma * A.dot(xbar)
        if verbose:
            print "\tDual projection"
        y = devil(Q, b, y, verbose=verbose, tol=eps)
        x -= tau * A.T.dot(y)
        if verbose:
            print "\tPrimal projection"
        x = devil(P, a, x, verbose=verbose, tol=eps)
        eps *= .5
        eps = max(eps, 1e-4)
        xbar = 2 * x - old_x
        error = .5 * (((x - old_x) ** 2).sum() / tau + (
                (y - old_y) ** 2).sum() / sigma)
        if callback:
            callback(locals())
        if verbose:
            print "Iteration: %03i/%03i: error=%g" % (k + 1, max_iter, error)

        if error < tol:
            if verbose:
                print "Converged after %i iterations." % (k + 1)
            break

    return x, y

if __name__ == "__main__":
    # L = np.array([[1, -2]])
    # z = np.array([2, -1])
    # r = 0.
    L = np.random.randn(50, 1000)
    z = np.random.randn(L.shape[1])
    r = L.dot(z)
    # devil(L, r, z)
    # evil(L, r, z, tol=1e-10)

    # gamma = 2. / linalg.svdvals(L)[0] ** 2
    # v = np.array([0.])
    # for _ in xrange(100):
    #     x = np.maximum(z - L.T.dot(v), 0.)
    #     v += gamma * (L.dot(x) - r)
    #     print x

    # print L.dot(x)
    # print sqrt(((x - z) ** 2).sum())

    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    A = A.T
    P = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    a = np.eye(P.shape[0])[0]
    Q = np.array([[1., 0., 0.], [-1., 1., 1.]])
    b = np.eye(Q.shape[0])[0]
    import pylab as pl
    from matplotlib import lines

    i = 2

    def cb(env):
        # update locus
        ax.add_line(lines.Line2D([env['old_x'][i], env['x'][i]],
                                 [env['old_y'][i], env['y'][i]],
                                 linestyle="-."))

        pl.draw()

    xstar = [0.99797555, 0.40236764, 0.59545986, 0.19841686, 0.79815282]
    ystar = [1.00958318, 0.50675152, 0.50634814]
    pl.figure(figsize=(13, 7))
    ax = pl.subplot("111")
    pl.xlabel("xk")
    pl.ylabel("yk")
    pl.xlim(0., 1.2)
    pl.ylim(0., 1.2)
    pl.ion()

    # draw center
    pl.scatter(xstar[i], ystar[i], marker='.')
    pl.draw()

    x, y = compute_nash(A, P, Q, a, b, callback=cb, tol=1e-6)
    print x, y
