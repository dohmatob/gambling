# Author: DOHMATOB Elvis <gmdopp@gmail.com>

import time
import numpy as np
from scipy import linalg


def devil(L, r, z, A=None, norm_L=None, norm_A=None, max_iter=100, verbose=0,
          method="condat", tol=1e-4):
    """Projects the point z onto the intersection of the hyperplane
    {x: Lx=r} and the convex set C.

    The general problem is:

        minimize .5 * ||Ax - z||^2
        x in C
        s.t Lx = r

    where A = I (i.e we seek a euclidean projection) by default.

    Notes
    -----
    Method is Chambolle-Pock's saddle-point technique.

    - The (operator) norm of a matrix is defined as its largest singular value.
    - The algorithm derives from equations (7) of ref [1] with:
      K^T = [A^T, L^L]; G = i_(R_+)^n; and
      F*(v, u) = .5 ||v||^2 + z^T v + r^T theta

    References
    ----------
    [1] A. Chambolle, T. Pock, "A first-order primal-dual algorithm for convex
        problems with applications to imaging"
    """

    method = method.upper()
    assert method in ['CONDAT', 'CHAMBOLLE-POCK']

    # vars
    q, n = L.shape
    if norm_L is None:
        norm_L = linalg.svdvals(L)[0]
    norm = norm_L
    if A is None:
        norm = max(norm, 1.)
    else:
        assert 0
        if norm_A is None:
            norm_A = linalg.svdvals(A)[0]
        norm = max(norm, norm_A)
    sigma = tau = .99 / norm
    assert sigma * tau * norm * norm < 1.
    x = np.zeros(n)
    xbar = x.copy()
    v = np.zeros(n)
    zeta = np.zeros(q)
    rho = .9  # acceleration parameter

    # matvec for A and A.T
    _Adot = lambda arg: arg if A is None else A.dot(arg)
    _ATdot = lambda arg: arg if A is None else A.T.dot(arg)

    # main loop
    for k in xrange(max_iter):
        # misc
        old_x = x.copy()
        old_v = v.copy()
        old_zeta = zeta.copy()

        # Chambolle-Pock formula

        if method == 'chambolle-pock':
            v += sigma * (_Adot(xbar) - z)
            v /= (1. + sigma)
            zeta += sigma * (L.dot(xbar) - r)
            x -= tau * (_ATdot(v) + L.T.dot(zeta))
            x = np.maximum(x, 0.)
            xbar = 2 * x - old_x
        else:
            # L. Condat's formula
            xbar = x - tau * (_Adot(v) + L.T.dot(zeta))
            xbar = np.maximum(xbar, 0.)
            vbar = v + sigma * (_Adot(2 * xbar - x) - z)
            vbar /= (1. + sigma)
            zetabar = zeta + sigma * (L.dot(2 * xbar - x) - r)
            x = rho * xbar + (1. - rho) * old_x
            v = rho * vbar + (1. - rho) * old_v
            zeta = rho * zetabar + (1. - rho) * old_zeta

        # check convergence
        error = .5 * (((x - old_x) ** 2).sum() / tau + (
                ((v - old_v) ** 2).sum() + (
                    (zeta - old_zeta) ** 2).sum()) / sigma)
        if verbose:
            print "\t%s teration: %03i/%03i: error=%g" % (
                method, k + 1, max_iter, error)
        if error < tol:
            if verbose:
                print "\tConverged after %i iterations." % (k + 1)
            break

    return x


def evil(L, r, z, norm=None, max_iter=100, verbose=0, tol=1e-4):
    """Algorithm from https://www.ljll.math.upmc.fr/~plc/svva1.pdf (p.17).
    """
    q, n = L.shape
    if norm is None:
        norm = linalg.svdvals(L)[0]
    gamma = 1. / norm ** 2
    v = np.zeros(q)
    x = np.zeros(n)
    for k in xrange(max_iter):
        old_x = x.copy()
        old_v = v.copy()
        x = np.maximum(z - L.T.dot(v), 0.)
        v += gamma * (L.dot(x) - r)
        error = ((x - old_x) ** 2).sum() + ((v - old_v) ** 2).sum()
        if verbose:
            print ("\tIteration: %03i/%03i: x^(k) = %s, v^(k) = %s, "
                   "||x^(k) - x^(k-1)||^2 + ||v^(k) - v^(k-1)||^2 "
                   "= %5.3e" % (k + 1, max_iter, x, v, error))
        if error < tol:
            print "\tConverged"
            break
    return x


def compute_ne(A, P, Q, a, b, norm_A=None, norm_P=None, norm_Q=None,
               max_iter=1000, callback=None, tol=1e-4, verbose=1,
               method="condat"):
    """Compute Nash Equilibrium point (x*, y*) in a two-person zero-sum
    sequential game in sequential form (in the sense of Bernhard
    von Stengel and co-workers).

    The hyperplanes {x: Px=a} and {y: Qy=b} encode the linear constraints
    on the realization plans x and y of players 1 and 2 respectively.
    A is the payoff matrix.

    Method is Chambolle-Pock's saddle-point technique.
    """

    tic = time.time()
    method = method.upper()
    assert method in ['CONDAT', 'CHAMBOLLE-POCK']

    # vars
    m, n = A.shape
    assert P.shape[1] == n
    assert Q.shape[1] == m
    if norm_A is None:
        norm_A = linalg.svdvals(A)[0]
    if norm_P is None:
        norm_P = linalg.svdvals(P)[0]
    if norm_Q is None:
        norm_Q = linalg.svdvals(Q)[0]
    sigma = tau = .99 / norm_A
    assert sigma * tau * norm_A * norm_A < 1.
    x = np.zeros(n)
    xbar = x.copy()
    y = np.zeros(m)
    eps = 1e-1

    # main loop
    errors = []
    times = []
    for k in xrange(max_iter):
        # misc
        old_x = x.copy()
        old_y = y.copy()

        if method == "CHAMBOLLE-POCK":
            # Chambolle-Pock's primal-dual iteration
            y += sigma * A.dot(xbar)
            if verbose:
                print "\tDual projection"
            y = devil(Q, b, y, norm_L=norm_Q, verbose=verbose, tol=eps)
            x -= tau * A.T.dot(y)
            if verbose:
                print "\tPrimal projection"
            x = devil(P, a, x, norm_L=norm_P, verbose=verbose, tol=eps)
            eps *= .5
            xbar = 2 * x - old_x

        else:
            # L. Condat's primal-dual iteration
            eps = 1.e-2 * 1. / (k + 1.) ** 4.
            rho = .9  # note that \sum_{k}{\rho_k \epsilon_k} < \infty
            xbar = devil(P, a, x - tau * A.T.dot(y), verbose=(verbose > 1),
                         tol=eps, method=method)
            ybar = devil(Q, b, y + sigma * A.dot(2 * xbar - x),
                         verbose=(verbose > 1), tol=eps, method=method)
            x = rho * xbar + (1. - rho) * old_x
            y = rho * ybar + (1. - rho) * old_y

        # check convergence
        error = .5 * (((x - old_x) ** 2).sum() / tau + (
                (y - old_y) ** 2).sum() / sigma)
        times.append(time.time() - tic)
        errors.append(error)
        if callback:
            callback(locals())
        if verbose:
            print ("%s iteration: %03i/%03i: ||x^(k) - x^(k-1)||^2 + ||y^(k)"
                   " - y^(k-1)||^2 =%g" % (method, k + 1, max_iter,
                                           error))
        if error < tol:
            if verbose:
                print "Converged after %i iterations." % (k + 1)
            break

    return x, y, errors, times

if __name__ == "__main__":
    """
    Example from Bernhard von Stengel's "Efficient Computation of
    Behavior Strategies", Fig 2.1.
    """

    ### Load the game tree ###################################################
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    A = A.T
    P = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    a = np.eye(P.shape[0])[0]
    Q = np.array([[1., 0., 0.], [-1., 1., 1.]])
    b = np.eye(Q.shape[0])[0]

    ### Visualization callbacks ##############################################
    i = 2.

    import pylab as pl
    # from matplotlib import lines
    # pl.figure(figsize=(15, 7))
    # ax = pl.subplot("111")
    # pl.xlabel("xk")
    # pl.ylabel("yk")
    # pl.xlim(0., 1.2)
    # pl.ylim(0., 1.2)
    # pl.ion()
    # pl.show()

    # def cb(env):
    #     # update locus
    #     ax.add_line(lines.Line2D([env['old_x'][i], env['x'][i]],
    #                              [env['old_y'][i], env['y'][i]],
    #                              linestyle="-"))

    #     pl.draw()

    ### Run primal-dual algo to compute NE ####################################
    pl.figure()
    for method in ["condat", "chambolle-pock"][::-1]:
        x, y, errors, times = compute_ne(A, P, Q, a, b, tol=1e-6,
                                         method=method, callback=None,
                                         verbose=0)

        # pl.axvline(x[i], linestyle='--')
        # pl.axhline(y[i], linestyle='--')

        print
        print "Nash Equilibrium:"
        print "x* = ", x
        print "y* =", y

        pl.loglog(times, errors, label=method)
    pl.legend(loc="best")
    pl.show()
