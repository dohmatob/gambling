# Author: DOHMATOB Elvis <gmdopp@gmail.com>

from math import sqrt
import numpy as np
from scipy import linalg


def condat_primal_dual(grad_F, prox_G, prox_Hstar, beta, L=None,
                       norm_L=None, max_iter=100, verbose=0, tol=1e-4,
                       align=""):
    """
    Primal-dual solver for the following sum-of-composite functions
    minimization problem:

        minimize F(x) + G(x) + H(Lx)
           x
    """

    # vars
    if L is None:
        norm_L = 1.
    elif norm_L is None:
        norm_L = linalg.svdvals(L)[0]
    if norm_L is None:
        norm_L = linalg.svdvals(L)[0]
    sigma = .99 / norm_L
    tau = .9 / (sigma * norm_L * norm_L + .5 * beta)
    assert 1. / tau - sigma * norm_L * norm_L >= beta * .5
    print norm_L, beta, sigma, tau
    x = np.zeros(L.shape[1])
    y = np.zeros(L.shape[0])
    ybar = y.copy()
    delta = 2. - .5 * beta / (1. / tau - sigma * norm_L * norm_L)
    assert 1. <= delta <= 2.
    rho = .99 * delta  # acceleration param
    assert 0. < rho < delta

    _Ldot = lambda x: x if L is None else L.dot(x)
    _LTdot = lambda v: v if L is None else L.T.dot(v)

    # main loop
    for k in xrange(max_iter):
        old_x = x.copy()
        old_y = y.copy()

        # primal-dual updates
        eps = 1.e-2 * 1. / (k + 1.) ** 4.
        xbar = x - tau * (grad_F(x) + _LTdot(y))
        xbar = prox_G(xbar, tau, tol=eps)
        ybar = y + sigma * _Ldot(2. * xbar - x)
        ybar = prox_Hstar(ybar, sigma, tol=eps)
        x = rho * xbar + (1. - rho) * old_x
        y = rho * ybar + (1. - rho) * old_y

        # check convergence
        error = sqrt((((x - old_x) ** 2).sum() / tau + (
                    (y - old_y) ** 2).sum() / sigma))
        if verbose:
            print ("%sCONDAT teration: %03i/%03i: ||x^(k) - x^(k-1)||^2"
                   " + ||y^(k) - y^(k-1)||^2 =%g" % (align, k + 1, max_iter,
                                                     error))
        if error < tol:
            if verbose:
                print "%sConverged after %i iterations." % (align, k + 1)
            break

    return x


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
    elif norm_A is None:
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
        if method == 'CHAMBOLLE-POCK':
            v += sigma * (_Adot(xbar) - z)
            v /= (1. + sigma)
            zeta += sigma * (L.dot(xbar) - r)
            x -= tau * (_ATdot(v) + L.T.dot(zeta))
            x = np.maximum(x, 0.)
            xbar = 2 * x - old_x
        else:
            # L. Condat's formula
            xbar = x - tau * (_ATdot(v) + L.T.dot(zeta))
            xbar = np.maximum(xbar, 0.)
            vbar = v + sigma * (_Adot(2 * xbar - x) - z)
            vbar /= (1. + sigma)
            zetabar = zeta + sigma * (L.dot(2 * xbar - x) - r)
            x = rho * xbar + (1. - rho) * old_x
            v = rho * vbar + (1. - rho) * old_v
            zeta = rho * zetabar + (1. - rho) * old_zeta

        # check convergence
        error = sqrt((((x - old_x) ** 2).sum() / tau + (
                    ((v - old_v) ** 2).sum() + (
                        (zeta - old_zeta) ** 2).sum()) / sigma))
        if verbose:
            print "\t%s teration: %03i/%03i: error=%g" % (
                method, k + 1, max_iter, error)
        if error < tol:
            if verbose:
                print "\tConverged after %i iterations." % (k + 1)
            break

    return x


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


def compute_ne(A, E, F, e, f, norm_A=None, norm_E=None, norm_F=None,
               max_iter=1000, callback=None, tol=1e-4, verbose=1):
    """Compute Nash Equilibrium point (x*, y*) in a two-person zero-sum
    sequential-form game (in the sense of Bernhard von Stengel et al.).

    The optimization problem is:

        maximize minimize < x, Ay >
        x in Q1  y in Q2

    or equivalently,

        minimize maximize < x, Ay >
        y in Q2  x in Q1

    The complexes Q1 = {x: Ex=e, x>=0} and Q2 = {y: Fy=f, y>=0} encode the
    linear constraints on the realization plans x and y of players 1 and 2
    respectively. A is the payoff matrix from player 1's perspective.
    """

    # vars
    m, n = A.shape
    assert E.shape[1] == m
    assert F.shape[1] == n
    if norm_A is None:
        norm_A = linalg.svdvals(A)[0]
    if norm_E is None:
        norm_E = linalg.svdvals(E)[0]
    if norm_F is None:
        norm_F = linalg.svdvals(F)[0]
    sigma = tau = .99 / norm_A
    assert sigma * tau * norm_A * norm_A < 1.
    x = np.zeros(m)
    y = np.zeros(n)
    ybar = y.copy()
    eps = 1e-1

    # main loop
    values = []
    old_value = np.inf
    for k in xrange(max_iter):
        # misc
        old_y = y.copy()

        x += sigma * A.dot(ybar)
        if verbose > 1:
            print "\tDual projection"
        x = evil(E, e, x, verbose=(verbose > 1), tol=eps, L_norm=norm_E)
        y -= tau * A.T.dot(x)
        if verbose > 1:
            print "\tPrimal projection"
        y = evil(F, f, y, verbose=(verbose > 1), tol=eps, L_norm=norm_F)
        eps = max(1e-5, eps * .1)  # decrease eps (but not too much)
        ybar = 2. * y - old_y

        # check convergence
        value = np.dot(x, A.dot(y))
        values.append(value)
        change = old_value - value
        old_value = value
        if callback:
            callback(locals())
        if verbose:
            print "Primal-Dual iter %03i/%03i: value=%5.2e, change=%5.2e" % (
                k + 1, max_iter, value, change)
        if abs(change) < tol:
            if verbose:
                print "Converged after %i iterations." % (k + 1)
            break

    return x, y, values

if __name__ == "__main__":
    """
    Example from Bernhard von Stengel's "Efficient Computation of
    Behavior Strategies", Fig 2.1.
    """

    ### Load the game tree ###################################################
    # import scipy.io
    # game = scipy.io.loadmat('game.mat')
    # A, P, Q, a, b = [game[key] for key in "APQab"]
    # A = A.T
    # a = a.ravel()
    # b = b.ravel()
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    P = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    a = np.eye(P.shape[0])[0]
    Q = np.array([[1., 0., 0.], [-1., 1., 1.]])
    b = np.eye(Q.shape[0])[0]

    ### Visualization callbacks ##############################################
    i = 2

    import pylab as pl
    pl.figure(figsize=(15, 7))
    ax = pl.subplot("111")
    pl.xlabel("xk")
    pl.ylabel("yk")
    pl.xlim(-1, 2)
    pl.ylim(-1, 2)
    pl.ion()
    pl.scatter(0, 0)
    pl.show()

    def cb(env):
        ax.scatter(env["x"][i], env['y'][i])
        pl.draw()

    ### Run primal-dual algo to compute NE ####################################
    x, y, values = compute_ne(
        A, P, Q, a, b, callback=cb, verbose=2, tol=1e-10)
    print
    print "Nash Equilibrium:"
    print "x* = ", x
    print "y* =", y

    pl.figure()
    pl.plot(values, 's-')
    value = values[-1]
    pl.axhline(value, linestyle="--", label="equilibrium value" % value)
    pl.ylabel("value")
    pl.xlabel("time (s)")
    pl.legend(loc="best")
    pl.show()
