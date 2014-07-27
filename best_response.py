from math import sqrt
import numpy as np
from scipy import linalg


def best_response(A, E, e, y0, max_iter=1000, tol=1e-2, theta=1.,
                  callback=None):
    """Computes best response to an opponent's realization plan in a two-person
    sequential game.

    Parameters
    ----------
    A: 2D array of size (m, n)
        Payoff matrix.

    E: 2D array of shape (p, n)
        Constraints matrix for our realization plans (x)

    e: 1D array of length p
        Our realization plans are constrained like so: Ex = e, x>=0.

    y0: 1D array of length m
        Fixed realization plan of opponent (player 2).

    """

    _, n = A.shape
    assert E.shape[1] == n
    p, _ = E.shape
    x = np.zeros(n)
    xbar = x.copy()
    zeta = np.zeros(p)
    L = linalg.svdvals(np.vstack((A, E)))[0]
    tau = sigma = .99 / L
    assert tau * sigma * L * L < 1.
    const = tau * A.T.dot(y0)
    error = 0
    errors = []
    for k in xrange(max_iter):
        old_x = x.copy()
        old_zeta = zeta.copy()

        # dual update
        zeta += sigma * (e - E.dot(xbar))
        x += const + E.T.dot(zeta)

        # primal update
        x = np.maximum(x, 0.)
        xbar = x + theta * (x - old_x)

        if callback:
            callback(locals())

        # check primal-dual convergence
        a = x - old_x
        b = zeta - old_zeta
        error = .5 * (np.dot(a, a) / tau + np.dot(
                    b, b) / sigma)
        errors.append(error)
        print ("Iteration %04i/%04i: .5 * (||x^(k+1) - "
               "x^(k)||^2" " / tau + ||zeta^(k+1) - zeta^(k)||^2 / sigma) "
               "= %.3e" % (k + 1, max_iter, error))
        if error < tol:
            print ("Converged (.5 * (||x^(k+1) - x^(k)||^2 / tau + "
                   "||zeta^(k+1) - zeta^(k)||^2 / sigma) < %.3e)" % tol)
            break

    return x, zeta, errors


if __name__ == "__main__":
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    A = A.T
    E = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    e = np.eye(E.shape[0])[0]
    y0 = np.array([1., 2. / 3., 1. / 3.])

    import pylab as pl
    from matplotlib import lines

    xstar = [1., 0., 1., 0., 1.]
    zetastar = [0., 0., 0.]

    pl.figure(figsize=(13, 7))
    thetas = [0., .5, 1.]
    for i, theta in enumerate(thetas):
        mind = []

        def cb(env):
            v = env['x'] - xstar
            mind.append(sqrt(np.dot(v, v)))

            # draw tangent
            ax.add_line(lines.Line2D([env['old_x'][0], env['x'][0]],
                                     [env['old_zeta'][0], env['zeta'][0]]))

            # draw normal
            ax.add_line(lines.Line2D([xstar[0], env['x'][0]],
                                     [zetastar[0], env['zeta'][0]]))

            pl.draw()

        ax = pl.subplot("3%i%i" % (len(thetas), i + 1))
        ax.set_title("theta=%g" % theta)
        pl.xlim(0., 2.)
        pl.ylim(-1., 1.)
        pl.xlabel("xk")
        if i == 0.:
            pl.ylabel("yk")
        pl.ion()

        # draw center
        pl.scatter(xstar[0], zetastar[0], marker="*")
        pl.draw()

        xstar, zetastar, errors = best_response(A, E, e, y0, theta=theta,
                                                callback=cb, tol=1e-8,
                                                max_iter=200)
        print "x* =", xstar
        print "zeta* =", zetastar
        print "e - Ex* =", E.dot(xstar) - e

        ax = pl.subplot("3%i%i" % (len(thetas), len(thetas) + i + 1))
        pl.loglog(errors)
        pl.xlabel("k")
        # pl.ylabel(".5 * (||x^(k+1) - x^(k)||^2 / tau + "
        #           "||zeta^(k+1) - zeta^(k)||^2 / sigma)")
        if i == 0.:
            pl.ylabel("error")

        ax = pl.subplot("3%i%i" % (len(thetas), 2 * len(thetas) + i + 1))
        pl.loglog(mind)
        pl.xlabel("k")
        # pl.ylabel(".5 * (||x^(k+1) - x^(k)||^2 / tau + "
        #           "||zeta^(k+1) - zeta^(k)||^2 / sigma)")
        if i == 0.:
            pl.ylabel("||x^(k) - x*||")

        pl.show()
