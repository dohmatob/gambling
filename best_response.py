import numpy as np
from scipy import linalg


def best_response(A, E, e, y0, max_iter=1000, tol=1e-2, callback=None):
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
    average_error = 0
    average_errors = []
    for k in xrange(max_iter):
        if callback:
            callback(locals())

        old_x = x.copy()
        old_zeta = zeta.copy()

        # dual update
        zeta += sigma * (e - E.dot(xbar))
        x += const + E.T.dot(zeta)

        # primal update
        x = np.maximum(x, 0.)
        xbar = 2 * x - old_x

        # check primal-dual convergence
        a = x - old_x
        b = zeta - old_zeta
        average_error = .5 * (np.dot(a, a) / tau + np.dot(
                    b, b) / sigma)
        average_errors.append(average_error)
        print ("Iteration %04i/%04i: running average .5 * (||x^(k+1) - "
               "x^(k)||^2" " / tau + ||zeta^(k+1) - zeta^(k)||^2 / sigma) "
               "= %.3e" % (k + 1, max_iter, average_error))
        if average_error < tol:
            print ("Converged (running average .5 * (||x^(k+1) - x^(k)||^2"
                   " / tau + ||zeta^(k+1) - zeta^(k)||^2 / sigma) < %g)" % tol)
            break

    return x, average_errors


if __name__ == "__main__":
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    A = A.T
    E = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    e = np.eye(E.shape[0])[0]
    y0 = np.array([1, .66, .33])

    import pylab as pl

    def cb(env):
        pl.scatter(env['x'][0], env['zeta'][0], marker="*")
        pl.draw()

    fig = pl.figure(figsize=(13, 13))
    pl.grid("on")
    pl.xlim(0., 2.)
    pl.ylim(-1., 1.)
    pl.xlabel("xk")
    pl.ylabel("yk")
    pl.ion()

    xstar, average_errors = best_response(A, E, e, y0, callback=cb, tol=1e-4)
    print "x* =", xstar
    print E.dot(xstar) - e

    pl.figure()
    pl.semilogx(average_errors, linewidth=3)
    pl.ylabel("running average .5 * (||x^(k+1) - x^(k)||^2 / tau + "
              "||zeta^(k+1) - zeta^(k)||^2 / sigma)")
    pl.xlabel("k")
    pl.show()
