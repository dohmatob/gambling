from math import sqrt
import numpy as np
from scipy import linalg


def best(A, E, e, y0, max_iter=1000, tol=1e-4, callback=None, **kwargs):
    c = -A.T.dot(y0)
    m, n = E.shape
    alpha = 1.
    tau = np.reciprocal([((np.abs(col) + 1e-1) ** (2. - alpha)).sum()
                         for col in E.T])
    sigma = np.reciprocal([(np.abs(row) ** alpha).sum() for row in E])
    x = np.zeros(n)
    zeta = np.zeros(m)
    old_value = np.inf
    values = []

    for k in xrange(max_iter):
        old_x = x.copy()
        x -= tau * (E.T.dot(zeta) + c)
        x = np.maximum(x, 0.)
        value = c.T.dot(x)
        values.append(value)
        change = old_value - value
        old_value = value
        zeta += sigma * (E.dot(2. * x - old_x) - e)
        if callback:
            callback(locals())

        print "Iteration %03i/%03i: value=%5.2e, change=%5.2e" % (
            k + 1, max_iter, value, change)
        change = abs(change)
        if change < tol:
            print "Converged (|change| < %5.2e)." % tol
            break

    import pylab as pl
    pl.figure()
    pl.suptitle(
        ("Computing best response strategy in sequence-form game using "
         "diagonally preconditioned primal-dual algorithm of Chambolle-Pock"))
    ax1 = pl.subplot(222)
    theta = np.linspace(0., 2 * np.pi, num=100)
    ax1.plot(np.cos(theta) / sqrt(tau[0]), np.sin(theta) / sqrt(tau[1]))
    ax1.axvline(0, linestyle="--")
    ax1.axhline(0, linestyle="--")
    ax1.set_title("\\tau")

    ax2 = pl.subplot(224)
    theta = np.linspace(0., 2 * np.pi, num=100)
    ax2.plot(np.cos(theta) / sqrt(sigma[0]), np.sin(theta) / sqrt(sigma[1]))
    ax2.axvline(0, linestyle="--")
    ax2.axhline(0, linestyle="--")
    ax2.set_title("\sigma")

    ax3 = pl.subplot(121)
    values = -np.array(values)
    value *= -1.
    ax3.semilogx(values)
    ax3.axhline(value, linestyle="--", c="r",
               label="value of game (= %g)" % value)
    ax3.set_ylabel("primal objective at kth iteration")
    ax3.set_xlabel("k")
    pl.legend(loc="best")
    pl.show()
    return x, zeta, None


# y0 = np.array([-1., -1.])
# A = -np.eye(2)
# E = np.array([[-20. / 3, 1.], [20., -1.]])
# e = np.array([20. / 3, 20.])
# xstar, _, _ = best(A, E, e, y0, tol=1e-8)
# print xstar
# print y0.dot(A.dot(xstar))
# print E.dot(xstar) - e

# assert 0


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
    values = []
    for k in xrange(max_iter):
        old_x = x.copy()
        old_zeta = zeta.copy()

        # dual update
        zeta += sigma * (e - E.dot(xbar))
        x += const + E.T.dot(zeta)

        # primal update
        x = np.maximum(x, 0.)
        xbar = x + theta * (x - old_x)

        value = -const.T.dot(x)
        values.append(value)
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

    import pylab as pl
    pl.plot(values)
    pl.axhline(value, linestyle="--", c="r",
               label="value of game (= %g)" % value)
    pl.ylabel("primal objective at kth iteration")
    pl.xlabel("k")
    pl.legend(loc="best")
    pl.title(
        ("Computing best response strategy in sequence-form game using "
         "diagonally preconditioned primal-dual algorithm of Chambolle-Pock"))
    pl.show()
    return x, zeta, None

    return x, zeta, errors


if __name__ == "__main__":
    # A = np.zeros((5, 3))
    # A[2:4, 1:] = [[1, -1], [-2, 4]]
    # A[-1, 0] = 1
    # A = A.T
    # E = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    # e = np.eye(E.shape[0])[0]
    # y0 = np.array([1., 2. / 3., 1. / 3.])

    p, m, n = 10, 50, 100
    E = np.zeros((p, n))
    E[0, 0] = 1.
    E[1:, :] = np.random.rand(E.shape[0] - 1, E.shape[1]) > .5
    E[1, 0] = -1
    for j in xrange(2, E.shape[0]):
        E[j, :] = np.roll(E[j - 1, :].copy(), (np.random.rand() > .5) + 3)

    e = np.zeros(E.shape[0])
    e[0] = 1
    A = np.ones((m, n))
    A[np.random.randn(*A.shape) > .8] = -5
    y0 = np.random.rand(m)
    y0[0] = 1.
    print E
    # assert 0
    import pylab as pl
    from matplotlib import lines

    xstar = [1., 0., 1., 0., 1.]
    zetastar = [0., 0., 0.]

    # pl.figure(figsize=(13, 7))
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

        # ax = pl.subplot("3%i%i" % (len(thetas), i + 1))
        # ax.set_title("theta=%g" % theta)
        # pl.xlim(0., 2.)
        # pl.ylim(-1., 1.)
        # pl.xlabel("xk")
        # if i == 0.:
        #     pl.ylabel("yk")
        # pl.ion()

        # # draw center
        # pl.scatter(xstar[0], zetastar[0], marker="*")
        # pl.draw()

        xstar, zetastar, errors = best(A, E, e, y0, theta=theta,
                                       callback=None, tol=1e-8,
                                       max_iter=2000)
        print "x* =", xstar
        print "zeta* =", zetastar
        print "e - Ex* =", E.dot(xstar) - e
        print xstar.dot(A.T.dot(y0))
        break

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
