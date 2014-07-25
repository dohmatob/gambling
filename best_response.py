import numpy as np
from scipy import linalg


def best_response(A, E, e, y0, max_iter=1000, tol=1e-4):
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
    old_gain = np.inf
    L = linalg.svdvals(np.vstack((A, E)))[0]
    tau = sigma = .99 / L
    assert tau * sigma * L * L < 1.
    const = tau * A.T.dot(y0)
    for k in xrange(max_iter):
        old_x = x.copy()
        gain = -np.dot(x.T, const) / tau
        gain_delta = old_gain - gain
        old_gain = gain
        print "Iteration %03i/%03i: gain=%g, change=%g" % (
            k + 1, max_iter, gain, gain_delta)
        gain_delta = abs(gain_delta)
        if gain_delta < tol:
            print "Converged (gain delta < %g)" % tol
            break
        zeta += sigma * (e - E.dot(xbar))
        x += const + E.T.dot(zeta)

        x = np.maximum(x, 0.)

        xbar = 2 * x - old_x

    return x


if __name__ == "__main__":
    A = np.zeros((5, 3))
    A[2:4, 1:] = [[1, -1], [-2, 4]]
    A[-1, 0] = 1
    A = A.T
    E = np.array([[1, 0, 0, 0, 0], [-1, 1, 1, 0, 0], [-1, 0, 0, 1, 1]])
    e = np.eye(E.shape[0])[0]
    y0 = np.array([1, .66, .33])
    xstar = best_response(A, E, e, y0, tol=1e-4)
    print "x* =", xstar
