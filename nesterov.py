"""
Solving matrix games via Nesterov smoothing. Quick-and-dirty code!
"""
# Author: Elvis DOHMATOB <elvis.dohmatob@inria.fr>

from math import sqrt
import numpy as np
from scipy import linalg


def proj_simplex(v, z=1.):
    """Projects v unto the simplex {x >= 0, x_0 + x_1 + ... x_n = z}.

    The method is John Duchi's O (n log n) Algorithm 1.
    """
    # deterministic O(n log n)
    u = np.sort(v)[::-1]  # sort v in increasing order
    aux = (np.cumsum(u) - z) / np.arange(1., len(v) + 1.)
    return np.maximum(v - aux[np.nonzero(u > aux)[0][-1]], 0.)


def test_proj_simplex():
    v = np.array([1.1, 0., 0.])
    np.testing.assert_array_equal(proj_simplex(v), [1., 0., 0.])

    v = np.array([0., 0., 0.])
    np.testing.assert_array_equal(proj_simplex(v), [1. / 3, 1. / 3, 1. / 3])


def nesterov_ne(A, c=None, b=None, epsilon=1e-3, max_iter=np.inf,
                dynamic_mu=False, mu_init=1e-2, mu_factor=.9):
    """Computes an approx Nash equilibrium for matrix game via Nesterov
    smoothing [1].

    Formally, the problem is

        minimize maximize <Ax, u> + <c, x> + <b, u>,
        u in S_2 x in Q_1

    where Q_j is player j's simplex.

    Parameters
    ----------
    A: ndarray, shape (m, n)
        Payoff matrix.

    c: ndarray, shape (n,), optional (default None)
        <c, x> is added to the payoff function <Ax, u>.

    b: ndarray, shape (n,), optional (default None)
        <b, u> is added to the payoff function <Ax, u>

    epsilon: positive float, optional (default 1e-3)
        Tolerance on primal-dual gap.

    max_iter: int, optional (default np.inf)
        Maximum number of iterations to run. If no value is specified,
        then it is inferred using Nesterov's formulae.

    dynamic_mu: boolean, optional (default False)
        If true, then the smoothing parameter mu will be set dynamically.

    mu_init: positive float, optional (default 1e-2)
        Initial value for the smoothing parameter mu.

    mu_factor: positive float, optional (default .9)
        Factor by which the smoothing parameter mu is shrunk at each update.

    References
    ----------
    [1] Y. Nesterov, "Smooth minimization of non-smooth functions."
    """
    # misc
    m, n = A.shape
    if not c is None: assert len(c) == n
    if not b is None: assert len(b) == m
    x_ = (1. / n) * np.ones(n)
    u_ = (1. / n) * np.ones(m)
    norm_A = linalg.norm(A, 2)
    D_1 = .5 * (1. - 1. / n)
    D_2 = .5 * (1. - 1. / m)

    # set parameters using formula 4.8 of [1]
    max_iter = min(int(np.floor(4 * norm_A * sqrt(D_1 * D_2) / epsilon)),
                   max_iter)
    mu_ = epsilon / (2. * D_2)
    x = x_.copy()
    grad_acc = np.zeros_like(x)
    values = []
    gaps = []
    mu = mu_init if dynamic_mu else mu_
    print "mu =", mu
    q = 2
    for k in range(max_iter):
        # misc
        L = norm_A ** 2 / mu  # there is an error in the L formula from [1]
        stepsize = 1. / L

        # make call to oracle
        aux = A.dot(x)
        if not b is None: aux -= b
        v = aux / mu
        v += u_
        u = proj_simplex(v)
        grad = A.T.dot(u)
        if not c is None: grad += c
        grad_acc += .5 * (k + 1.) * grad

        # callback
        value = u.dot(A.dot(x))
        values.append(value)
        gap = aux.max() - grad.min()
        if not c is None: gap += c.dot(x)
        if not b is None: gap += b.dot(u)
        gaps.append(gap)
        assert gap + 1e-10 >= 0., "The world is a weird place!"
        print "Iter %03i/%03i: game value <Ax, u> = %g, primal-dual gap=%g" % (
            k + 1, max_iter, value, gap)

        # check convergence
        if gap < epsilon:
            print "Converged (primal-dual gap < %g)." % epsilon
            break

        # y update
        y = proj_simplex(x - stepsize * grad)

        # z update
        z = proj_simplex(x_ - stepsize * grad_acc)

        # x update
        factor = 2. / (k + 3.)
        x = factor * z
        x += (1. - factor) * y

        # decrease mu ?
        if dynamic_mu and mu > mu_ and k > 0 and k % q == 0:
            # the idea is to decrease mu at iterations k = 2, 4, 8, 16, 32, ...
            q *= 2
            mu *= mu_factor
            print "Decreasing mu to %g" % mu

    return x, u, values, gaps


def gilpin_ne(A, epsilon=1e-4, max_iter=np.inf):
    """Gilpin's et al. first-order scheme for computing Nash equilibria.

    The formal problem is:

         minimize maximize <y, Ax>,
         y in Q_2   x in Q_1

    """
    A = A.T  # make compatible with nesterov_ne
    m, n = A.shape
    R = np.vstack((np.hstack((np.zeros((m, m)), -A)),
                   np.hstack((A.T, np.zeros((n, n))))))
    norm_R = linalg.norm(R, 2)
    u_ = (1. / m) * np.ones(m)
    v_ = (1. / n) * np.ones(n)
    x = u_.copy()
    y = v_.copy()
    D = 1. - .5 * (1. / m + 1. / n)
    gamma = np.e
    eps = 1.
    gradx_acc = np.zeros(m)
    grady_acc = np.zeros(n)
    values = []
    gaps = []
    k = 0.
    while k < max_iter and eps >= epsilon:
        mu = eps / (2. * D)

        # misc
        L = norm_R ** 2 / mu
        stepsize = 1. / L

        gradx_acc *= 0.
        gradx_acc *= 0.
        while True:
            # make call to oracle
            aux1 = A.dot(y)
            aux = -aux1 / mu
            aux += u_
            u = proj_simplex(aux)
            aux2 = A.T.dot(x)
            aux = aux2 / mu
            aux += v_
            v = proj_simplex(aux)
            gradx, grady = A.dot(v), -A.T.dot(u)
            gradx_acc += .5 * (k + 1.) * gradx
            grady_acc += .5 * (k + 1.) * grady

            value = x.dot(aux1)
            values.append(value)
            gap = aux2.max() - aux1.min()
            gaps.append(gap)
            assert gap + 1e-10 >= 0., "The world is a weird place!"
            print ("%03i: game value <Ax, u> = %g, primal-dual "
                   "gap=%g") % (k + 1, value, gap)

            # check convergence
            if gap < eps:
                print "Converged (primal-dual gap < %g)." % eps
                break

            # y update
            yx = proj_simplex(x - stepsize * gradx)
            yy = proj_simplex(y - stepsize * grady)

            # z update
            zx = proj_simplex(u_ - stepsize * gradx_acc)
            zy = proj_simplex(v_ - stepsize * grady_acc)

            # x update
            factor = 2. / (k + 3.)
            x = factor * zx
            x += (1. - factor) * yx
            y = factor * zy
            y += (1. - factor) * yy

            k += 1
            if k >= max_iter: break

        # decrease eps
        eps /= gamma
        print "Decreasing epsilon to %g" % eps

    return x, y, values, gaps

if __name__ == "__main__":
    # matplotlib confs
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['text.latex.preamble'] = ['\\boldmath']
    matplotlib.rc('font', family='serif', serif='cm10')
    matplotlib.rc('text', usetex=True)
    plt.rc('font', size=50)  # make the power in sci notation bigger

    from primal_dual import primal_dual_sg_ne
    rng = np.random.RandomState(42)
    A = np.array([[-2., 3.], [3, -4]])
    A = rng.randn(20, 10)
    fig1 = plt.figure(figsize=(17, 13))
    ax1 = plt.subplot("111")
    plt.grid("on")
    fig2 = plt.figure(figsize=(17, 13))
    ax2 = plt.subplot("111")
    plt.grid("on")
    for solver in ["nesterov", "gilpin", "primal-dual sg"]:
        x, u, values, gaps = eval("%s_ne" % solver.replace(" ", "_").replace(
            "-", "_"))(A, epsilon=1e-4, max_iter=100000)
        ax1.loglog(gaps, label="\\textbf{%s}" % solver, linewidth=4)
        ax1.set_xlabel("\\textbf{$k$}", fontsize=50)
        ax1.set_ylabel("\\textbf{primal-dual gap}", fontsize=50)

        ax2.semilogx(values, label="\\textbf{%s}" % solver, linewidth=4)
        ax2.set_xlabel("\\textbf{$k$}", fontsize=50)
        ax2.set_ylabel("\\textbf{game value}", fontsize=50)
    for i, (fig, ax) in enumerate(zip([fig1, fig2], [ax1, ax2])):
        if i > 0:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0., 0.))
        ax.tick_params(axis='both', which='major', labelsize=50)
        ax.legend(loc="best", prop=dict(size=45), handlelength=1.5)
        plt.tight_layout()
        plt.figure(fig.number)
        plt.savefig("%i.png" % i)
    plt.show()
