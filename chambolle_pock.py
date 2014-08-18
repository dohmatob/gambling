import numpy as np


proj_interval_a_b = lambda a, b, y: (a <= y) * (y <= b) * y + b * (
    y >= b) + a * (y <= a)


def averaged_projections(proj_C, proj_D, y, max_iter=100, callback=None,
                         verbose=0):
    """Averaged projections algorithm.

    Projects a point onto the intersection of two sets C and D. It is
    assumed that the euclidean projections onto C and D can be effectively
    computed separately.
    """
    for k in xrange(max_iter):
        if callback:
            old_y = y.copy()
        if verbose or 1:
            print "Iteration: %03i/%03i: proj=%s" % (k + 1, max_iter, y)

        # project
        y = .5 * (proj_C(y) + proj_D(y))

        # check convergence
        if callback:
            if callback(old_y, y):
                return y
    return y


def prox_l1(x, tau):
    d = np.abs(x)
    nz = d.nonzero()
    x[nz] *= np.maximum(1. - tau / d[nz], 0.)
    return x


def prox_conjugate(prox_f, x, tau):
    if tau == 0.:
        return x
    x /= tau
    x -= prox_f(x, 1. / tau)
    return x


def chambolle_pock(K, KT, prox_G, prox_Fstar, x0, y0, max_iter=100,
                   callback=None, theta=1., tol=1e-4):
    tau = sigma = .99 / K.L
    assert tau * sigma * K.L * K.L < 1.
    x = np.array(x0)
    y = np.array(y0)
    xbar = x.copy()
    for k in xrange(max_iter):
        old_x = x.copy()
        old_y = y.copy()
        y += sigma * K(xbar)
        y = prox_Fstar(y, sigma)
        x -= tau * KT(y)
        x = prox_G(x, tau)
        xbar = x + theta * (x - old_x)

        if callback:
            callback(locals())

        a = x - old_x
        b = y - old_y
        error = .5 * (np.dot(a, a) / tau + np.dot(b, b) / sigma)
        print "Iteration %04i/%04i: error=%.3e" % (k + 1, max_iter, error)
        if error < tol:
            break

    return x, y

if __name__ == "__main__":

    """
    This example finds the point in an interval [a, b] with the smallest
    absolute values.
    """

    class Id(object):
        def __call__(self, x):
            return x

        @property
        def L(self):
            return 1.

    a = -1
    b = 2
    K = KT = Id()
    prox_G = prox_l1
    prox_F = lambda y, _: proj_interval_a_b(a, b, y)
    prox_Fstar = lambda y, tau: prox_conjugate(prox_F, y, tau)
    xy0 = np.array([[10.], [100.]])
    xy0 += 5 * np.random.randn(*xy0.shape)
    x0, y0 = xy0
    import pylab as pl
    from matplotlib import lines
    import time

    def cb(env):
        # update locus
        ax.add_line(lines.Line2D([env['old_x'][0], env['x'][0]],
                                 [env['old_y'][0], env['y'][0]],
                                 linestyle="-."))

        pl.draw()

    pl.figure(figsize=(13, 7))
    thetas = [0., .5, 1.]
    for i, theta in enumerate(thetas):
        ax = pl.subplot("1%i%i" % (len(thetas), i + 1))
        ax.set_title("theta=%g" % theta)
        pl.xlim(-150., 150.)
        pl.ylim(-150., 150.)
        pl.xlabel("xk")
        if i == 0.:
            pl.ylabel("yk")
        pl.ion()

        # draw center
        pl.scatter(0., 0., marker="*")
        pl.draw()

        print chambolle_pock(K, KT, prox_G, prox_Fstar, x0, y0, callback=cb,
                             theta=theta)
