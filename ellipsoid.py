from math import sqrt
import numpy as np
from scipy.sparse.linalg import LinearOperator
from nose.tools import assert_true
from nilearn.decoding.sparse_models.common import (
    gradient_id, div_id, _unmask)


def power(A, p, max_iter=1000):
    """Power iteration to estimate the dorminant eigval of an operator."""
    b = np.random.randn(p)
    for _ in xrange(max_iter):
        b = A(b)
        b /= sqrt((b ** 2).sum())
    lambd = np.dot(b, A(b)) / np.dot(b, b)
    return lambd


def diagonal_preconditioners(X, shape, l1_ratio, alpha=1.,
                            salt=1e-6):
    """Computes diagonal preconditioners for sparse-variation model."""
    shape_ = [len(shape) + 1] + list(shape)
    p = np.prod(shape)
    beta = 2. - alpha

    # compute \Tau and \Sigma_2
    linop = LinearOperator(
        (4 * p, p),
        lambda w: gradient_id(w.reshape(shape),
                              l1_ratio=l1_ratio).ravel(),
        dtype=X.dtype)

    tau = np.array([(np.abs(c) ** alpha).sum()
                    for c in X.T])
    sigma2 = [1. / (np.abs(r) ** beta).sum() for r in X]
    e = np.zeros(p)
    for j in xrange(linop.shape[1]):
        e *= 0
        e[j] = 1
        tau[j] += (np.abs(linop(e)) ** alpha).sum()
    tau = np.reciprocal(tau)

    # compute \Sigma_1
    sigma1 = np.zeros(4 * p)
    linop = LinearOperator(
        (p, (len(shape) + 1) * p),
        lambda w: -div_id(w.reshape(shape_),
                          l1_ratio=l1_ratio).ravel(),
        dtype=X.dtype)
    e = np.zeros(linop.shape[1])
    for j in xrange(linop.shape[1]):
        e *= 0
        e[j] = 1
        sigma1[j] += 1. / ((np.abs(linop(e)) + salt) ** alpha).sum()
    sigma1 = sigma1.reshape((-1, p))
    return tau, sigma1, sigma2


class DiagonallyPreconditionedStack(object):
    """Diagonally precondioned operator and resolvents for primal-dual.

    Notes
    -----
    Here, K = vertical stack of \tilde{\nabla} and X.
    """

    def __init__(self, X, shape, l1_ratio, alpha=1.,
                 mask=None, salt=1e-6):
        self.X = X
        self.l1_ratio = l1_ratio
        self.alpha = alpha  # XXX this is alpha for preconditioning!!!
        self.shape = shape
        self.mask = mask
        if self.mask is not None:
            if not shape is None:
                assert tuple(self.shape) == self.mask.shape
            else:
                self.shape = self.mask.shape
                self.mask = self.mask.ravel()
        self.p = np.prod(shape)
        self.shape_ = [len(self.shape) + 1] + list(self.shape)
        self.dual_dim = np.prod(self.shape_)
        self.tau, self.sigma1, self.sigma2 = diagonal_preconditioners(
            X, shape, l1_ratio, alpha=alpha, salt=salt)
        self.flat_sigma1 = self.sigma1.ravel()

    def maskvec(self, w):
        return w if self.mask is None else w[self.mask]

    def kmatvec(self, w):
        """Implements operator K = vertical stack of \tilde{\nabla} and X"""
        return np.append(gradient_id(w.reshape(self.shape),
                                     l1_ratio=self.l1_ratio).ravel(),
                         self.X.dot(self.maskvec(w)))

    def split_vec(self, vec):
        return vec[:self.dual_dim], vec[self.dual_dim:]

    def kTmatvec(self, v, zeta):
        """Implements adjoint operator of K."""
        return -div_id(v.reshape(self.shape_),
                       l1_ratio=self.l1_ratio).ravel() + _unmask(
            self.X.T.dot(zeta), self.mask)

    def taumatvec(self, w):
        return self.tau * w

    def sigmamatvec(self, z):
        return np.append(self.flat_sigma1 * z[:len(self.flat_sigma1)],
                         self.sigma2 * z[len(self.flat_sigma1):])

    def proj_hyperellipsoid(self, v):
        """Project v onto product of hyperellipsoids given by:
        \Sigma_1({z: |z|_(2, \infty) <= 1}).
        """

        v = v.reshape((-1, self.p))
        v *= np.minimum(1. / np.sqrt((v * v * self.sigma1).sum(axis=1))
                        [:, np.newaxis], 1.)
        return v.ravel()


def test_norm_condition():
    """Check that \|sqrt(\Sigma)Ksqrt(\Tau)\| <= 1."""
    n_samples = 2
    shape = (2, 2, 2)
    shape_ = [len(shape) + 1] + list(shape)
    dual_dim = np.prod(shape_)
    X = np.random.randn(n_samples, np.prod(shape))

    for l1_ratio in [0., .5, 1.]:
        for alpha in [0., .1, .5, 1., 1.]:  # XXX check fails for alpha = 1.75!
            tau, sigma1, sigma2 = diagonal_preconditioners(
                X, shape, l1_ratio, alpha=alpha)
            tau = np.sqrt(tau)
            sigma1 = np.sqrt(sigma1).ravel()
            sigma2 = np.sqrt(sigma2).ravel()

            def kmatvec(w):
                return np.append(gradient_id(w.reshape(shape),
                                             l1_ratio=l1_ratio).ravel(),
                                 X.dot(w))

            def split_vec(vec):
                return vec[:dual_dim], vec[dual_dim:]

            def kTmatvec(v, zeta):
                return -div_id(v.reshape(shape_),
                               l1_ratio=l1_ratio).ravel() + X.T.dot(zeta)

            def taumatvec(w):
                return tau * w

            def sigmamatvec(z):
                return np.append(sigma1 * z[:len(sigma1)],
                                 sigma2 * z[len(sigma1):])

            def matvec(w):
                return taumatvec(kTmatvec(*split_vec(
                            sigmamatvec(sigmamatvec(kmatvec(taumatvec(w)))))))

            eigval = power(matvec, np.prod(shape))
            assert_true(eigval <= 1., msg=dict(l1_ratio=l1_ratio,
                                               alpha=alpha, eigval=eigval))


def proj_hyperellipsoid(sigma, v):
    """Euclidean-projects the point v onto the hyperellipsoid given by:

        x: < diag(sigma) x, x > = 1

    Notes
    -----
    v is modified in-place.
    """
    v *= min(1. / sqrt((v * v * sigma).sum()), 1.)
    return v


if __name__ == "__main__":
    import pylab as pl
    sigma = np.array([12., 1.])
    theta = np.linspace(0., 2 * np.pi, num=100)
    v = np.array([-5., -3.])
    p = proj_hyperellipsoid(1. / sigma, v.copy())
    pl.plot(sqrt(sigma[0]) * np.cos(theta), sqrt(sigma[1]) * np.sin(theta))
    pl.scatter(*p, marker="*", s=50)
    pl.plot([0., v[0]], [0., v[1]], "o--", c="r")
    pl.show()
