import numpy as np
from scipy import linalg
from nose.tools import assert_true
from sklearn.utils import check_random_state

alpha = 1.
rng = check_random_state(42)
n, p = 48, 128
K = np.eye(p)
X = rng.randn(n, p)
w = np.zeros(p)
w[::p // 8] = 1
y = X.dot(w) + rng.randn(n) * .1
XTy = X.T.dot(y)
_, S, V = linalg.svd(X, full_matrices=False)
S **= 2


def prox_G(w, tau, copy=False):
    """Computes inv(Id + tau * XTX)(w + XTy) via SVD of X"""
    if copy: w = w.copy()
    w /= tau
    w += XTy
    out = np.dot(V.T * (1. / (S + (1. / tau)) - tau), np.dot(V, w))
    out += tau * w
    return out


def prox_F_star(z, eta, copy=False):
    """Projects z orthogonally onto the l_infty ball of radius `eta`."""
    if copy: z = z.copy()
    z /= eta
    absz = np.abs(z)
    nz = (absz > 0.)
    z[nz] /= np.maximum(absz[nz], 1.)
    z[nz] *= eta
    return z


def test_prox_G():
    w = rng.randn(p)
    Id = np.eye(p)
    for tau in [.1, .5, 1., 10.]:
        np.testing.assert_array_almost_equal(
            prox_G(w, tau, copy=True),
            linalg.inv(Id + tau * X.T.dot(X)).dot(w + tau * XTy))


def test_prox_F_star():
    z = rng.randn(p)
    for eta in [.1, .5, 1., 10.]:
        assert_true(np.max(np.abs(prox_F_star(z, eta, copy=True))) <= eta)


def pd(max_iter=1500, tol=1e-6, method="cp", **kwargs):
    L = linalg.norm(K, 2)
    energies = []
    env = dict(old_energy=np.inf)

    def callback(variables):
        w = variables['w']
        k = variables['k']
        aux = X.dot(w) - y
        energy = .5 * aux.dot(aux)
        energy += alpha * np.abs(w).sum()
        env["change"] = env["old_energy"] - energy
        env["old_energy"] = energy
        energies.append(energy)
        # print "%s iter %03i/%03i: energy=%g, change=%g" % (
        #     method, k + 1, max_iter, energy, env['change'])

    if "tseng_bd" in method:
        from tseng_bd import tseng_bd
        w, z, gaps = tseng_bd(K, prox_G, lambda z, _: prox_F_star(z, alpha),
                              Lwz=L, max_iter=max_iter, callback=callback,
                              **kwargs)

    elif method == "cp":
        gaps = None
        _, p = X.shape
        tau = eta = .9 / L
        theta = 1.
        w = np.zeros(p)
        w_ = np.zeros(p)
        z = np.zeros(p)
        for k in range(max_iter):
            callback(locals())
            if np.abs(env['change']) < tol:
                print "Converged (tol=%g)." % tol
                break
            old_w = w.copy()
            z += eta * K.dot(w_)
            z = prox_F_star(z, alpha)
            w -= tau * K.T.dot(z)
            w = prox_G(w, tau)
            w_ = theta * (w - old_w) + w
    else:
        raise ValueError("Unknown method '%s'!" % method)
    return (w, z, gaps), np.array(energies)


import matplotlib.pyplot as plt
energies = {}
# for sigma in [.1, .5, .9, 1., 1.1]:
#     method = "tseng_bd $\\sigma=%g$" % sigma
for method in ["cp", "tseng_bd"][1:]:
    (w, _, e), _ = pd(method=method, max_iter=5000, tol=0.)
    energies[method] = e
min_e = np.min(map(np.min, energies.values()))
for method, e in energies.items():
    e -= min_e
    plt.loglog(e, linewidth=2, label=method)
plt.xlabel("$k$")
plt.ylabel("$\\delta E$")
plt.legend(loc="best")
plt.figure()
ax = plt.subplot(121)
ax.plot(y)
ax.set_title("input signal")
ax = plt.subplot(122)
ax.plot(w)
ax.set_title("output signal")
plt.show()
