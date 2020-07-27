
# A fast numpy reference implementation of GHT, as per
# "A Generalization of Otsu's Method and Minimum Error Thresholding"
# Jonathan T. Barron, ECCV, 2020
import numpy as np

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
    """Some math that is shared across multiple algorithms."""
    assert np.all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np.all(x[1:] >= x[:-1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0 / (w0 + w1)
    p1 = w1 / (w0 + w1)
    mu0 = csum(n * x) / w0
    mu1 = dsum(n * x) / w1
    d0 = csum(n * x**2) - w0 * mu0**2
    d1 = dsum(n * x**2) - w1 * mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
    """[summary]

    Args:
        n ([type]): bin counts
        x ([type], optional): bin positions. Defaults to None.
        nu (int, optional): [description]. Defaults to 0.
        tau (int, optional): [description]. Defaults to 0.
        kappa (int, optional): Concentration of the beta pdf.
                               The concentration κ parameter determines the strength of the effect of omega,
                               and setting κ = 0 disables this regularizer entirely.
                               Defaults to 0.
        omega (float, optional): Mode of the beta pdf.
                                 Setting ω near 0 biases the algorithm towards thresholding near the start of the histogram,
                                 and setting it near 1 biases the algorithm towards thresholding near the end of the histogram.
                                 Defaults to 0.5.

    Returns:
        argmax(x, f0 + f1), f0 + f1
    """
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
    v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
    f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
    f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
    return argmax(x, f0 + f1), f0 + f1
