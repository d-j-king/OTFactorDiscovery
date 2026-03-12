# Flow-based optimal transport barycenter solvers.
#
# Original author: Kai M. Hung (AM-SURE 2023)
# Improvements: numerical stability, vectorised gradients, sigma selection.

import numpy as np


# ---------------------------------------------------------------------------
# Kernel utilities
# ---------------------------------------------------------------------------

def select_sigma(X: np.ndarray, percentile: float = 50) -> float:
    """
    Data-driven bandwidth selection via the median (or arbitrary percentile)
    heuristic.

    For a Gaussian kernel K(x, y) = exp(-||x - y||² / 2σ²), a principled
    default is σ = median(pairwise distances) / sqrt(2).  This ensures that
    the kernel values span a meaningful range rather than saturating near 0
    or 1.

    Parameters
    ----------
    X : ndarray of shape (N, d)
    percentile : float, default=50
        Which percentile of pairwise distances to use.  50 = median heuristic.

    Returns
    -------
    float : recommended sigma value

    Example
    -------
    >>> sigma = select_sigma(Y)
    >>> K = gaussian_kernel(Y, sigma=sigma)
    """
    # Only subsample for large N to keep cost manageable
    if X.shape[0] > 500:
        rng = np.random.default_rng(0)
        idx = rng.choice(X.shape[0], size=500, replace=False)
        X = X[idx]

    sq_dists = (
        np.sum(X ** 2, axis=1, keepdims=True)
        - 2 * X @ X.T
        + np.sum(X ** 2, axis=1, keepdims=True).T
    )
    # Upper triangle, excluding diagonal (self-distances = 0)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median_sq = np.percentile(upper, percentile)
    # sigma such that exp(-d² / 2σ²) = exp(-0.5) at d = median → σ = median/√2
    sigma = np.sqrt(median_sq / 2.0)
    return max(sigma, 1e-6)  # never return zero


def gaussian_kernel(X: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian (RBF) kernel matrix.

    K[i, j] = exp( -||X[i] - X[j]||² / 2σ² )

    Parameters
    ----------
    X : ndarray of shape (N, d)
    sigma : float, default=1.0

    Returns
    -------
    K : ndarray of shape (N, N)
    """
    # Numerically stable squared-distance computation via the identity
    # ||a - b||² = ||a||² - 2⟨a,b⟩ + ||b||²
    sq_norms = np.sum(X ** 2, axis=1)
    sq_dists = sq_norms[:, None] - 2.0 * (X @ X.T) + sq_norms[None, :]
    # Clamp small negatives from floating-point rounding
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.exp(-sq_dists / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# Gradient of the KL loss — vectorised
# ---------------------------------------------------------------------------

def gaussian_kernel_kl_grad(
    y: np.ndarray,
    x: np.ndarray,
    lam: float,
    k_y: np.ndarray,
    k_z: np.ndarray,
    eps: float = 1e-10,
    verbose: int = 0,
) -> np.ndarray:
    """
    Gradient of the barycenter loss w.r.t. y (vectorised).

    Loss:  L(y) = Σ_i [ ½||y_i - x_i||² + λ * log( Σ_l K_y(i,l)·K_z(i,l) )
                                            - λ * log( Σ_l K_y(i,l) ) ]

    The gradient w.r.t. y_i decomposes as:

        ∇_{y_i} L = (y_i - x_i)
                  + λ * [ Σ_l (K_y·K_z)[i,l] * (y_i - y_l) / σ² ] / [ Σ_l (K_y·K_z)[i,l] ]
                  - λ * [ Σ_l  K_y[i,l]       * (y_i - y_l) / σ² ] / [ Σ_l  K_y[i,l]       ]

    This implementation is fully vectorised — no Python loop over samples.

    Parameters
    ----------
    y : ndarray (N, d) — current barycenter points
    x : ndarray (N, d) — observed data
    lam : float — regularisation strength (independence penalty)
    k_y : ndarray (N, N) — Gaussian kernel matrix for y
    k_z : ndarray (N, N) — Gaussian kernel matrix for z (pre-computed, constant)
    eps : float — small constant to prevent division by zero
    verbose : int

    Returns
    -------
    grad : ndarray (N, d)
    """
    N, d = y.shape
    sigma_sq = 1.0  # NOTE: must match the sigma used to build k_y

    # Joint kernel
    k_yz = k_y * k_z  # (N, N)

    # Denominators (with eps guard)
    denom_yz = np.sum(k_yz, axis=1, keepdims=True) + eps  # (N, 1)
    denom_y  = np.sum(k_y,  axis=1, keepdims=True) + eps  # (N, 1)

    # y_i - y_l  broadcasted: diff[i, l, :] = y[i] - y[l]
    # Shape: (N, N, d)
    diff = y[:, np.newaxis, :] - y[np.newaxis, :, :]  # (N, N, d)

    # Weighted sums over l:
    #   term1[i, :] = Σ_l k_yz[i,l] * (y_i - y_l)   / denom_yz[i]
    #   term2[i, :] = Σ_l k_y[i,l]  * (y_i - y_l)   / denom_y[i]
    term1 = np.einsum('il,ild->id', k_yz, diff) / denom_yz  # (N, d)
    term2 = np.einsum('il,ild->id', k_y,  diff) / denom_y   # (N, d)

    grad = (y - x) + lam * (term1 - term2) / sigma_sq

    if verbose > 0:
        print(f"  grad norm = {np.linalg.norm(grad):.6f}")

    return grad


# ---------------------------------------------------------------------------
# Point-wise gradient (kept for reference / debugging)
# ---------------------------------------------------------------------------

def gaussian_kernel_grad(
    y: np.ndarray,
    i: int,
    l,
    gauss_kernel: np.ndarray,
    sigma: float = 1.0,
    second_kernel: np.ndarray = None,
) -> np.ndarray:
    """
    Gradient of the kernel K_y(i, l) w.r.t. y[i].

    ∂K(y_i, y_l)/∂y_i = -K(y_i, y_l) * (y_i - y_l) / σ²

    Parameters
    ----------
    y : (N, d)
    i : int — sample index
    l : int or array of ints — neighbor index/indices
    gauss_kernel : (N, N) kernel matrix
    sigma : float
    second_kernel : (N, N) or None — if provided, weights by K_y * K_z

    Returns
    -------
    gradient : (|l|, d) or (d,) if l is scalar
    """
    k_il = gauss_kernel[i, l]
    if second_kernel is not None:
        k_il = k_il * second_kernel[i, l]

    diff = y[i] - y[l]
    return -np.multiply(k_il[..., np.newaxis], diff) / sigma ** 2


# ---------------------------------------------------------------------------
# Barycenter solver
# ---------------------------------------------------------------------------

def compute_barycenter(
    x: np.ndarray,
    z: np.ndarray,
    y_init: np.ndarray,
    lam: float,
    sigma_y: float = None,
    sigma_z: float = None,
    barycenter_cost_grad=None,
    kern_y=gaussian_kernel,
    kern_z=gaussian_kernel,
    epsilon: float = 0.001,
    lr: float = 0.01,
    max_iter: int = 1000,
    verbose: int = 0,
    adaptive_lr: bool = False,
    growing_lambda: bool = True,
    warm_stop: int = 200,
    max_lambda: float = 300.0,
    monitor=None,
) -> tuple:
    """
    Flow-based OT barycenter via gradient descent.

    Minimises  L(y) = Σ_i ½||y_i - x_i||² + λ · KL(p_y || p_z)
    where KL is estimated using Gaussian kernel density.

    Parameters
    ----------
    x : ndarray (N, d) — observed data
    z : ndarray (N, k) — known factor (held fixed)
    y_init : ndarray (N, d) — initial barycenter estimate
    lam : float — initial regularisation strength
    sigma_y : float or None — Gaussian bandwidth for y.  If None, the median
        heuristic (``select_sigma``) is applied at each iteration.
    sigma_z : float or None — Gaussian bandwidth for z.  If None, auto-selected
        once at the start.
    barycenter_cost_grad : callable or None
        Custom gradient function.  Signature:
        ``grad(y, x, lam, k_y, k_z) -> ndarray (N, d)``.
        Defaults to the vectorised ``gaussian_kernel_kl_grad``.
    kern_y, kern_z : callables — kernel functions, default ``gaussian_kernel``
    epsilon : float — convergence threshold on gradient norm
    lr : float — initial learning rate
    max_iter : int
    verbose : int
    adaptive_lr : bool — scale lr by 1.01 / 0.5 based on gradient trend
    growing_lambda : bool — linearly warm up λ from ``lam`` to ``max_lambda``
    warm_stop : int — iterations before early stopping is allowed
    max_lambda : float — final λ when growing_lambda is True
    monitor : Monitor or None

    Returns
    -------
    y : ndarray (N, d) — barycenter solution
    loss_history : list of float — per-iteration gradient norms
    """
    if barycenter_cost_grad is None:
        barycenter_cost_grad = gaussian_kernel_kl_grad

    y = y_init.copy().astype(float)

    # Auto-select bandwidths once
    if sigma_z is None:
        sigma_z = select_sigma(z)
        if verbose >= 1:
            print(f"Auto-selected sigma_z = {sigma_z:.4f}")

    k_z = kern_z(z, sigma=sigma_z)  # constant throughout
    old_grad_norm = float('inf')
    grad_norm_history: list[float] = []

    if growing_lambda:
        lambda_growth = (max_lambda - lam) / max(warm_stop, 1)

    for it in range(1, max_iter + 1):
        # (Re-)compute k_y at current y
        _sigma_y = sigma_y if sigma_y is not None else select_sigma(y)
        k_y = kern_y(y, sigma=_sigma_y)

        grad = barycenter_cost_grad(y, x, lam, k_y, k_z, verbose=verbose)
        grad_norm = float(np.linalg.norm(grad))
        grad_norm_history.append(grad_norm)

        y = y - lr * grad

        # convergence check (only after warm-up)
        if it > warm_stop and grad_norm < epsilon:
            if verbose >= 1:
                print(f"Converged at iteration {it}  (grad norm = {grad_norm:.2e})")
            break

        # adaptive learning rate
        if adaptive_lr:
            if 1e-6 < lr < 1.0:
                lr = lr * 1.01 if grad_norm < old_grad_norm else lr * 0.5
        old_grad_norm = grad_norm

        # lambda warm-up
        if growing_lambda and it <= warm_stop:
            lam += lambda_growth

        if verbose >= 1 and it % 100 == 0:
            print(f"  iter {it:4d} | grad norm = {grad_norm:.4e} | λ = {lam:.2f}")

        if monitor is not None:
            monitor.eval({"y": y, "Lambda": lam, "Iteration": it, "Gradient Norm": grad_norm})

    if verbose >= 1:
        print(f"Final: {it} iterations, grad norm = {grad_norm:.4e}")

    return y, grad_norm_history


# ---------------------------------------------------------------------------
# Loss evaluation
# ---------------------------------------------------------------------------

def kl_barycenter_loss(
    Y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    lam: float,
    sigma_y: float = 1.0,
    sigma_z: float = 1.0,
    eps: float = 1e-10,
) -> float:
    """
    Evaluate the full barycenter objective (no optimisation step).

    L(Y) = Σ_i [ ½||Y_i - X_i||² + λ · log( Σ_l K_y(i,l)·K_z(i,l) )
                                   - λ · log( Σ_l K_y(i,l) )
                                   - λ · log( Σ_l K_z(i,l) ) ]

    Parameters
    ----------
    Y : ndarray (N, d) — current barycenter
    X : ndarray (N, d) — observed data
    Z : ndarray (N, k) — factor
    lam : float
    sigma_y, sigma_z : float

    Returns
    -------
    float : total loss
    """
    k_y  = gaussian_kernel(Y, sigma=sigma_y)
    k_z  = gaussian_kernel(Z, sigma=sigma_z)
    k_yz = k_y * k_z

    recon = 0.5 * np.sum((Y - X) ** 2, axis=1)           # (N,)
    log_joint = np.log(np.sum(k_yz, axis=1) + eps)        # (N,)
    log_marg_y = np.log(np.sum(k_y,  axis=1) + eps)       # (N,)
    log_marg_z = np.log(np.sum(k_z,  axis=1) + eps)       # (N,)

    kl_term = log_joint - log_marg_y - log_marg_z         # (N,)
    return float(np.sum(recon + lam * kl_term))


# ---------------------------------------------------------------------------
# Convenience: single-entry kernel (kept for reference; note bug fix)
# ---------------------------------------------------------------------------

def gaussian_kernel_single(x_i: np.ndarray, x_l: np.ndarray, sigma: float) -> float:
    """
    Single entry of the Gaussian kernel.

    K(x_i, x_l) = exp( -||x_i - x_l||² / 2σ² )

    Note: the original version was missing the square on the norm, which
    made this inconsistent with ``gaussian_kernel``.  Fixed here.
    """
    return float(np.exp(-np.linalg.norm(x_i - x_l) ** 2 / (2.0 * sigma ** 2)))
