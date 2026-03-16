"""
Generalized K-Means with pluggable cost metrics.

Author: Daniel Wang
Original date: 2023-06-23

The standard K-Means algorithm is implicitly tied to squared-Euclidean
distance: the within-cluster minimizer of that cost is the mean. This
module generalises K-Means to any Lp norm (p ≥ 1), any power of the
Euclidean distance, or a fully user-supplied cost function. The center
that minimises each cost is computed analytically where possible (mean
for L2², coordinate-wise median for L1, geometric median via Weiszfeld
for L2) and via gradient descent with backtracking line search otherwise.

The class follows the scikit-learn estimator interface.
"""

from __future__ import annotations

import re
import warnings
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import scipy.optimize as spo
import sklearn.metrics as skm
from sklearn.base import BaseEstimator, ClusterMixin

# ---------------------------------------------------------------------------
# Optional: tqdm for progress bars
# ---------------------------------------------------------------------------
try:
    from tqdm.auto import trange as _trange, tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


# ---------------------------------------------------------------------------
# KGenCenters
# ---------------------------------------------------------------------------

class KGenCenters(BaseEstimator, ClusterMixin):
    """Generalized K-Means supporting arbitrary cost metrics.

    Extends K-Means beyond squared-Euclidean distance. The minimizer of the
    within-cluster cost is no longer always the mean — depending on the
    metric it may be the coordinate-wise median (L1), geometric median
    (L2), or a value found via gradient descent (Lp for p > 2,
    euclidean^n).

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters.
    init : {'++', 'forgy', 'random_partition'}, default='++'
        Initialization strategy.  ``'++'`` uses a cost-aware variant of
        k-means++ seeding (recommended).
    plusplus_dist : str or None, default=None
        Override the distance metric used for the k-means++ seeding step.
        Useful when the fitting cost is expensive (e.g. ``euclidean^4``)
        but you want ``'squared_euclidean'`` seeding.
    n_init : int, default=10
        Number of independent random initializations.  The run with the
        lowest final inertia is kept.  Higher values reduce sensitivity to
        bad starts.
    max_iter : int, default=300
        Maximum number of assignment/update iterations per run.
    random_state : int or None, default=None
        Seed for reproducibility.
    verbose : bool, default=False
        Print a progress bar (requires tqdm) or per-iteration status.

    Attributes (set after ``fit``)
    --------------------------------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
        Sum of per-sample distances to their assigned center.
    n_iter_ : int
        Number of iterations in the best run.
    inertia_history_ : list of float
        Per-iteration inertia from the best run — useful for convergence
        diagnostics (``model.plot_convergence()``).

    Examples
    --------
    >>> from kgencenters import KGenCenters
    >>> import numpy as np
    >>> X = np.random.randn(200, 2)

    # Built-in metric
    >>> model = KGenCenters(n_clusters=3, init='++', n_init=10, random_state=0)
    >>> model.fit(X, cost_metric='euclidean')

    # Custom metric (Huber-like)
    >>> def huber_dist(X, centers, delta=1.0):
    ...     diff = X[:, None, :] - centers[None, :, :]
    ...     norms = np.linalg.norm(diff, axis=-1)
    ...     return np.where(norms < delta, 0.5 * norms**2, delta * norms - 0.5 * delta**2)
    >>> model.fit(X, cost_metric=huber_dist)

    # Custom metric + custom center function
    >>> def huber_center(pts, delta=1.0):
    ...     # gradient descent minimizer of sum of Huber(||c - x_i||)
    ...     return KGenCenters._gradient_descent_center(pts, pts.mean(0), p=2, n=1)
    >>> model.fit(X, cost_metric=huber_dist, center_fn=huber_center)

    # Compare metrics at once
    >>> KGenCenters.compare_metrics(X, n_clusters=3)

    # Elbow plot
    >>> KGenCenters.elbow_plot(X, cost_metric='euclidean')
    """

    # Canonical aliases for distance metrics
    _ALIASES: dict[str, str] = {
        'euclidean^2': 'squared_euclidean',
        'L2^2':        'squared_euclidean',
        'euclidean^1': 'euclidean',
        'L2':          'euclidean',
        'L1':          'manhattan',
    }

    def __init__(
        self,
        n_clusters: int = 3,
        init: str = '++',
        plusplus_dist: Optional[str] = None,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.plusplus_dist = plusplus_dist
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        # Set by fit()
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.inertia_history_: Optional[list] = None

    def __repr__(self) -> str:
        return (
            f"KGenCenters(n_clusters={self.n_clusters}, init='{self.init}', "
            f"n_init={self.n_init}, max_iter={self.max_iter}, "
            f"random_state={self.random_state})"
        )

    # ------------------------------------------------------------------
    # Metric validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_Lp(metric: str) -> bool:
        """Return True if metric matches 'Lp' for p ≥ 1 (e.g. 'L3', 'L1.5')."""
        return bool(re.match(r'^L([1-9]\d*(\.\d+)?)$', metric))

    @staticmethod
    def _is_euclidean_power(metric: str) -> bool:
        """Return True if metric matches 'euclidean^n' for integer n ≥ 1."""
        return bool(re.match(r'^euclidean\^([1-9]\d*)$', metric))

    @classmethod
    def _normalize_metric(cls, metric: Union[str, Callable]) -> Union[str, Callable]:
        """Resolve canonical aliases."""
        if callable(metric):
            return metric
        return cls._ALIASES.get(metric, metric)

    @classmethod
    def _validate_metric(cls, metric: Union[str, Callable]) -> None:
        if callable(metric):
            return
        known = {'squared_euclidean', 'euclidean', 'manhattan'}
        m = cls._normalize_metric(metric)
        if m in known or cls._is_Lp(m) or cls._is_euclidean_power(m):
            return
        raise ValueError(
            f"Unknown cost_metric '{metric}'. "
            "Built-in options: 'squared_euclidean', 'euclidean', 'manhattan', "
            "'Lp' (e.g. 'L3', 'L1.5'), 'euclidean^n' (e.g. 'euclidean^3'). "
            "Alternatively, pass a callable (X, centers) -> (n_samples, n_clusters) "
            "distance array."
        )

    # ------------------------------------------------------------------
    # Distance calculation
    # ------------------------------------------------------------------

    def _calculate_distances(
        self,
        X: np.ndarray,
        cost_metric: Union[str, Callable],
        centers: np.ndarray,
    ) -> np.ndarray:
        """
        Return (n_samples, n_clusters) distance matrix.

        Fully vectorised — no Python loops over samples or centers.
        """
        if callable(cost_metric):
            return cost_metric(X, centers)

        m = self._normalize_metric(cost_metric)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, K, d)

        if m == 'squared_euclidean':
            return np.einsum('ijk,ijk->ij', diff, diff)  # faster than sum(diff**2)

        if m == 'manhattan':
            return np.sum(np.abs(diff), axis=-1)

        if m == 'euclidean':
            return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))

        if self._is_Lp(m):
            p = float(m[1:])
            if p == 2.0:
                return np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
            abs_diff = np.abs(diff)
            return np.sum(abs_diff ** p, axis=-1) ** (1.0 / p)

        if self._is_euclidean_power(m):
            n = int(m[10:])
            sq_norms = np.einsum('ijk,ijk->ij', diff, diff)  # (N, K)
            if n == 1:
                return np.sqrt(np.maximum(sq_norms, 0.0))
            if n == 2:
                return sq_norms
            # Overflow-safe via log-space: exp(n * log(||·||))
            with np.errstate(divide='ignore'):
                log_norms = np.where(sq_norms > 0, 0.5 * n * np.log(sq_norms), -np.inf)
            return np.exp(log_norms)

        raise ValueError(f"Unrecognised metric '{cost_metric}'.")  # should not reach here

    # ------------------------------------------------------------------
    # Weiszfeld: geometric median
    # ------------------------------------------------------------------

    @staticmethod
    def _weiszfeld(X: np.ndarray, tolerance: float = 1e-5, max_steps: int = 100) -> np.ndarray:
        """
        Geometric median of rows of X via Weiszfeld iteration.

        Numerically robust: points that coincide with the current estimate
        are excluded from the weight update rather than receiving infinite
        weight.
        """
        center = np.mean(X, axis=0)
        for _ in range(max_steps):
            dists = np.linalg.norm(X - center, axis=1)
            mask = dists > 1e-10          # exclude points on the current estimate
            if not np.any(mask):
                break                     # all points collapsed — center is exact
            w = 1.0 / dists[mask]
            new_center = np.average(X[mask], weights=w, axis=0)
            if np.linalg.norm(new_center - center) < tolerance:
                center = new_center
                break
            center = new_center
        return center

    # ------------------------------------------------------------------
    # Gradient descent center (Lp / euclidean^n)
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_descent_center(
        pts: np.ndarray,
        init_center: np.ndarray,
        p: float,
        n: int,
        tolerance: float = 1e-5,
        max_descents: int = 300,
        descent_rate: Optional[float] = None,
    ) -> np.ndarray:
        """
        Minimise  f(c) = (1/N) * Σ_i  ||c - x_i||_p^n  over c.

        Uses gradient descent with backtracking (Armijo) line search.
        Step size is auto-scaled to the data's standard deviation when
        ``descent_rate`` is None.
        """
        c = init_center.copy().astype(float)

        # Auto step-size: scale to typical spread of points in cluster
        if descent_rate is None:
            scale = np.mean(np.std(pts, axis=0))
            lr = 0.1 * scale if scale > 1e-10 else 0.01
        else:
            lr = float(descent_rate)

        N = len(pts)

        for _ in range(max_descents):
            # ---- analytical gradient ----
            grad = np.zeros_like(c)
            for x in pts:
                diff = c - x
                abs_diff = np.abs(diff)
                nz = abs_diff > 1e-12
                if not np.any(nz):
                    continue
                # ||c - x||_p
                lp_val = np.sum(abs_diff[nz] ** p) ** (1.0 / p)
                if lp_val < 1e-12:
                    continue
                # d/dc_j ||c-x||_p^n
                # = n * ||c-x||_p^(n-p) * sign(diff_j) * |diff_j|^(p-1)
                coeff = n * lp_val ** (n - p)
                g = coeff * np.sign(diff[nz]) * abs_diff[nz] ** (p - 1)
                grad[nz] += np.nan_to_num(g)
            grad /= N

            grad_norm = np.linalg.norm(grad)
            if grad_norm < tolerance:
                break

            # ---- backtracking line search (Armijo) ----
            def _loss(c_):
                return sum(
                    np.sum(np.abs(c_ - x) ** p) ** (n / p) for x in pts
                ) / N

            f0 = _loss(c)
            step = lr
            for _ in range(30):
                if _loss(c - step * grad) < f0 - 1e-4 * step * grad_norm ** 2:
                    break
                step *= 0.5

            c = c - step * grad

        return c

    # ------------------------------------------------------------------
    # k-means++ initialisation
    # ------------------------------------------------------------------

    def _plusplus(
        self,
        X: np.ndarray,
        cost_metric: Union[str, Callable],
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Probabilistic seeding: choose centers proportional to min-distance squared."""
        centers = np.empty((self.n_clusters, X.shape[1]))
        centers[0] = X[rng.randint(X.shape[0])]

        for k in range(1, self.n_clusters):
            dists = self._calculate_distances(X, cost_metric, centers[:k])
            min_dists = np.min(dists, axis=1)
            total = min_dists.sum()
            probs = min_dists / total if total > 0 else np.ones(X.shape[0]) / X.shape[0]
            centers[k] = X[rng.choice(X.shape[0], p=probs)]

        return centers

    # ------------------------------------------------------------------
    # Single fit run
    # ------------------------------------------------------------------

    def _fit_once(
        self,
        X: np.ndarray,
        link_labels: Optional[np.ndarray],
        cost_metric: Union[str, Callable],
        center_fn: Optional[Callable],
        tolerance: float,
        max_steps: int,
        descent_rate: Optional[float],
        max_descents: int,
        rng: np.random.RandomState,
    ):
        """One complete run of the k-GenCenters algorithm."""

        # ---- initialise centers ----
        if self.init == 'forgy':
            centers = X[rng.choice(X.shape[0], size=self.n_clusters, replace=False)].copy()
        elif self.init == 'random_partition':
            asst = rng.randint(0, self.n_clusters, size=X.shape[0])
            centers = np.array([
                X[asst == i].mean(axis=0) if np.any(asst == i) else X[rng.randint(X.shape[0])]
                for i in range(self.n_clusters)
            ])
        elif self.init == '++':
            pp_metric = (
                self._normalize_metric(self.plusplus_dist)
                if self.plusplus_dist is not None
                else cost_metric
            )
            centers = self._plusplus(X, pp_metric, rng)
        else:
            raise ValueError(
                f"Invalid init='{self.init}'. Choose '++', 'forgy', or 'random_partition'."
            )

        # ---- constraint book-keeping ----
        if link_labels is None:
            ll = np.full(X.shape[0], np.nan, dtype=float)
        else:
            ll = np.asarray(link_labels, dtype=float)

        unconstrained = np.isnan(ll)
        constraint_groups: dict[float, np.ndarray] = {}
        for lbl in np.unique(ll[~unconstrained]):
            constraint_groups[lbl] = np.where(ll == lbl)[0]

        assigns = np.full(X.shape[0], -1, dtype=int)
        inertia_history: list[float] = []
        scale = np.mean(np.std(X, axis=0)) or 1.0

        for iteration in range(self.max_iter):
            distances = self._calculate_distances(X, cost_metric, centers)

            # assignment: constrained groups
            for indices in constraint_groups.values():
                assigns[indices] = np.argmin(np.sum(distances[indices], axis=0))

            # assignment: unconstrained points
            if np.any(unconstrained):
                assigns[unconstrained] = np.argmin(distances[unconstrained], axis=1)

            inertia_history.append(
                float(np.sum(distances[np.arange(X.shape[0]), assigns]))
            )

            new_centers = self._update_centers(
                X, assigns, centers, cost_metric,
                center_fn=center_fn,
                tolerance=tolerance, max_steps=max_steps,
                descent_rate=descent_rate, max_descents=max_descents,
            )

            # scale-aware convergence check
            if np.allclose(centers, new_centers, atol=tolerance * scale, rtol=0):
                centers = new_centers
                break
            centers = new_centers

        return centers, assigns.copy(), iteration + 1, inertia_history

    # ------------------------------------------------------------------
    # Center update dispatch
    # ------------------------------------------------------------------

    def _update_centers(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centers: np.ndarray,
        cost_metric: Union[str, Callable],
        center_fn: Optional[Callable],
        tolerance: float,
        max_steps: int,
        descent_rate: Optional[float],
        max_descents: int,
    ) -> np.ndarray:
        n_clusters = centers.shape[0]
        new_centers = centers.copy()
        m = self._normalize_metric(cost_metric) if isinstance(cost_metric, str) else cost_metric

        for k in range(n_clusters):
            pts = X[labels == k]
            if len(pts) == 0:
                # Empty cluster: reinitialise to a random data point
                new_centers[k] = X[np.random.randint(X.shape[0])]
                continue

            # user-supplied center function takes priority
            if center_fn is not None:
                new_centers[k] = center_fn(pts)
                continue

            # callable cost: use scipy minimise as fallback
            if callable(m):
                def _obj(c, pts=pts, m=m):
                    return float(np.sum(m(pts, c[np.newaxis, :])))
                res = spo.minimize(_obj, centers[k], method='L-BFGS-B',
                                   options={'maxiter': max_descents, 'ftol': tolerance})
                new_centers[k] = res.x
                continue

            if m == 'squared_euclidean':
                new_centers[k] = np.mean(pts, axis=0)

            elif m == 'manhattan':
                new_centers[k] = np.median(pts, axis=0)

            elif m == 'euclidean':
                new_centers[k] = self._weiszfeld(pts, tolerance, max_steps)

            elif self._is_Lp(m):
                p = float(m[1:])
                if p == 1.0:
                    new_centers[k] = np.median(pts, axis=0)
                elif p == 2.0:
                    new_centers[k] = self._weiszfeld(pts, tolerance, max_steps)
                else:
                    new_centers[k] = self._gradient_descent_center(
                        pts, centers[k], p=p, n=1,
                        tolerance=tolerance, max_descents=max_descents,
                        descent_rate=descent_rate,
                    )

            elif self._is_euclidean_power(m):
                n = int(m[10:])
                if n == 1:
                    new_centers[k] = self._weiszfeld(pts, tolerance, max_steps)
                elif n == 2:
                    new_centers[k] = np.mean(pts, axis=0)
                else:
                    new_centers[k] = self._gradient_descent_center(
                        pts, centers[k], p=2, n=n,
                        tolerance=tolerance, max_descents=max_descents,
                        descent_rate=descent_rate,
                    )

        return new_centers

    # ------------------------------------------------------------------
    # Public: fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        link_labels=None,
        cost_metric: Union[str, Callable] = 'squared_euclidean',
        center_fn: Optional[Callable] = None,
        tolerance: float = 1e-5,
        max_steps: int = 100,
        descent_rate: Optional[float] = None,
        max_descents: int = 300,
    ) -> 'KGenCenters':
        """
        Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        link_labels : array-like of shape (n_samples,), optional
            Must-link constraints.  Points sharing the same non-NaN integer
            value are forced into the same cluster at every iteration.
        cost_metric : str or callable, default='squared_euclidean'
            The cost used for assignment and center update.

            Built-in strings
            ~~~~~~~~~~~~~~~~
            ``'squared_euclidean'``  (aliases: ``'euclidean^2'``, ``'L2^2'``)
                Center = mean.  Standard K-Means.
            ``'euclidean'``  (aliases: ``'euclidean^1'``, ``'L2'``)
                Center = geometric median (Weiszfeld).
            ``'manhattan'``  (alias: ``'L1'``)
                Center = coordinate-wise median.
            ``'Lp'``  (e.g. ``'L3'``, ``'L1.5'``)
                Lp norm for any p ≥ 1.  Center found via gradient descent.
            ``'euclidean^n'``  (e.g. ``'euclidean^3'``)
                n-th power of Euclidean distance.  Center via gradient descent.

            Callable
            ~~~~~~~~
            ``cost_metric(X, centers) -> ndarray of shape (n_samples, n_clusters)``
                Fully custom distance matrix.  Pair with ``center_fn`` to
                specify a matching center-update rule.

        center_fn : callable or None, default=None
            Custom center-update function ``center_fn(cluster_points) -> center``.
            Overrides the built-in center computation for every cluster.
            Useful when pairing a callable ``cost_metric`` with an analytical
            minimizer, or when experimenting with non-standard geometries.

        tolerance : float, default=1e-5
            Convergence tolerance for Weiszfeld and gradient descent.
        max_steps : int, default=100
            Max Weiszfeld iterations (``'euclidean'`` metric).
        descent_rate : float or None, default=None
            Learning rate for gradient descent (``Lp`` / ``euclidean^n``).
            If None, auto-scaled to ``0.1 × std(X)``.
        max_descents : int, default=300
            Max gradient descent steps per center update per iteration.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        cost_metric = self._normalize_metric(cost_metric)
        self._validate_metric(cost_metric)

        rng = np.random.RandomState(self.random_state)

        best_centers = None
        best_labels = None
        best_inertia = np.inf
        best_n_iter = 0
        best_history: list[float] = []

        n_init = 1 if self.init == 'random_partition' else self.n_init

        runs = range(n_init)
        if self.verbose and _TQDM:
            runs = _tqdm(runs, desc=f"KGenCenters [{cost_metric}]", leave=False)

        for _ in runs:
            seed = rng.randint(0, 2 ** 31)
            run_rng = np.random.RandomState(seed)
            centers, labels, n_iter, history = self._fit_once(
                X, link_labels, cost_metric, center_fn,
                tolerance, max_steps, descent_rate, max_descents, run_rng,
            )
            cur_inertia = history[-1] if history else np.inf
            if cur_inertia < best_inertia:
                best_inertia = cur_inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter
                best_history = history

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        self.inertia_history_ = best_history

        # backward-compatible aliases
        self.centers = self.cluster_centers_
        self.assigns = self.labels_

        return self

    # ------------------------------------------------------------------
    # Public: predict / fit_predict / score
    # ------------------------------------------------------------------

    def fit_predict(self, X, link_labels=None, cost_metric='squared_euclidean', **kwargs) -> np.ndarray:
        """Fit and return cluster labels."""
        return self.fit(X, link_labels=link_labels, cost_metric=cost_metric, **kwargs).labels_

    def predict(self, X, cost_metric=None) -> np.ndarray:
        """
        Assign new points to the nearest center.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        cost_metric : str or callable
            Must match the metric used during ``fit``.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Call fit() before predict().")
        if cost_metric is None:
            raise ValueError("Provide cost_metric (must match the one used in fit).")
        X = np.asarray(X, dtype=float)
        cost_metric = self._normalize_metric(cost_metric)
        return np.argmin(
            self._calculate_distances(X, cost_metric, self.cluster_centers_), axis=1
        )

    def score(self, X, cost_metric=None) -> float:
        """Negative inertia (sklearn convention: higher is better)."""
        if cost_metric is None:
            raise ValueError("Provide cost_metric.")
        return -self.inertia(X, cost_metric)

    # ------------------------------------------------------------------
    # Public: evaluate / inertia
    # ------------------------------------------------------------------

    def evaluate(self, true_labels) -> float:
        """
        Clustering accuracy against ground-truth labels via the Hungarian
        algorithm (optimal label permutation).

        Parameters
        ----------
        true_labels : array-like of shape (n_samples,)

        Returns
        -------
        float in [0, 1]
        """
        if self.labels_ is None:
            raise ValueError("Call fit() first.")
        confusion = skm.confusion_matrix(true_labels, self.labels_)
        row_ind, col_ind = spo.linear_sum_assignment(confusion, maximize=True)
        return confusion[row_ind, col_ind].sum() / confusion.sum()

    def inertia(self, X, cost_metric=None) -> float:
        """
        Sum of per-sample distances to assigned centers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        cost_metric : str or callable
        """
        if cost_metric is None:
            raise ValueError("Provide cost_metric.")
        if self.labels_ is None:
            raise ValueError("Call fit() first.")
        X = np.asarray(X, dtype=float)
        cost_metric = self._normalize_metric(cost_metric)
        dists = self._calculate_distances(X, cost_metric, self.cluster_centers_)
        return float(np.sum(dists[np.arange(X.shape[0]), self.labels_]))

    # ------------------------------------------------------------------
    # Public: voronoi
    # ------------------------------------------------------------------

    def voronoi(
        self,
        cost_metric=None,
        x_range=(-10, 10),
        y_range=(-10, 10),
        resolution: int = 200,
    ):
        """
        Compute Voronoi decision boundaries for 2-D data.

        Parameters
        ----------
        cost_metric : str or callable
        x_range, y_range : (float, float)
            Axis extents of the grid.  Auto-detected from cluster centers if
            omitted — but explicit values are recommended.
        resolution : int, default=200

        Returns
        -------
        boundaries : ndarray of bool, shape (resolution, resolution)
            True on cell boundaries.
        xx, yy : ndarray
            The meshgrid axes (for use with ``plt.contourf(xx, yy, boundaries)``).

        Raises
        ------
        ValueError
            If data are not 2-dimensional.  For higher-dimensional data,
            project to 2-D first (e.g. with PCA) before calling voronoi.

        Example
        -------
        >>> boundaries, xx, yy = model.voronoi('euclidean')
        >>> plt.contourf(xx, yy, boundaries, alpha=0.3)
        >>> plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Call fit() first.")
        if cost_metric is None:
            raise ValueError("Provide cost_metric.")
        if self.cluster_centers_.shape[1] != 2:
            raise ValueError(
                f"voronoi() requires 2-D data; got {self.cluster_centers_.shape[1]} features. "
                "Project to 2-D first (e.g. sklearn.decomposition.PCA(n_components=2))."
            )

        cost_metric = self._normalize_metric(cost_metric)
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x, y)
        grid = np.c_[xx.ravel(), yy.ravel()]

        dists = self._calculate_distances(grid, cost_metric, self.cluster_centers_)
        region = np.argmin(dists, axis=1).reshape(resolution, resolution)

        bx = np.diff(region, axis=0) != 0
        by = np.diff(region, axis=1) != 0
        bx = np.pad(bx, ((0, 1), (0, 0)), constant_values=0)
        by = np.pad(by, ((0, 0), (0, 1)), constant_values=0)
        boundaries = np.logical_or(bx, by)

        return boundaries, xx, yy

    # ------------------------------------------------------------------
    # Class utility: compare_metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compare_metrics(
        X,
        true_labels=None,
        metrics: Sequence = (
            'squared_euclidean', 'euclidean', 'manhattan', 'L3',
        ),
        n_clusters: int = 3,
        n_init: int = 5,
        random_state: int = 0,
        **fit_kwargs,
    ):
        """
        Fit KGenCenters with several cost metrics and return a comparison table.

        Parameters
        ----------
        X : array-like
        true_labels : array-like or None
            If provided, Hungarian-matched accuracy is included.
        metrics : sequence of str or callables
        n_clusters : int
        n_init : int
        random_state : int
        **fit_kwargs : forwarded to ``fit()``

        Returns
        -------
        pandas.DataFrame (if pandas is installed) or dict, indexed by metric.
        Columns: ``inertia``, ``n_iter``, ``accuracy`` (if true_labels given).

        Example
        -------
        >>> KGenCenters.compare_metrics(X, true_labels=y, n_clusters=3)
        """
        results = {}
        for m in metrics:
            label = m if isinstance(m, str) else getattr(m, '__name__', repr(m))
            model = KGenCenters(
                n_clusters=n_clusters, n_init=n_init,
                random_state=random_state, verbose=False,
            )
            model.fit(X, cost_metric=m, **fit_kwargs)
            entry: dict = {'inertia': model.inertia_, 'n_iter': model.n_iter_}
            if true_labels is not None:
                entry['accuracy'] = round(model.evaluate(true_labels), 4)
            results[label] = entry

        try:
            import pandas as pd
            return pd.DataFrame(results).T
        except ImportError:
            return results

    # ------------------------------------------------------------------
    # Instance utility: elbow_plot
    # ------------------------------------------------------------------

    @staticmethod
    def elbow_plot(
        X,
        cost_metric: Union[str, Callable] = 'squared_euclidean',
        k_range: Iterable[int] = range(1, 11),
        n_init: int = 5,
        random_state: int = 0,
        ax=None,
        **fit_kwargs,
    ):
        """
        Plot inertia vs. k to aid in choosing the number of clusters.

        Parameters
        ----------
        X : array-like
        cost_metric : str or callable
        k_range : iterable of ints, default range(1, 11)
        n_init, random_state : passed to KGenCenters
        ax : matplotlib Axes, optional

        Returns
        -------
        ks : list of int
        inertias : list of float

        Example
        -------
        >>> ks, inertias = KGenCenters.elbow_plot(X, 'euclidean', k_range=range(1, 8))
        """
        import matplotlib.pyplot as plt

        ks, inertias = [], []
        for k in k_range:
            model = KGenCenters(
                n_clusters=k, n_init=n_init,
                random_state=random_state, verbose=False,
            )
            model.fit(X, cost_metric=cost_metric, **fit_kwargs)
            ks.append(k)
            inertias.append(model.inertia_)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, inertias, 'o-', color='steelblue', linewidth=2)
        ax.set_xlabel('Number of clusters k')
        ax.set_ylabel('Inertia')
        label = cost_metric if isinstance(cost_metric, str) else getattr(cost_metric, '__name__', 'custom')
        ax.set_title(f'Elbow plot  [{label}]')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return ks, inertias

    # ------------------------------------------------------------------
    # Instance utility: plot_convergence
    # ------------------------------------------------------------------

    def plot_convergence(self, ax=None):
        """
        Plot per-iteration inertia from the best run of the last ``fit()`` call.

        Returns
        -------
        ax : matplotlib Axes

        Example
        -------
        >>> model.fit(X, cost_metric='euclidean')
        >>> model.plot_convergence()
        """
        if self.inertia_history_ is None:
            raise ValueError("Call fit() first.")
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.inertia_history_, color='steelblue', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Inertia')
        ax.set_title('Convergence  (best run)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return ax
