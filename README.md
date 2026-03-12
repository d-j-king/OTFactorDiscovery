# OTFactorDiscovery

Exploratory research code from the **AM-SURE 2023** summer program at NYU's Courant Institute of Mathematical Sciences, advised by **Esteban Tabak**. The project investigates how optimal transport (OT) ideas can inform factor discovery — identifying and isolating structure in data distributions.

This repository contains two related sub-projects. Each stands on its own but shares a common conceptual motivation: optimal transport as a lens for understanding what drives variability in data.

---

## Sub-projects

### 1. Generalized K-Means (`daniel/Clustering/`)

**Author:** Daniel Wang

Standard K-Means is implicitly tied to squared-Euclidean distance: the within-cluster minimizer of that cost is the mean. The key question here — *what happens when you swap in a different cost function?* The center that minimizes the new cost is no longer the mean. Depending on the metric it is a coordinate-wise median (L1), a geometric median (L2), or something computed via gradient descent (Lp, euclidean^n).

`KGenCenters` is a scikit-learn-compatible clustering class implementing this generalized K-Means framework. It inherits from `BaseEstimator` and `ClusterMixin`, so it plugs directly into sklearn pipelines and grid search.

**Supported cost metrics:**

| String | Description | Center update |
|---|---|---|
| `'squared_euclidean'` | Standard K-Means cost (alias: `'L2^2'`) | Mean |
| `'euclidean'` | L2 distance (alias: `'L2'`) | Weiszfeld geometric median |
| `'manhattan'` | L1 distance (alias: `'L1'`) | Coordinate-wise median |
| `'Lp'` (e.g. `'L3'`, `'L1.5'`) | Any Lp norm, p ≥ 1 | Gradient descent |
| `'euclidean^n'` (e.g. `'euclidean^3'`) | Euclidean distance to the n-th power | Gradient descent |
| Callable | `(X, centers) → (N, K)` distance array | User-supplied or gradient descent |

**Additional features:**
- `n_init`: multiple random restarts, keeps the best result by inertia
- `init='++`: cost-aware k-means++ seeding
- Must-link constraints via `link_labels`
- 2D Voronoi boundary visualization for any metric
- `compare_metrics()`: benchmark multiple cost metrics side by side
- `elbow_plot()`: inertia vs. k to help choose the number of clusters
- `plot_convergence()`: per-iteration inertia curve from the fitted run
- `score()` (negative inertia), `fit_predict()` for sklearn compatibility

**Quick start:**

```python
from modules.kgencenters import KGenCenters
import numpy as np

X = np.random.randn(300, 2)

# Fit with multiple restarts
model = KGenCenters(n_clusters=3, init='++', n_init=10, random_state=42)
model.fit(X, cost_metric='euclidean')

print(model.labels_)            # cluster assignments
print(model.cluster_centers_)   # (K, d) centers
print(model.inertia_)           # total within-cluster cost

# Predict on new data
labels_new = model.predict(X_new, cost_metric='euclidean')

# Evaluate against ground truth (Hungarian matching)
acc = model.evaluate(true_labels)

# Compare several metrics at once — returns a DataFrame
df = KGenCenters.compare_metrics(X, true_labels=true_labels, n_clusters=3)

# Elbow plot
KGenCenters.elbow_plot(X, cost_metric='squared_euclidean', k_range=range(1, 10))

# Convergence plot (after fit)
model.plot_convergence()
```

See `demo_for_new_users.ipynb` for a full walkthrough, and the other notebooks for initialization comparisons, outlier sensitivity studies, and experiments on real datasets (seeds, glass, E. coli, Parkinson's speech features).

---

### 2. Variability Reduction via Optimal Transport (`var-reduction/`)

**Author:** Kai M. Hung

Given data with an unwanted source of variability (a "factor" Z), find a transformed version Y of the data that is independent of Z while staying as close as possible to the original X. The mathematical object encoding this trade-off is an OT barycenter.

This sub-project implements a flow-based gradient descent approach to compute that barycenter. Independence is estimated using a kernel-based KL divergence (Gaussian kernel density), and the trade-off between reconstruction fidelity and independence is controlled by a regularisation parameter λ.

**Core functions in `flow_ot.py`:**

```python
from src.model.flow_ot import compute_barycenter, select_sigma, kl_barycenter_loss

# Automatic bandwidth selection (median heuristic)
sigma_z = select_sigma(Z)

# Run gradient descent — returns barycenter + per-iteration gradient norms
y, history = compute_barycenter(
    x, z, y_init, lam=1.0,
    sigma_y=None,   # None → auto-selected each iteration
    sigma_z=sigma_z,
    max_iter=1000, lr=0.01,
    growing_lambda=True, max_lambda=300.0
)

# Evaluate the full objective
loss = kl_barycenter_loss(y, x, z, lam=1.0, sigma_y=1.0, sigma_z=sigma_z)
```

**To run the main experiment:**

```bash
cd var-reduction
mkdir -p outputs
conda create --name otfactor --file requirements.txt
conda activate otfactor
python3 barycenter_fit.py
```

This outputs (1) a convergence plot of KL divergence over training and (2) an animation showing the distribution converging toward the barycenter.

See also:
- `flow_ot_exp.ipynb` — demonstration on the Iris dataset
- `gaussian.ipynb` — synthetic Gaussian experiments
- `ssl_test.ipynb` — semi-supervised variant (factor labels partially known)
- [Report](https://math.nyu.edu/media/math/filer_public/51/b1/51b198de-3072-4c10-b729-96111bbc661c/varreduceot.pdf) | [Slides](https://math.nyu.edu/media/math/filer_public/07/0c/070c1104-9061-4b11-bd0a-ae7ebc50d48d/variability_reduction_with_optimal_transport.pdf)

---

## Background & Related Work

The framing connecting both sub-projects — optimal transport, barycenters, and factor structure — is laid out in the program paper:

> **Optimal Transport for Factor Discovery** (AM-SURE 2023)
> [math.nyu.edu — am_sure_5.pdf](https://math.nyu.edu/media/math/filer_public/48/72/48728e1e-4bf3-4198-88c4-92ad56ac73cd/am_sure_5.pdf)

The `daniel/Saddle_point_problem/` directory contains additional optimization experiments — gradient descent and saddle-point solvers explored alongside the main clustering work.

---

## Notes on Scope

This code was written for learning and exploration. It is not a production library. A few things to keep in mind:

- `KGenCenters` follows the scikit-learn interface and is reasonably well-tested on the datasets in the repo. Center updates for `Lp` and `euclidean^n` metrics use backtracking line search (Armijo condition) with an auto-scaled step size; they are reliable but not tuned for speed.
- The var-reduction module is research-grade: the core algorithm is functional and validated on synthetic data, but it does not aim for the performance or generality of a polished package. Bandwidth selection defaults to the median heuristic (`select_sigma`), which works well for typical data but may need adjustment.
- Both sub-projects are designed to be readable and customizable rather than fast. The code structure is intentionally transparent.

---

## Context

AM-SURE (Applied Mathematics Summer Undergraduate Research Experience) is a research program at the Courant Institute of Mathematical Sciences at NYU. This work was completed in summer 2023 under the supervision of Esteban Tabak.
