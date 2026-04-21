# gen-centers

Generalized K-Means clustering with configurable cost metrics. Research code from the **AM-SURE 2023** program at NYU's Courant Institute of Mathematical Sciences, advised by **Esteban Tabak**.

Standard K-Means is implicitly tied to squared-Euclidean distance: the within-cluster minimizer of that cost is the mean. Swapping in a different cost function changes what the center update looks like — it becomes a coordinate-wise median (L1), a geometric median (L2), or something requiring gradient descent (Lp, euclidean^n).

The framing comes from optimal transport: in the categorical-factor case, factor discovery reduces to a clustering problem, and OT yields natural generalizations of K-Means through new transport costs and initialization schemes. See the [project report](https://math.nyu.edu/media/math/filer_public/48/72/48728e1e-4bf3-4198-88c4-92ad56ac73cd/am_sure_5.pdf) and [slides](https://math.nyu.edu/media/math/filer_public/19/94/19947867-f928-4bdd-b2ec-4a97cd1f566a/final_presentation_clustering.pdf).

---

## Overview

`KGenCenters` is a scikit-learn-compatible clustering class implementing the generalized K-Means framework. It inherits from `BaseEstimator` and `ClusterMixin`, so it plugs directly into sklearn pipelines and grid search.

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
- `init='++'`: cost-aware k-means++ seeding
- Must-link constraints via `link_labels`
- 2D Voronoi boundary visualization for any metric
- `compare_metrics()`: benchmark multiple cost metrics side by side
- `elbow_plot()`: inertia vs. k to help choose the number of clusters
- `plot_convergence()`: per-iteration inertia curve from the fitted run
- `score()` (negative inertia), `fit_predict()` for sklearn compatibility

---

## Quick Start

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

## Notes on Scope

This code was written for learning and exploration. It is not a production library.

`KGenCenters` follows the scikit-learn interface and is reasonably well-tested on real datasets (seeds, glass, E. coli, Parkinson's speech features). Center updates for `Lp` and `euclidean^n` metrics use backtracking line search (Armijo condition) with an auto-scaled step size — reliable, but not tuned for speed. The code is intentionally transparent and customizable rather than optimized for speed.
