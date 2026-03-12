# OTFactorDiscovery

Exploratory research code from the **AM-SURE 2023** summer program at NYU's Courant Institute of Mathematical Sciences, advised by **Esteban Tabak**. The project investigates how optimal transport (OT) ideas can inform factor discovery — identifying and isolating structure in data distributions.

This repository contains two related sub-projects. Each stands on its own but shares a common conceptual motivation: optimal transport as a lens for understanding what drives variability in data.

---

## Sub-projects

### 1. Generalized K-Means (`daniel/Clustering/`)

**Author:** Daniel Wang

Standard K-Means is tied to squared Euclidean distance because the cluster center that minimizes that cost is the mean. The key question here: *what happens when you swap in a different cost function?* The center that minimizes the new cost is no longer the mean — it could be a median, a geometric median, or something requiring gradient descent to find.

`KGenCenters` is a scikit-learn-style clustering class that implements this generalized K-Means framework. It supports:

- **Cost metrics:** `squared_euclidean`, `euclidean`, `manhattan` (L1), `Lp` for any p ≥ 1, and `euclidean^n` for positive integer powers
- **Center update strategies:** mean (L2²), coordinate-wise median (L1), Weiszfeld algorithm (L2), gradient descent (general Lp / euclidean^n)
- **Initialization:** Forgy, random partition, and k-means++ (with optional override distance for the probabilistic seeding step)
- **Must-link constraints:** a group of points can be forced into the same cluster via a `link_labels` array
- **Voronoi boundaries:** compute the decision boundaries for any supported metric in 2D

Different cost metrics satisfy different properties — some are proper metrics (L1, L2), others are not (L2² is not a metric, euclidean^n for n > 1 is not a metric), and the geometry of the resulting Voronoi cells changes accordingly. The notebooks in `daniel/Clustering/` explore these differences and test on several real datasets.

**Interface:**

```python
from modules.kgencenters import KGenCenters

model = KGenCenters(n_clusters=3, init='++', random_state=42)
model.fit(X, cost_metric='euclidean')
labels = model.predict(X_new, cost_metric='euclidean')
accuracy = model.evaluate(true_labels)
inertia  = model.inertia(X, cost_metric='euclidean')
```

See `demo_for_new_users.ipynb` for a walkthrough, and the other notebooks for initialization comparisons, outlier sensitivity studies, and real-data experiments (seeds, glass, E. coli, Parkinson's speech features).

---

### 2. Variability Reduction via Optimal Transport (`var-reduction/`)

**Author:** Kai M. Hung

Factor discovery can be framed as: given data with an unwanted source of variability (a "factor" Z), find a transformed version of the data that is independent of Z while staying as close as possible to the original. The mathematical object encoding "as close as possible" is an OT barycenter.

This sub-project implements a flow-based gradient descent approach to compute that barycenter. Independence between the transformed data Y and the factor Z is estimated using a kernel-based divergence (KL divergence via Gaussian kernels), and the trade-off between closeness and independence is controlled by a regularization parameter λ.

To run the main experiment:

```bash
cd var-reduction
conda create --name otfactor --file requirements.txt
conda activate otfactor
python3 barycenter_fit.py
```

This outputs (1) a convergence plot for KL divergence and (2) an animation showing the distribution converging to the barycenter.

See also:
- `flow_ot_exp.ipynb` — demonstration on the Iris dataset
- `gaussian.ipynb` — synthetic Gaussian experiments
- `ssl_test.ipynb` — semi-supervised variant
- [Report](https://math.nyu.edu/media/math/filer_public/51/b1/51b198de-3072-4c10-b729-96111bbc661c/varreduceot.pdf) | [Slides](https://math.nyu.edu/media/math/filer_public/07/0c/070c1104-9061-4b11-bd0a-ae7ebc50d48d/variability_reduction_with_optimal_transport.pdf)

---

## Background & Related Work

The framing connecting both sub-projects — optimal transport, barycenters, and factor structure — is laid out in the program paper:

> **Optimal Transport for Factor Discovery** (AM-SURE 2023)  
> [math.nyu.edu — am_sure_5.pdf](https://math.nyu.edu/media/math/filer_public/48/72/48728e1e-4bf3-4198-88c4-92ad56ac73cd/am_sure_5.pdf)

The Saddle Point Problem directory (`daniel/Saddle_point_problem/`) contains additional optimization experiments — gradient descent / saddle point solvers explored alongside the main clustering work.

---

## Notes on Scope

This code was written for learning and exploration. It is not a production library. A few things to keep in mind if you use or adapt it:

- `KGenCenters` is well-tested on the datasets in the repo, but edge cases (particularly for `euclidean^n` with large n) are noted in the code. Gradient descent-based center updates use fixed hyperparameters that may need tuning for different datasets.
- The var-reduction module is research-grade: the core algorithm is functional and validated on synthetic data, but it does not aim for the performance or generality of a polished package.
- Both sub-projects are designed to be readable and customizable rather than fast. If you are adapting them, the code structure is intentionally transparent.

---

## Context

AM-SURE (Applied Mathematics Summer Undergraduate Research Experience) is a research program at the Courant Institute of Mathematical Sciences at NYU. This work was completed in summer 2023 under the supervision of Esteban Tabak.
