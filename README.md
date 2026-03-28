# OTFactorDiscovery
AM-SURE 2023 Project: Optimal Transport for Factor Discovery | Daniel J. King, Kai M. Hung

Two sub-projects motivated by optimal transport as a framework for factor discovery.

**Variability Reduction via OT** (`var-reduction/`) — led by Kai M. Hung. Given data with an unwanted source of variability (a factor Z), find a transformed version that is independent of Z while staying close to the original. The mathematical object encoding this is an OT barycenter. [[report](https://math.nyu.edu/media/math/filer_public/51/b1/51b198de-3072-4c10-b729-96111bbc661c/varreduceot.pdf)] [[slides](https://math.nyu.edu/media/math/filer_public/07/0c/070c1104-9061-4b11-bd0a-ae7ebc50d48d/variability_reduction_with_optimal_transport.pdf)]

**Generalized K-Means** (`gen-centers/`) — led by Daniel J. King. In the categorical-factor case, factor discovery reduces to a clustering problem. Taking an OT lens yields generalizations of K-Means through new transport costs and initialization schemes. [[report](https://math.nyu.edu/media/math/filer_public/48/72/48728e1e-4bf3-4198-88c4-92ad56ac73cd/am_sure_5.pdf)] [[slides](https://math.nyu.edu/media/math/filer_public/19/94/19947867-f928-4bdd-b2ec-4a97cd1f566a/final_presentation_clustering.pdf)]

See each subdirectory's `README.md` for details.
