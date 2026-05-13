## CAKE: Confidence in Assignments via K-partition Ensembles


<p align="center">
  <a href="https://pypi.org/project/cake-ensemble/"><img src="https://img.shields.io/pypi/v/cake-ensemble.svg?color=blue" alt="PyPI version"></a><a href="https://pypi.org/project/cake-ensemble/"><img src="https://img.shields.io/pypi/pyversions/cake-ensemble.svg" alt="Python versions"></a><a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a><a href="https://pepy.tech/project/cake-ensemble"><img src="https://pepy.tech/badge/cake-ensemble" alt="Downloads"></a>
</p>


Accepted for publication in *Machine Learning with Applications*.


Clustering assigns each data point to a group, but it does not tell us how reliable that assignment is.  
While global validation metrics assess overall quality, they provide little insight into the trustworthiness of *individual* assignments.

Ensemble-based clustering improves robustness by aggregating multiple partitions. However, most ensemble-style uncertainty measures focus primarily on **cross-run agreement** (e.g., aligned vote counts or dispersion of labels across runs). Agreement alone is not sufficient: a point may be consistently assigned due to systematic bias or rigid decision boundaries, even if it is weakly integrated into the cluster structure.

Conversely, purely geometric signals such as the Silhouette score evaluate local separation within a single run but ignore cross-run instability. A point may appear geometrically well placed in one partition, yet switch clusters across runs because it lies near a boundary or because multiple partitions explain it nearly equally well.

<img src="https://raw.githubusercontent.com/semoglou/cake/main/figs/failures.png" alt="Assignment Stability vs Geometric Consistency" width="800">

These complementary failure modes motivate a confidence signal that jointly accounts for both **assignment stability** and **consistency of geometric support** at the point level.

**CAKE** (Confidence in Assignments via K-partition Ensembles) provides a principled, per-instance confidence score by fusing two complementary statistics computed over a clustering ensemble:

- 🔁 **Assignment Stability**: pairwise agreement across runs after optimal label alignment, using the Hungarian algorithm.   
- 📐 **Geometric Consistency**: aggregated Silhouette statistics across runs.  

<img src="https://raw.githubusercontent.com/semoglou/cake/main/figs/frame.png" alt="CAKE pipeline" width="700">

These components are combined into a single, interpretable score in **[0, 1]**, enabling:

- Identification of stable core members.  
- Detection of ambiguous boundary points.  
- Filtering of unreliable assignments.  
- Ranking of instances by clustering confidence.  

<img src="https://raw.githubusercontent.com/semoglou/cake/main/figs/synthmap.png" alt="CAKE visualization" width="400">

<em>CAKE scores on synthetic data. Red ↑; Blue ↓.</em>

# 

## Citation

If you find this work useful, please consider citing:

Semoglou, A., & Pavlopoulos, J. (2026).  
**Assigning Confidence: K-partition Ensembles.**  
Preprint: [https://arxiv.org/abs/2602.18435](https://arxiv.org/abs/2602.18435)

```bibtex
@misc{semoglou2026assigningconfidencekpartitionensembles,
   title={Assigning Confidence: K-partition Ensembles},
   author={Aggelos Semoglou and John Pavlopoulos},
   year={2026},
   eprint={2602.18435},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2602.18435},
}
```

## Installation

Install **CAKE** from [PyPI](https://pypi.org/project/cake-ensemble/):

```bash
pip install cake-ensemble
```

Ιmport the main functions in Python as: 

```python
from cake_ensemble import (
    sil_samples,
    sil_samples_stats,
    align_labels,
    pairwise_stability,
    cake,
    consensus_labels,
    kmeans_ensemble,
    cake_with_consensus,
)
```

## API Reference

CAKE provides utilities for computing sample-level silhouette statistics, aligning clustering labels, measuring assignment stability, computing CAKE confidence scores, deriving consensus labels, and building KMeans ensembles.

---

#### `sil_samples`

Computes silhouette scores for each sample, either exactly using scikit-learn or approximately using distances to cluster centroids.

```python
sil_samples(X, labels, approximation=False, centers=None)
```

**Inputs**

- `X`: array-like of shape `(n_samples, n_features)`  
  Input data matrix.

- `labels`: array-like of shape `(n_samples,)`  
  Cluster label assigned to each sample.

- `approximation`: bool, default `False`  
  If `False`, computes exact silhouette scores.  
  If `True`, computes a faster centroid-based approximation.

- `centers`: array-like of shape `(n_clusters, n_features)` or None, default `None`  
  Optional cluster centers used when `approximation=True`.  
  If not provided, centers are computed from `X` and `labels`.

**Returns**

- `silhouette_scores`: array of shape `(n_samples,)`  
  Silhouette score for each sample.

---

#### `sil_samples_stats`

Computes sample-level silhouette statistics across multiple clustering runs.

```python
sil_samples_stats(X, labels_list, approximation=False, centers_list=None)
```

**Inputs**

- `X`: array-like of shape `(n_samples, n_features)`  
  Input data matrix.

- `labels_list`: list of array-like, each of shape `(n_samples,)`  
  List of label vectors from multiple clustering runs.

- `approximation`: bool, default `False`  
  Whether to use centroid-based approximate silhouette scores.

- `centers_list`: list of array-like or None, default `None`  
  Optional list of cluster-center arrays, one per clustering run.  
  Used when `approximation=True`.

**Returns**

- `mean_silhouette`: array of shape `(n_samples,)`  
  Mean silhouette score of each sample across the ensemble.

- `std_silhouette`: array of shape `(n_samples,)`  
  Standard deviation of each sample’s silhouette score across the ensemble.

---

#### `align_labels`

Aligns the labels of one clustering solution to another using the Hungarian algorithm.

```python
align_labels(target, source)
```

**Inputs**

- `target`: array-like of shape `(n_samples,)`  
  Reference label vector.

- `source`: array-like of shape `(n_samples,)`  
  Label vector to be remapped so that it best matches `target`.

**Returns**

- `aligned_labels`: array of shape `(n_samples,)`  
  The source labels remapped to match the target label space.

---

#### `pairwise_stability`

Computes pointwise assignment stability across all pairs of clustering runs.

```python
pairwise_stability(labels_runs)
```

**Inputs**

- `labels_runs`: list of array-like, each of shape `(n_samples,)`  
  List of cluster label vectors from multiple clustering runs.

**Returns**

- `assignment_stability`: array of shape `(n_samples,)`  
  For each sample, the fraction of run-pairs where the sample receives the same aligned label.

---

#### `cake`

Computes CAKE confidence scores for each sample in a clustering ensemble.

```python
cake(X, labels_list, method='product', approximation=False, centers_list=None, geom_norm='clip')
```

**Inputs**

- `X`: array-like of shape `(n_samples, n_features)`  
  Input data matrix.

- `labels_list`: list of array-like, each of shape `(n_samples,)`  
  List of cluster label vectors from multiple clustering runs.

- `method`: str, default `'product'`  
  Method used to combine assignment stability and geometric stability.  
  Options: `'product'` or `'harmonic_mean'`.

- `approximation`: bool, default `False`  
  Whether to use centroid-based approximate silhouette scores.

- `centers_list`: list of array-like or None, default `None`  
  Optional list of cluster centers, one per clustering run.  
  Used when `approximation=True`.

- `geom_norm`: str, default `'clip'`  
  Strategy for mapping the geometric component to `[0, 1]`.  
  `'clip'` uses max(mean silhouette − std silhouette, 0).  
  `'affine'` maps the raw value from `[-1, 1]` to `[0, 1]`.

**Returns**

- `cake_scores`: array of shape `(n_samples,)`  
  Final CAKE confidence score for each sample.

- `assignment_stability`: array of shape `(n_samples,)`  
  Pairwise label-agreement stability across clustering runs.

- `geometric_stability`: array of shape `(n_samples,)`  
  Silhouette-based geometric reliability score.

- `summary`: pandas DataFrame  
  Per-sample table containing mean silhouette, silhouette standard deviation, geometric stability, assignment stability, and CAKE score.

---

#### `consensus_labels`

Computes consensus clustering labels from multiple clustering runs using label alignment and majority vote.

```python
consensus_labels(labels_list, method='medoid')
```

**Inputs**

- `labels_list`: list of array-like, each of shape `(n_samples,)`  
  List of cluster label vectors from multiple clustering runs.

- `method`: str, default `'medoid'`  
  Consensus strategy.  
  `'medoid'` selects the most representative run as reference, aligns all runs to it, and computes majority-vote labels.  
  `'best_ref'` tries every run as reference and keeps the one with the highest average consensus strength.

**Returns**

- `consensus`: array of shape `(n_samples,)`  
  Consensus cluster label for each sample.

- `consensus_strength`: array of shape `(n_samples,)`  
  Fraction of aligned runs voting for the selected consensus label.

- `reference_index`: int  
  Index of the selected reference run.

Note: majority-vote consensus does not guarantee that all reference cluster labels appear in the final consensus.

---

#### `kmeans_ensemble`

Builds a KMeans clustering ensemble by running KMeans multiple times with different random seeds.

```python
kmeans_ensemble(
    X,
    n_clusters,
    n_runs=50,
    random_state=1,
    init='random',
    n_init=1,
    max_iter=300,
    tol=1e-4,
    algorithm='lloyd',
    return_models=False
)
```

**Inputs**

- `X`: array-like of shape `(n_samples, n_features)`  
  Input data matrix.

- `n_clusters`: int  
  Number of clusters for each KMeans run.

- `n_runs`: int, default `50`  
  Number of KMeans runs in the ensemble.

- `random_state`: int or None, default `1`  
  Base random seed used to generate run-specific seeds.

- `init`: str or array-like, default `'random'`  
  KMeans initialization strategy.

- `n_init`: int or `'auto'`, default `1`  
  Number of initializations per KMeans run.  
  For ensemble diversity, `n_init=1` is recommended.

- `max_iter`: int, default `300`  
  Maximum number of iterations for each KMeans run.

- `tol`: float, default `1e-4`  
  Convergence tolerance.

- `algorithm`: str, default `'lloyd'`  
  KMeans algorithm.

- `return_models`: bool, default `False`  
  If `True`, also returns the fitted KMeans models.

**Returns**

- `labels_list`: list of arrays  
  One label vector per KMeans run. Each array has shape `(n_samples,)`.

- `centers_list`: list of arrays  
  Cluster centers for each KMeans run.

- `ensemble_summary`: pandas DataFrame  
  Per-run metadata including run index, seed, inertia, and number of iterations.

- `models`: list of fitted KMeans models  
  Returned only when `return_models=True`.

---

#### `cake_with_consensus`

Computes CAKE confidence scores together with consensus clustering labels.

```python
cake_with_consensus(
    X,
    labels_list,
    cake_method='product',
    consensus_method='medoid',
    approximation=False,
    centers_list=None,
    geom_norm='clip'
)
```

**Inputs**

- `X`: array-like of shape `(n_samples, n_features)`  
  Input data matrix.

- `labels_list`: list of array-like, each of shape `(n_samples,)`  
  List of cluster label vectors from multiple clustering runs.

- `cake_method`: str, default `'product'`  
  Method used to combine assignment stability and geometric stability.  
  Options: `'product'` or `'harmonic_mean'`.

- `consensus_method`: str, default `'medoid'`  
  Consensus strategy passed to `consensus_labels`.  
  Options: `'medoid'` or `'best_ref'`.

- `approximation`: bool, default `False`  
  Whether to use centroid-based approximate silhouette scores.

- `centers_list`: list of array-like or None, default `None`  
  Optional list of cluster centers, one per clustering run.  
  Used when `approximation=True`.

- `geom_norm`: str, default `'clip'`  
  Strategy for normalizing the geometric stability component.  
  Options: `'clip'` or `'affine'`.

**Returns**

- `cake_scores`: array of shape `(n_samples,)`  
  Final CAKE confidence score for each sample.

- `consensus`: array of shape `(n_samples,)`  
  Consensus cluster label for each sample.

- `consensus_strength`: array of shape `(n_samples,)`  
  Fraction of aligned runs voting for the consensus label.

- `assignment_stability`: array of shape `(n_samples,)`  
  Pairwise assignment stability across the ensemble.

- `geometric_stability`: array of shape `(n_samples,)`  
  Silhouette-based geometric reliability score.

- `summary`: pandas DataFrame  
  Per-sample table containing consensus labels, consensus strength, CAKE components, and final CAKE score.

- `reference_index`: int  
  Index of the reference run selected by the consensus method.

## Quick Start
This example builds a KMeans ensemble and then computes CAKE confidence scores using `cake`.

```python
from sklearn.datasets import make_blobs
from cake_ensemble import kmeans_ensemble, cake

# Create example data
X, _ = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.0,
    random_state=42,
)

# Build a KMeans ensemble
labels_list, centers_list, ensemble_summary = kmeans_ensemble(
    X,
    n_clusters=3,
    n_runs=50,
    random_state=1,
)

# Compute CAKE scores
cake_scores, assignment_stability, geometric_stability, summary = cake(
    X,
    labels_list,
    method="product",
    approximation=True,
    centers_list=centers_list,
    geom_norm="clip",
)

print(summary.head())
```

The returned `summary` is a pandas DataFrame with one row per data point: 

|   | Mean Silhouette | STD Silhouette | Geometric Stability | Assignment Stability |     CAKE |
| - | --------------: | -------------: | ------------------: | -------------------: | -------: |
| 0 |        0.871807 |       0.176727 |            0.695080 |             0.865306 | 0.601457 |
| 1 |        0.886966 |       0.169302 |            0.717664 |             0.800000 | 0.574131 |
| 2 |        0.806492 |       0.023170 |            0.783321 |             0.832653 | 0.652235 |
| 3 |        0.842851 |       0.069890 |            0.772961 |             0.817143 | 0.631619 |
| 4 |        0.771558 |       0.196408 |            0.575151 |             0.800000 | 0.460120 |


The `CAKE` column contains the final confidence score for each clustering assignment. Higher values indicate assignments that are both stable across the ensemble and geometrically well supported.


Complete code examples demonstrating core functionality can be found in the [demos](https://github.com/semoglou/cake/tree/main/demos) section of the repository.

## License

This project is licensed under the [MIT License](https://github.com/semoglou/cake/blob/main/LICENSE).
