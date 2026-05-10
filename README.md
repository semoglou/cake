## CAKE: Confidence in Assignments via K-partition Ensembles
Clustering assigns each data point to a group, but it does not tell us how reliable that assignment is.  
While global validation metrics assess overall quality, they provide little insight into the trustworthiness of *individual* assignments.

Ensemble-based clustering improves robustness by aggregating multiple partitions. However, most ensemble-style uncertainty measures focus primarily on **cross-run agreement** (e.g., aligned vote counts or dispersion of labels across runs). Agreement alone is not sufficient: a point may be consistently assigned due to systematic bias or rigid decision boundaries, even if it is weakly integrated into the cluster structure.

Conversely, purely geometric signals such as the Silhouette score evaluate local separation within a single run but ignore cross-run instability. A point may appear geometrically well placed in one partition, yet switch clusters across runs because it lies near a boundary or because multiple partitions explain it nearly equally well.

![Assignment Stability vs Geometric Consistency](https://github.com/semoglou/cake/blob/main/figs/failures.png)

*Complementary failure modes: stability without geometry, geometry without stability.*


These complementary failure modes motivate a confidence signal that jointly accounts for both **assignment stability** and **consistency of geometric support** at the point level.

**CAKE** (Confidence in Assignments via K-partition Ensembles) provides a principled, per-instance confidence score by fusing two complementary statistics computed over a clustering ensemble:

- 🔁 **Assignment Stability**: pairwise agreement across runs after optimal label alignment, using the Hungarian algorithm.   
- 📐 **Geometric Consistency**: aggregated Silhouette statistics across runs.  

These components are combined into a single, interpretable score in **[0, 1]**, enabling:

- Identification of stable core members.  
- Detection of ambiguous boundary points.  
- Filtering of unreliable assignments.  
- Ranking of instances by clustering confidence.  

![Pipeline](https://github.com/semoglou/cake/blob/main/figs/pipel.png)

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

## Function Overview

| Function            | Purpose                                                                                | Key arguments                                                                         | Returns                                                                                                         |
| ------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| cake                | Computes per-point CAKE confidence scores from a clustering ensemble.                  | X, labels_list, method, approximation, centers_list, geom_norm                        | cake_scores, assignment_stability, geometric_stability, summary                                                 |
| cake_with_consensus | Computes CAKE scores together with consensus clustering labels.                        | X, labels_list, cake_method, consensus_method, approximation, centers_list, geom_norm | cake_scores, consensus, consensus_strength, assignment_stability, geometric_stability, summary, reference_index |
| kmeans_ensemble     | Builds a KMeans ensemble by running KMeans multiple times with different random seeds. | X, n_clusters, n_runs, random_state, init, n_init                                     | labels_list, centers_list, ensemble_summary                                                                     |
| consensus_labels    | Computes consensus labels using Hungarian alignment and majority vote.                 | labels_list, method                                                                   | consensus, consensus_strength, reference_index                                                                  |
| pairwise_stability  | Computes per-point assignment stability across all pairs of clustering runs.           | labels_runs                                                                           | assignment_stability                                                                                            |
| align_labels        | Aligns one clustering label vector to another using the Hungarian algorithm.           | target, source                                                                        | aligned_labels                                                                                                  |
| sil_samples         | Computes exact or approximate silhouette scores for each sample.                       | X, labels, approximation, centers                                                     | silhouette_scores                                                                                               |
| sil_samples_stats   | Aggregates sample-level silhouette scores across multiple clustering runs.             | X, labels_list, approximation, centers_list                                           | mean_silhouette, std_silhouette                                                                                 |
