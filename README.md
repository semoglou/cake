## CAKE: Confidence in Assignments via K-partition Ensembles
Clustering assigns each data point to a group, but it does not tell us how reliable that assignment is.  
While global validation metrics assess overall quality, they provide little insight into the trustworthiness of *individual* assignments.

Ensemble-based clustering improves robustness by aggregating multiple partitions. However, most ensemble-style uncertainty measures focus primarily on **cross-run agreement** (e.g., aligned vote counts or dispersion of labels across runs). Agreement alone is not sufficient: a point may be consistently assigned due to systematic bias or rigid decision boundaries, even if it is weakly integrated into the cluster structure.

Conversely, purely geometric signals such as the Silhouette score evaluate local separation within a single run but ignore cross-run instability. A point may appear geometrically well placed in one partition, yet switch clusters across runs because it lies near a boundary or because multiple partitions explain it nearly equally well.

![Assignment Stability vs Geometric Consistency](https://github.com/semoglou/cake/blob/main/figs/failures.png)

*Complementary failure modes: stability without geometry, geometry without stability.*


These complementary failure modes motivate a confidence signal that jointly accounts for both **assignment stability** and **consistency of geometric support** at the point level.

#

## The CAKE Framework

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
arXiv preprint arXiv:2602.18435.  
https://arxiv.org/abs/2602.18435

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

