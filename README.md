## CAKE: Confidence in Assignments via K-partition Ensembles
Clustering is widely used for unsupervised structure discovery, yet it offers limited insight into how reliable each individual assignment is. Diagnostics, such as convergence behavior or objective values, may reflect global quality, but they do not indicate whether particular instances are assigned confidently, especially for initialization-sensitive algorithms like *k*-means. This assignment-level instability can undermine both accuracy and robustness. Ensemble approaches improve global consistency by aggregating multiple runs, but they typically lack tools for quantifying pointwise confidence in a way that combines cross-run agreement with geometric support from the learned cluster structure. We introduce **CAKE** (Confidence in Assignments via K-partition Ensembles), a framework that evaluates each point using two complementary statistics computed over a clustering ensemble: assignment stability and consistency of local geometric fit. These are combined into a single, interpretable score in [0,1]. Our theoretical analysis shows that CAKE remains effective under noise and separates stable from unstable points. Experiments on synthetic and real-world datasets indicate that CAKE effectively highlights ambiguous points and stable core members, providing a confidence ranking that can guide filtering or prioritization to improve clustering quality.

# 

For a **preprint**, see: https://arxiv.org/abs/2602.18435 and to cite this work, please use the following:
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
