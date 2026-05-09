import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

def sil_samples(X, labels, approximation=False, centers=None):
    """
    Compute silhouette scores for each point in the dataset,
    with approximate fast centroid-based computation option.
    """
    # Ensure arrays
    X = np.asarray(X)
    labels = np.asarray(labels)
    if X.ndim != 2:
       raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if labels.ndim != 1 or labels.shape[0] != X.shape[0]:
       raise ValueError("labels must be a 1D array of length n_samples.")

    unique_labels, inv = np.unique(labels, return_inverse=True)
    k = unique_labels.size
    if k < 2:
       raise ValueError("Silhouette computation requires at least 2 clusters.")

    # Exact silhouette scores
    if approximation == False:
       silhouette_scores = silhouette_samples(X, labels=labels)
       return silhouette_scores

    # Centroid-based approximate silhouette scores
    n_samples, n_features = X.shape

    if centers is None:
       centers = np.array([X[inv == i].mean(axis=0) for i in range(k)], dtype=float)
    else:
       centers = np.asarray(centers, dtype=float)
       if centers.ndim != 2 or centers.shape[1] != n_features:
          raise ValueError(f"centers must have shape (k, d) with d={n_features}.")
       if centers.shape[0] != k:
          raise ValueError(f"centers.shape[0] must equal number of clusters k={k}.")
       if not np.array_equal(unique_labels, np.arange(k)):
          raise ValueError("When passing ndarray centers, labels must be dense 0..k-1.")

    # Squared distances to all centroids
    D_sq = euclidean_distances(X, centers, squared=True)

    # a(i): distance to own centroid
    a = np.sqrt(np.maximum(D_sq[np.arange(n_samples), inv], 0.0))

    # b(i): distance to nearest other centroid
    D_sq[np.arange(n_samples), inv] = np.inf
    b = np.sqrt(np.min(D_sq, axis=1))

    # Silhouette per point
    denom = np.maximum(np.maximum(a, b), 1e-12)
    s_point = (b - a) / denom

    # Singleton clusters -> silhouette = 0
    counts = np.bincount(inv, minlength=k).astype(int)
    s_point[counts[inv] < 2] = 0.0

    silhouette_scores = np.clip(s_point, -1.0, 1.0)

    return silhouette_scores

def sil_samples_stats(X, labels_list, approximation=False, centers_list=None):
    """
    Compute and aggregate silhouette scores across multiple clustering runs,
    returning per-sample mean and standard deviation of silhouette scores.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    n_runs = len(labels_list)
    if n_runs < 2:
       raise ValueError("Clustering Ensemble must contain at least 2 partitions.")

    if centers_list is not None and len(centers_list) != n_runs:
       raise ValueError("centers_list must match the number of labelings in labels_list")

    # Initialize matrix to hold silhouette scores
    sil_scores = np.zeros((n_runs, n_samples), dtype=float)

    for i, labels in enumerate(labels_list):
        labels_arr = np.asarray(labels)
        centers = None
        if approximation and centers_list is not None:
           centers = centers_list[i]

        # Compute silhouette scores for this run
        sil_scores[i] = sil_samples(X, labels_arr, approximation=approximation, centers=centers)

    # Compute mean and standard deviation of sample-silhouette scores over the ensemble
    mean_sil_samples = sil_scores.mean(axis=0)
    std_sil_samples = sil_scores.std(axis=0)

    return mean_sil_samples, std_sil_samples

def align_labels(target, source):
    """
    Aligns the labels in "source" to match the labels in "target"
    using the Hungarian Algorithm based on a contingency matrix.

    Parameters:
    - target: array-like of shape (n_samples,)
              The reference label vector to align to.
    - source: array-like of shape (n_samples,)
              The label vector to be permuted for alignment.

    Returns:
    - aligned: np.ndarray of shape (n_samples,)
               The source labels, remapped to best match the target labels.
    """
    target = np.asarray(target)
    source = np.asarray(source)
    unique_target = np.unique(target)
    unique_source = np.unique(source)
    if len(unique_target) != len(unique_source):
       raise ValueError(
           f"Cannot align: Target has {len(unique_target)} clusters {unique_target.tolist()}, "
           f"but source has {len(unique_source)} clusters {unique_source.tolist()}"
       )
    # Encode labels to indices based on their unique values
    le_target = LabelEncoder().fit(unique_target)
    le_source = LabelEncoder().fit(unique_source)

    target_encoded = le_target.transform(target)
    source_encoded = le_source.transform(source)

    # Confusion matrix with indices aligned to encoded labels
    c_matrix = confusion_matrix(target_encoded, source_encoded)

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-c_matrix)

    # Create mapping from source labels to target labels
    mapping = {
        le_source.classes_[src_col]: le_target.classes_[tgt_row]
        for tgt_row, src_col in zip(row_ind, col_ind)
    }
    # Remap source labels using the mapping
    aligned = np.vectorize(mapping.get)(source)

    return aligned

def pairwise_stability(labels_runs):
    """
    Computes per-point clustering stability across multiple runs by aligning labels
    pairwise using the Hungarian algorithm to account for label permutations.

    Parameters:
    - labels_runs: list of array-like
        List of label arrays from multiple clustering runs. Each array must have the
        same number of samples and clusters (unique labels).

    Returns:
    - stability: np.ndarray of shape (n_samples,)
        Per-point stability score between 0 and 1, indicating the fraction of run-pairs
        where the point's label matches after optimal alignment.

    Notes:
    - Requires all runs to have the same number of clusters (unique labels).
    - Label alignment is done pairwise between runs using the Hungarian algorithm.
    - High stability (~1) indicates stable cluster assignments across runs.
    """
    labels_runs = [np.asarray(labels) for labels in labels_runs]
    n_runs, n_samples = len(labels_runs), len(labels_runs[0])
    if n_runs < 2:
       raise ValueError("pairwise_stability needs at least 2 runs.")

    # Counters: How many run-pairs agree per point
    agreement_counts = np.zeros(n_samples, dtype=int)
    total_pairs = 0

    # For every unique pair of runs
    for r1 in range(n_runs):
        labels_r1 = labels_runs[r1]
        for r2 in range(r1+1, n_runs):
            labels_r2 = labels_runs[r2]

            # Align labels_r2 to labels_r1 using Hungarian
            aligned_r2 = align_labels(labels_r1, labels_r2)
            matches = (labels_r1 == aligned_r2)

            # Add matches to total per-point agreement count
            agreement_counts += matches
            total_pairs += 1

    return agreement_counts/total_pairs

def cake(X, labels_list, method='product', approximation=False, centers_list=None, geom_norm='clip'):
    """
    Compute a confidence score per point for clustering ensembles, defined as:
    stability_i * geometric_stability_i or
    2 * stability_i * geometric_stability_i / [stability_i + geometric_stability_i]
    where stability is the pairwise label agreement across runs,
    and mean_sil_i, std_sil_i are the statistics from sil_samples_stats.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - labels_list: list of array-like, each of shape (n_samples,)
        A list of cluster labelings for the same dataset X.
    - approximation: bool, default=False
        Whether to use the approximate silhouette computation.
    - centers_list: list of pd.Series or array-like, optional
        If approximation=True, an optional list of centroids for each clustering.
    - method: str, default='product'
        'product' or 'harmonic_mean' for the formulation of the CAKE scores.
    - geom_norm: str, default='clip'
        - 'clip': max(mean_sil - std_sil, 0) clipped at 1 (original behavior; discards negatives).
        - 'affine': maps geom_raw in [-1,1] to [0,1] via (x+1)/2 (preserves negatives).

    Returns:
    - cake_scores: np.ndarray, shape (n_samples,)
        The CAKE scores for each point.
    - stability: np.ndarray, shape (n_samples,)
        The stability scores for each point.
    - geom_stability: np.ndarray, shape (n_samples,)
        The silhouette-based reliability scores for each point (mean_sil_i - std_sil_i).
    - summary: pd.DataFrame
        Summary of above metrics per point in X.
    """
    # Compute silhouette statistics
    mean_sil, std_sil = sil_samples_stats(X, labels_list,
                                          approximation=approximation,
                                          centers_list=centers_list)

    # Assignment stability across the ensemble
    stability = pairwise_stability(labels_list)

    # Silhouette-based stability across the ensemble
    geom_raw = mean_sil - std_sil
    if geom_norm == 'clip':
       geom_stability = np.clip(mean_sil - std_sil, 0.0, 1.0)
    else:
       geom_stability = np.clip((geom_raw + 1) / 2, 0, 1) # preserves information for negative scores

    # Calculate confidence scores
    if method == 'product':
       cake_scores = stability * geom_stability
    elif method == 'harmonic_mean':
       num = 2.0 * stability * geom_stability
       denom = np.maximum(stability + geom_stability, 1e-8)
       cake_scores = num / denom
    else:
       raise ValueError(
           f"Unknown method: {method}. Use 'product' or 'harmonic_mean'."
       )

    # Summary DataFrame
    summary = pd.DataFrame({
        'Mean Silhouette': mean_sil,
        'STD Silhouette': std_sil,
        'Geometric Stability': geom_stability,
        'Assignment Stability': stability,
        'CAKE': cake_scores
    })

    return cake_scores, stability, geom_stability, summary

def consensus_labels(labels_list, method="medoid"):
    """
    Compute consensus clustering labels from multiple clustering runs
    using Hungarian label alignment and majority vote.

    Parameters:
    - labels_list: list of array-like, each of shape (n_samples,)
        A list of cluster labelings for the same dataset.
    - method: str, default='medoid'
        Consensus strategy:
        - 'medoid': choose the most representative run as reference, align all runs
          to it, then compute majority-vote labels.
        - 'best_ref': try each run as reference, align all runs to it, compute
          majority-vote labels, and keep the consensus with the highest average
          consensus strength.

    Returns:
    - consensus: np.ndarray of shape (n_samples,)
        Consensus label for each point.
    - consensus_strength: np.ndarray of shape (n_samples,)
        Fraction of runs voting for the selected consensus label per point.
    - reference_index: int
        Index of the selected reference run.
    """
    labels_runs = [np.asarray(labels) for labels in labels_list]
    n_runs = len(labels_runs)

    if n_runs < 2:
        raise ValueError("consensus_labels needs at least 2 partitions.")

    n_samples = len(labels_runs[0])

    for labels in labels_runs:
        if labels.ndim != 1:
            raise ValueError("All label arrays in labels_list must be 1D.")
        if len(labels) != n_samples:
            raise ValueError("All label arrays in labels_list must have the same length.")

    n_clusters = len(np.unique(labels_runs[0]))
    for labels in labels_runs:
        if len(np.unique(labels)) != n_clusters:
            raise ValueError("All label arrays must have the same number of clusters.")

    if method == "medoid":
        avg_agreements = np.zeros(n_runs, dtype=float)

        for r1 in range(n_runs):
            agreements = []

            for r2 in range(n_runs):
                if r1 == r2:
                    continue

                aligned_r2 = align_labels(labels_runs[r1], labels_runs[r2])
                agreements.append(np.mean(labels_runs[r1] == aligned_r2))

            avg_agreements[r1] = np.mean(agreements)

        reference_candidates = [int(np.argmax(avg_agreements))]

    elif method == "best_ref":
        reference_candidates = list(range(n_runs))

    else:
        raise ValueError(
            f"Unknown consensus method: {method}. Use 'medoid' or 'best_ref'."
        )

    best_score = -np.inf
    best_consensus = None
    best_consensus_strength = None
    best_reference_index = None

    for reference_index in reference_candidates:
        reference_labels = labels_runs[reference_index]

        aligned_runs = []

        for r in range(n_runs):
            if r == reference_index:
                aligned_runs.append(reference_labels)
            else:
                aligned_runs.append(align_labels(reference_labels, labels_runs[r]))

        aligned_runs = np.asarray(aligned_runs)

        candidate_consensus = np.empty(n_samples, dtype=aligned_runs.dtype)
        candidate_consensus_strength = np.empty(n_samples, dtype=float)

        for i in range(n_samples):
            values, counts = np.unique(aligned_runs[:, i], return_counts=True)
            winner = np.argmax(counts)
            candidate_consensus[i] = values[winner]
            candidate_consensus_strength[i] = counts[winner] / n_runs

        score = np.mean(candidate_consensus_strength)

        if score > best_score:
            best_score = score
            best_consensus = candidate_consensus
            best_consensus_strength = candidate_consensus_strength
            best_reference_index = reference_index

    return best_consensus, best_consensus_strength, best_reference_index

def kmeans_ensemble(
    X,
    n_clusters,
    n_runs=50,
    random_state=1,
    init="random",
    n_init=1,
    max_iter=300,
    tol=1e-4,
    algorithm="lloyd",
    return_models=False,
    **kmeans_kwargs
):
    """
    Build a KMeans clustering ensemble by running KMeans multiple times
    with different random seeds.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Dataset to cluster.
    - n_clusters: int
        Number of clusters.
    - n_runs: int, default=50
        Number of KMeans runs / ensemble partitions.
    - random_state: int or None, default=1
        Base random seed. If provided, seeds are generated reproducibly.
    - init: str or array-like, default="random"
        KMeans initialization method.
    - n_init: int or 'auto', default=1
        Number of initializations per KMeans run. For ensemble diversity,
        n_init=1 is recommended.
    - max_iter: int, default=300
        Maximum number of iterations per KMeans run.
    - tol: float, default=1e-4
        Tolerance for convergence.
    - algorithm: str, default='lloyd'
        KMeans algorithm.
    - return_models: bool, default=False
        If True, also return the fitted KMeans models.
    - **kmeans_kwargs:
        Additional keyword arguments passed to sklearn.cluster.KMeans.

    Returns:
    - labels_list: list of np.ndarray
        List of label arrays, one per KMeans run.
    - centers_list: list of np.ndarray
        List of cluster centers, one per KMeans run.
    - ensemble_summary: pd.DataFrame
        Per-run metadata: run index, seed, inertia, number of iterations.
    - models: list of KMeans, optional
        Returned only if return_models=True.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

    n_samples = X.shape[0]

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")

    if n_clusters > n_samples:
        raise ValueError("n_clusters cannot be greater than n_samples.")

    if n_runs < 2:
        raise ValueError("kmeans_ensemble needs at least 2 runs.")

    if random_state is None:
        seeds = [None] * n_runs
    else:
        rng = np.random.default_rng(random_state)
        seeds = rng.integers(
            low=0,
            high=np.iinfo(np.int32).max,
            size=n_runs
        ).tolist()

    labels_list = []
    centers_list = []
    inertias = []
    n_iters = []
    models = []

    for run, seed in enumerate(seeds):
        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            algorithm=algorithm,
            random_state=seed,
            **kmeans_kwargs
        )

        model.fit(X)

        labels_list.append(model.labels_.copy())
        centers_list.append(model.cluster_centers_.copy())
        inertias.append(model.inertia_)
        n_iters.append(model.n_iter_)

        if return_models:
            models.append(model)

    ensemble_summary = pd.DataFrame({
        "Run": np.arange(n_runs),
        "Seed": seeds,
        "Inertia": inertias,
        "Iterations": n_iters,
    })

    if return_models:
        return labels_list, centers_list, ensemble_summary, models

    return labels_list, centers_list, ensemble_summary
  
def cake_with_consensus(
    X,
    labels_list,
    cake_method="product",
    consensus_method="medoid",
    approximation=False,
    centers_list=None,
    geom_norm="clip",
):
    """
    Compute CAKE confidence scores together with consensus clustering labels.

    This function combines:
    - CAKE scores from the clustering ensemble.
    - Consensus labels from the same ensemble using Hungarian alignment
      and majority vote.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Dataset used for silhouette-based geometric stability.
    - labels_list: list of array-like, each of shape (n_samples,)
        A list of cluster labelings for the same dataset X.
    - cake_method: str, default='product'
        CAKE score formulation:
        - 'product'
        - 'harmonic_mean'
    - consensus_method: str, default='medoid'
        Consensus strategy used by consensus_labels:
        - 'medoid'
        - 'best_ref'
    - approximation: bool, default=False
        Whether to use approximate centroid-based silhouette scores.
    - centers_list: list of array-like, optional
        Cluster centers for each run, used when approximation=True.
    - geom_norm: str, default='clip'
        Geometric normalization:
        - 'clip': max(mean_sil - std_sil, 0), clipped to [0, 1].
        - 'affine': maps geom_raw from [-1, 1] to [0, 1].

    Returns:
    - cake_scores: np.ndarray of shape (n_samples,)
        CAKE confidence scores.
    - consensus: np.ndarray of shape (n_samples,)
        Consensus label for each point.
    - consensus_strength: np.ndarray of shape (n_samples,)
        Fraction of aligned runs voting for the consensus label.
    - stability: np.ndarray of shape (n_samples,)
        Pairwise assignment stability scores.
    - geom_stability: np.ndarray of shape (n_samples,)
        Silhouette-based geometric reliability scores.
    - summary: pd.DataFrame
        Per-point summary including consensus labels, consensus strength,
        CAKE components, and CAKE score.
    - reference_index: int
        Reference run selected by the consensus method.
    """
    cake_scores, stability, geom_stability, summary = cake(
        X,
        labels_list,
        method=cake_method,
        approximation=approximation,
        centers_list=centers_list,
        geom_norm=geom_norm,
    )

    consensus, consensus_strength, reference_index = consensus_labels(
        labels_list,
        method=consensus_method,
    )

    summary = summary.copy()
    summary.insert(0, "Consensus Label", consensus)
    summary.insert(1, "Consensus Strength", consensus_strength)

    return (
        cake_scores,
        consensus,
        consensus_strength,
        stability,
        geom_stability,
        summary,
        reference_index,
    )