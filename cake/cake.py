import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_samples, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder

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
    - geo_norm: str, default='clip'
        - 'affine' (default): maps geom_raw in [-1,1] to [0,1] via (x+1)/2 (preserves negatives).
        - 'clip': max(mean_sil - std_sil, 0) clipped at 1 (original behavior; discards negatives).

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