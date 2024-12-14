import numpy as np
from sklearn.metrics import homogeneity_score


def map_labels(original_labels):
    """
    Map original labels to merged classes:
    {0,4,5} -> 0 (religion)
    {1,3}   -> 1 (science)
    {2}     -> 2 (hockey)
    """
    mapped = []
    for y in original_labels:
        if y in {0, 4, 5}:
            mapped.append(0)  # religion
        elif y in {1, 3}:
            mapped.append(1)  # science
        else:
            mapped.append(2)  # hockey
    return np.array(mapped)


def sample_vec_to_labels(data, sample_vec, feature_vec, n_samples, labels_true):
    """
    Assign samples to clusters based on the bicluster with the highest feature sum.

    :param data: 2D numpy array of shape (n_samples, n_features)
    :param sample_vec: A list of sets, where each set contains the indices of samples in that bicluster.
    :param feature_vec: A list of sets, where each set contains the indices of features in that bicluster.
    :param n_samples: Total number of samples.
    :return: labels_pred: 1D numpy array with the cluster assignments for each sample.
    """
    labels_pred = np.full(n_samples, -1, dtype=int)

    # For each sample, determine which bicluster it belongs to most strongly.
    for i in range(n_samples):
        best_bicluster = -1
        best_sum = -np.inf

        # Check every bicluster to see if sample i is included.
        for bc_id, samples_set in enumerate(sample_vec):
            if i in samples_set:
                # Compute the sum of data[i, features_of_this_bicluster]
                # Convert feature_set to a list for indexing
                features_list = list(feature_vec[bc_id])
                current_sum = np.sum(data[i, features_list])

                # If this sum is greater than any found so far, update best choice.
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_bicluster = bc_id

        # Assign the sample to the best bicluster found (if any)
        if best_bicluster != -1:
            labels_pred[i] = best_bicluster

    # If any samples are not assigned to any bicluster, assign them to a new cluster or leave them as is.
    # Here, we create a dummy cluster for unassigned samples:
    unassigned = labels_pred == -1

    temp_labels_pred, temp_labels_true = [], []

    for i in range(n_samples):
        if i in unassigned:
            continue
        temp_labels_pred.append(labels_pred[i])
        temp_labels_true.append(labels_true[i])

    return np.array(temp_labels_pred), np.array(temp_labels_true)


def onmi_score(labels_true, sample_vec, n_true_clusters):
    """
    Compute the Overlapping Normalized Mutual Information (ONMI) between the ground truth
    and an overlapping predicted clustering.

    :param labels_true: Merged ground truth labels, one per sample.
    :param sample_vec: Overlapping clusters as a list of sets of sample indices.
    :param n_true_clusters: Number of merged ground truth classes.
    :return: ONMI score
    """
    n_samples = len(labels_true)
    n_pred_clusters = len(sample_vec)

    # Construct G matrix (N x n_true_clusters)
    # Each sample belongs to exactly one class
    G = np.zeros((n_samples, n_true_clusters))
    for i, l in enumerate(labels_true):
        G[i, l] = 1

    # Construct C matrix (N x n_pred_clusters)
    # A sample can be in multiple predicted clusters
    C = np.zeros((n_samples, n_pred_clusters))
    for j, cluster in enumerate(sample_vec):
        for s in cluster:
            C[s, j] = 1

    temp_C, temp_G = [], []

    for i in range(n_samples):
        if np.sum(C[i]) == 0 and np.sum(G[i]) == 0:
            continue
        temp_C.append(C[i])
        temp_G.append(G[i])

    C, G = np.array(temp_C), np.array(temp_G)
    n_samples = C.shape[0]

    # Compute probabilities
    p_g = G.sum(axis=0) / n_samples  # shape: (n_true_clusters,)
    p_c = C.sum(axis=0) / n_samples  # shape: (n_pred_clusters,)

    # Compute joint probabilities p(c_j, g_l)
    # p(c_j, g_l) = (1/N) * sum over i of C[i,j]*G[i,l]
    p_cg = (C.T @ G) / n_samples  # shape: (n_pred_clusters, n_true_clusters)

    # Compute H(G'), H(C)
    H_G = -np.sum(p_g[p_g > 0] * np.log(p_g[p_g > 0]))
    H_C = -np.sum(p_c[p_c > 0] * np.log(p_c[p_c > 0]))

    # Compute I(C,G')
    # Only consider pairs where p_cg > 0
    nonzero = p_cg > 0
    I_CG = np.sum(
        p_cg[nonzero] * np.log(p_cg[nonzero] / (p_c[np.newaxis, :].T * p_g)[nonzero])
    )

    # ONMI
    # Note: If either H_C or H_G is zero, then ONMI is not defined.
    if H_C == 0 or H_G == 0:
        return 0.0
    ONMI = I_CG / np.sqrt(H_C * H_G)
    return ONMI


def evaluate_algorithms(
    data,
    labels,
    sample_vec_isa,
    feature_vec_isa,
    sample_vec_kmeans,
    feature_vec_kmeans,
    sample_vec_spectral,
    feature_vec_spectral,
):
    """
    Evaluate ISA, K-Means, and Spectral results using ONMI and Homogeneity.
    """
    # Map labels to merged classes (note that you should hard-code mapped classes)
    mapped_labels = map_labels(labels)
    n_true_clusters = len(np.unique(mapped_labels))

    # ISA evaluation
    isa_single_labels, mapped_labels_isa = sample_vec_to_labels(
        data, sample_vec_isa, feature_vec_isa, len(labels), mapped_labels
    )
    isa_homogeneity = homogeneity_score(mapped_labels_isa, isa_single_labels)
    isa_onmi = onmi_score(mapped_labels, sample_vec_isa, n_true_clusters)

    print("ISA Results:")
    print(f"Homogeneity: {isa_homogeneity:.3f}")
    print(f"ONMI: {isa_onmi:.3f}\n")

    # K-Means evaluation
    kmeans_single_labels, mapped_labels_kmeans = sample_vec_to_labels(
        data, sample_vec_kmeans, feature_vec_kmeans, len(labels), mapped_labels
    )
    kmeans_homogeneity = homogeneity_score(mapped_labels_kmeans, kmeans_single_labels)
    kmeans_onmi = onmi_score(mapped_labels, sample_vec_kmeans, n_true_clusters)

    print("K-Means Results:")
    print(f"Homogeneity: {kmeans_homogeneity:.3f}")
    print(f"ONMI: {kmeans_onmi:.3f}\n")

    # Spectral evaluation
    spectral_single_labels, mapped_labels_spectral = sample_vec_to_labels(
        data, sample_vec_spectral, feature_vec_spectral, len(labels), mapped_labels
    )
    spectral_homogeneity = homogeneity_score(
        mapped_labels_spectral, spectral_single_labels
    )
    spectral_onmi = onmi_score(mapped_labels, sample_vec_spectral, n_true_clusters)

    print("Spectral Results:")
    print(f"Homogeneity: {spectral_homogeneity:.3f}")
    print(f"ONMI: {spectral_onmi:.3f}\n")
