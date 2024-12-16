import numpy as np


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
    labels,
    sample_vec_isa,
    sample_vec_kmeans=None,
    sample_vec_spectral=None,
    just_isa=False,
):
    """
    Evaluate ISA, K-Means, and Spectral results using ONMI
    """
    # Map labels to merged classes (note that you should hard-code mapped classes)
    n_true_clusters = len(np.unique(labels))

    # ISA evaluation
    isa_onmi = onmi_score(labels, sample_vec_isa, n_true_clusters)

    if just_isa:
        return isa_onmi

    results = "Method,ONMI\n"
    results += f"ISA,{isa_onmi:.3f}\n"

    print("ISA Results:")
    print(f"ONMI: {isa_onmi:.3f}\n")

    # K-Means evaluation

    kmeans_onmi = onmi_score(labels, sample_vec_kmeans, n_true_clusters)

    results += f"K-Means,{kmeans_onmi:.3f}\n"
    print("K-Means Results:")
    print(f"ONMI: {kmeans_onmi:.3f}\n")

    # Spectral evaluation
    spectral_onmi = onmi_score(labels, sample_vec_spectral, n_true_clusters)

    results += f"Spectral,{spectral_onmi:.3f}"
    print("Spectral Results:")
    print(f"ONMI: {spectral_onmi:.3f}\n")

    with open("./Results/metrics.csv", "w") as f:
        f.write(results)
