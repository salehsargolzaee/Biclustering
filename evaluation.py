import numpy as np


def estimate_expected_mi(C, G, n_iters=100):
    """
    Estimate the expected mutual information (E[I(C,G')]) by random permutations.
    We shuffle the ground truth labels (G) while keeping the predicted clusters (C) fixed.
    This preserves the structure of C and tests how much MI we get by random chance.
    """
    n_samples = C.shape[0]

    # Flatten G to label assignments
    true_assignments = np.argmax(G, axis=1)

    # Precompute distributions for C
    p_c = C.sum(axis=0) / n_samples

    mi_values = []
    for _ in range(n_iters):
        shuffled_labels = np.random.permutation(true_assignments)
        G_shuffled = np.zeros_like(G)
        for i, lbl in enumerate(shuffled_labels):
            G_shuffled[i, lbl] = 1

        # Compute MI for the shuffled labeling
        p_g = G_shuffled.sum(axis=0) / n_samples
        p_cg = (C.T @ G_shuffled) / n_samples

        nonzero = p_cg > 0
        I_CG = np.sum(
            p_cg[nonzero]
            * np.log(p_cg[nonzero] / (p_c[np.newaxis, :].T * p_g)[nonzero])
        )
        mi_values.append(I_CG)

    return np.mean(mi_values)


def adjusted_onmi_score(labels_true, sample_vec, n_true_clusters, n_iters=100):
    """
    Compute the Adjusted Overlapping Normalized Mutual Information (AONMI)
    between the ground truth and an overlapping predicted clustering.
    """
    n_samples = len(labels_true)
    n_pred_clusters = len(sample_vec)

    # Construct G and C matrices
    G = np.zeros((n_samples, n_true_clusters))
    for i, l in enumerate(labels_true):
        G[i, l] = 1

    C = np.zeros((n_samples, n_pred_clusters))
    for j, cluster in enumerate(sample_vec):
        for s in cluster:
            C[s, j] = 1

    # Remove samples not assigned in either
    mask = (C.sum(axis=1) > 0) | (G.sum(axis=1) > 0)
    C = C[mask]
    G = G[mask]
    n_samples = C.shape[0]

    if n_samples == 0:
        # No valid assignments
        return 0.0

    p_g = G.sum(axis=0) / n_samples
    p_c = C.sum(axis=0) / n_samples

    # Compute entropies
    H_G = -np.sum(p_g[p_g > 0] * np.log(p_g[p_g > 0]))
    H_C = -np.sum(p_c[p_c > 0] * np.log(p_c[p_c > 0]))

    p_cg = (C.T @ G) / n_samples
    nonzero = p_cg > 0
    I_CG = np.sum(
        p_cg[nonzero] * np.log(p_cg[nonzero] / (p_c[np.newaxis, :].T * p_g)[nonzero])
    )

    # If no entropy or no information, return 0
    if H_C == 0 or H_G == 0:
        return 0.0

    E_ICG = estimate_expected_mi(C, G, n_iters=n_iters)

    numerator = I_CG - E_ICG
    denominator = (np.sqrt(H_C * H_G)) - E_ICG
    if denominator == 0:
        return 0.0
    return numerator / denominator


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
    isa_onmi = adjusted_onmi_score(labels, sample_vec_isa, n_true_clusters)

    if just_isa:
        return isa_onmi

    results = "Method,ONMI\n"
    results += f"ISA,{isa_onmi:.3f}\n"

    print("ISA Results:")
    print(f"ONMI: {isa_onmi:.3f}\n")

    # K-Means evaluation

    kmeans_onmi = adjusted_onmi_score(labels, sample_vec_kmeans, n_true_clusters)

    results += f"K-Means,{kmeans_onmi:.3f}\n"
    print("K-Means Results:")
    print(f"ONMI: {kmeans_onmi:.3f}\n")

    # Spectral evaluation
    spectral_onmi = adjusted_onmi_score(labels, sample_vec_spectral, n_true_clusters)

    results += f"Spectral,{spectral_onmi:.3f}"
    print("Spectral Results:")
    print(f"ONMI: {spectral_onmi:.3f}\n")

    with open("./Results/metrics.csv", "w") as f:
        f.write(results)
