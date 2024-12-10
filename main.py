import os
import random
from itertools import combinations
from time import time
from typing import List, Set
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import umap
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud


def row_normalization(data: np.array) -> np.array:
    """
    Normalize rows.
    :param data: The data.
    :return: The normalized rows.
    """
    centered_data = data - data.mean(axis=0)[None, :]
    return centered_data / np.linalg.norm(centered_data, axis=1)[:, None]


def column_normalization(data: np.array) -> np.array:
    """
    Normalize columns.
    :param data: The data.
    :return: The normalized columns.
    """
    centered_data = data - data.mean(axis=1)[:, None]
    return centered_data / np.linalg.norm(centered_data, axis=0)[None, :]


def vector_to_set(vector: np.ndarray) -> Set[int]:
    """
    Convert a binary or weighted vector to a set of indices with non-zero values.
    :param vector: 1D NumPy array.
    :return: Set of indices where the vector has non-zero values.
    """
    return set(np.nonzero(vector)[0])


def jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """
    Compute the Jaccard similarity between two sets.
    :param set1: First set.
    :param set2: Second set.
    :return: Jaccard similarity as a float.
    """
    union = set1.union(set2)
    return (len(set1.intersection(set2)) / len(union)) if union else 0.0


def fuse_fixed_points_vectors(
    feature_vectors: np.ndarray,
    sample_vectors: np.ndarray,
    similarity_threshold: float = 0.7,
) -> tuple[list[set], list[set[set[int] | list[set[int]]]]]:
    """
    Fuse similar fixed points based on their feature and sample vectors.
    :param feature_vectors: 2D NumPy array where each row represents a feature vector for a fixed point. Columns
    correspond to features.
    :param sample_vectors: 2D NumPy array where each row represents a sample vector for a fixed point. Columns
    correspond to samples.
    :param similarity_threshold: Threshold above which fixed points are considered similar.
    :return: List of sets, each representing fused features in a module and list of sets, each representing fused
    samples in a module
    """
    if feature_vectors.shape[0] != sample_vectors.shape[0]:
        raise ValueError(
            "feature_vectors and sample_vectors must have the same number of fixed points (rows)."
        )

    if feature_vectors.shape[0] == 0:
        return list(set()), list(set())

    num_fixed_points = feature_vectors.shape[0]
    g = nx.Graph()
    # Nodes are indices of fixed points.
    g.add_nodes_from(range(num_fixed_points))
    # Precompute sets for all fixed points.
    feature_sets = [vector_to_set(feature_vectors[i]) for i in range(num_fixed_points)]
    sample_sets = [vector_to_set(sample_vectors[i]) for i in range(num_fixed_points)]
    # Iterate over all unique pairs to compute similarity.
    for i, j in combinations(range(num_fixed_points), 2):
        feature_sim = jaccard_similarity(feature_sets[i], feature_sets[j])
        sample_sim = jaccard_similarity(sample_sets[i], sample_sets[j])
        # Define overall similarity as the average of feature and sample similarity.
        overall_sim = (feature_sim + sample_sim) / 2
        if overall_sim >= similarity_threshold:
            g.add_edge(i, j)
    # Find connected components (clusters of similar fixed points).
    clusters = list(nx.connected_components(g))
    fused_features_list = []
    fused_samples_list = []
    for cluster in clusters:
        fused_features = set()
        fused_samples = set()
        for idx in cluster:
            fused_features.update(feature_sets[idx])
            fused_samples.update(sample_sets[idx])
        fused_features_list.append(fused_features)
        fused_samples_list.append(fused_samples)
    return fused_features_list, fused_samples_list


def display_fused_modules(
    fused_features: List[Set[int]],
    fused_samples: List[Set[int]],
    feature_labels: List[str],
    sample_labels: List[str],
    name: str,
) -> None:
    """
    Display the fused modules with feature and sample labels.
    :param fused_features: List of sets containing feature indices.
    :param fused_samples: List of sets containing sample indices.
    :param feature_labels: List mapping feature indices to feature names.
    :param sample_labels: List mapping sample indices to sample names.
    :param name: Model name.
    :return: Nothing.
    """
    results = ""
    for i, (features, samples) in enumerate(zip(fused_features, fused_samples), 1):
        feature_names = [feature_labels[idx] for idx in features]
        sample_names = [sample_labels[idx] for idx in samples]
        results += f"{name} Fused Module {i}:\n"
        results += f"Features: {sorted(feature_names)}\n"
        results += f"Samples_indices: {list(map(lambda x: int(x), samples))}\n"
        results += f"Samples_labels: {list(map(lambda x: int(x), sample_names))}\n\n"
        # print(results)
    with open(os.path.join("Results", name, f"biclusters.txt"), "w") as file:
        file.write(results)


def threshold(x: np.array, t: int) -> np.array:
    """
    Get the threshold.
    :param x: X value.
    :param t: Feature or sample.
    :return: The threshold.
    """
    s = np.std(x)
    new_x = (x - np.mean(x)) / (s if (s != 0 and not np.isinf(s)) else 1)
    return (new_x > t).astype(int)


def f(x: np.array, t_feature: int = None, t_sample: int = None) -> np.array:
    """
    Get the threshold of feature or sample.
    :param x: X value.
    :param t_feature: Feature.
    :param t_sample: Sample/
    :return: The threshold.
    """
    return (
        x * threshold(x=x, t=t_feature)
        if t_feature is not None
        else x * threshold(x=x, t=t_sample)
    )


def isa(
    data: np.array,
    n_initial: int = 1000,
    n_updates=1000,
    thresh_feature: float = 0,
    thresh_sample: float = 0,
    fusion_similarity_threshold: float = 0.8,
):
    """
    Perform ISA.
    :param data: The data.
    :param n_initial: Number of initial values.
    :param n_updates: Number of updates.
    :param thresh_feature: Feature threshold.
    :param thresh_sample: Sample threshold.
    :param fusion_similarity_threshold: Fusion similarity threshold.
    :return: The result of ISA.
    """
    n_samples, n_features = data.shape
    r_data = row_normalization(data)
    c_data = column_normalization(data)
    sample_vector = np.zeros(n_samples)
    all_feature_vectors = []
    all_sample_vectors = []
    for _ in range(n_initial):
        seed = np.random.binomial(n=1, p=0.5, size=n_features)
        feature_vector = seed
        for i in range(n_updates):
            sample_vector = f(r_data.dot(feature_vector), t_sample=thresh_sample)
            feature_vector = f(c_data.T.dot(sample_vector), t_feature=thresh_feature)

        if sum(sample_vector != 0) < 2 or sum(feature_vector != 0) < 2:
            continue

        all_feature_vectors.append(feature_vector)
        all_sample_vectors.append(sample_vector)
    all_feature_vectors = np.array(all_feature_vectors)
    all_sample_vectors = np.array(all_sample_vectors)
    # Perform fusion.
    fused_features, fused_samples = fuse_fixed_points_vectors(
        all_feature_vectors, all_sample_vectors, fusion_similarity_threshold
    )
    return fused_samples, fused_features


def manifold(
    sample_vec: List[Set[int] or np.ndarray],
    feature_vec,
    data,
    title: str,
    name: str,
    manifold_learner: str = "tsne",
    random_state: int = 42,
) -> None:
    """
    Perform manifold on a sample.
    :param sample_vec: Samples.
    :param feature_vec: Features.
    :param data: The data.
    :param title: The title.
    :param manifold_learner: The manifold mode being either "tsne" or "umap".
    :param random_state: The random state.
    :return: Nothing.
    """
    solution = []
    cluster_labels = []
    all_features = set()
    for i in range(len(feature_vec)):
        all_features.update(feature_vec[i])
    all_features = sorted(list(all_features))
    for i in range(len(sample_vec)):
        samples = sorted(list(sample_vec[i]))
        solution.extend(data[samples, :][:, all_features])
        cluster_labels.extend([i + 1] * len(sample_vec[i]))

    if len(solution) <= 1:
        return

    solution = np.array(solution)
    solution = row_normalization(solution)

    if manifold_learner == "tsne":
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=len(solution) - 1,
        )
        embedded = tsne.fit_transform(solution)
    else:
        reducer = umap.UMAP(random_state=random_state)
        embedded = reducer.fit_transform(solution)
    fig = plt.figure(figsize=(10, 6))
    for i, cluster_label in enumerate(set(cluster_labels)):
        cluster_points = embedded[np.array(cluster_labels) == cluster_label]
        plt.scatter(
            cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}"
        )
    plt.title(title)
    plt.legend()
    fig.savefig(os.path.join("Results", name, f"{title}.png"))
    plt.close(fig)


def display(
    sample_vec, feature_vec, name: str, vectorizer, data, labels, n_features
) -> None:
    """
    Create plots.
    :param sample_vec: The sample vectors.
    :param feature_vec: The features.
    :param name: The name for plots saving.
    :param vectorizer: The vectorizer.
    :param data: The data.
    :param labels: The labels.
    :param n_features: The number of features.
    :return: Nothing.
    """
    os.makedirs(os.path.join("Results", name), exist_ok=True)

    display_fused_modules(
        fused_features=feature_vec,
        fused_samples=sample_vec,
        feature_labels=vectorizer.get_feature_names_out(),
        sample_labels=labels,
        name=name,
    )

    words = vectorizer.get_feature_names_out()
    # Get all samples and features.
    upper = min(len(sample_vec), len(feature_vec))
    for ind in range(upper):
        samples, features = sorted(list(sample_vec[ind])), sorted(
            list(feature_vec[ind])
        )
        bicluster = data[samples][:, features]
        # Perform hierarchical clustering for rows and columns.
        row_linkage = linkage(bicluster, method="ward")
        col_linkage = linkage(bicluster.T, method="ward")
        # Get the order of rows and columns based on the clustering.
        row_order = leaves_list(row_linkage)
        col_order = leaves_list(col_linkage)
        # Reorder the data based on clustering.
        reordered_data = bicluster[row_order, :][:, col_order]
        fig = plt.figure(figsize=(25, 8), dpi=200)
        sns.heatmap(reordered_data, cmap="viridis", annot=False)
        title = f"{name if "/" not in name else ""} Bicluster {ind + 1} Heatmap"
        plt.title(title)
        plt.xlabel("Terms")
        plt.ylabel("Documents")
        plt.xticks([])
        plt.yticks([])
        fig.savefig(os.path.join("Results", name, f"{title}.png"))
        plt.close(fig)
        samples, features = sorted(list(sample_vec[ind])), sorted(
            list(feature_vec[ind])
        )
        bicluster_terms = {
            words[word_ind]: np.mean(data[samples, :][:, word_ind])
            for word_ind in features
        }
        wc = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(bicluster_terms)
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        title = f"{name if "/" not in name else ""} Word Cloud for Bicluster {ind + 1}"
        plt.title(title)
        fig.savefig(os.path.join("Results", name, f"{title}.png"))
        plt.close(fig)
    manifold(
        sample_vec,
        feature_vec,
        data,
        f"{name if "/" not in name else ""} t_sne of solution",
        name,
    )
    manifold(
        [np.where(labels == label)[0] for label in np.unique(labels)],
        np.array([np.arange(n_features) for _ in labels]),
        data,
        f"{name if "/" not in name else ""} t_sne of the original data",
        name,
    )
    manifold(
        sample_vec,
        feature_vec,
        data,
        f"{name if "/" not in name else ""} u_map of biclustering solution",
        name,
        manifold_learner="umap",
    )
    manifold(
        [np.where(labels == label)[0] for label in np.unique(labels)],
        np.array([np.arange(n_features) for _ in labels]),
        data,
        f"{name if "/" not in name else ""} u_map of original data",
        name,
        manifold_learner="umap",
    )


def ISA_hyperparameter_test(
    data: np.array,
    vectorizer,
    labels,
    n_features,
    ts_feature: List[float] = [0.0],
    ts_sample: List[float] = [0.0],
    fs_ts: List[float] = [0.7],
) -> None:
    """
    Perform ISA using different hyperparameters.
    :param data: The data.
    :param vectorizer: The vectorizer.
    :param labels: The labels.
    :param n_features: The number of features.
    :param ts_feature: A list containing feature thresholds.
    :param ts_sample: A list contatining sample thresholds.
    :param fs_ts: A list Fusion similarity threshold.
    :return: None
    """

    for i in range(len(ts_feature)):
        for j in range(len(ts_sample)):
            for k in range(len(fs_ts)):
                sample_vec, feature_vec = isa(
                    data,
                    n_initial=1000,
                    n_updates=20,
                    thresh_feature=ts_feature[i],
                    thresh_sample=ts_sample[j],
                    fusion_similarity_threshold=fs_ts[k],
                )
                name = os.path.join(
                    "ISA", f"f-{ts_feature[i]} s-{ts_sample[j]} fs-{fs_ts[k]}"
                )
                display(
                    sample_vec, feature_vec, name, vectorizer, data, labels, n_features
                )


def main() -> None:
    """
    Perform experiments.
    :return: Nothing.
    """
    # Set random states.
    random_state = 42
    random.seed(random_state)
    np.random.seed(random_state)
    # Ensure the results folder exists.
    if not os.path.exists("Results"):
        os.mkdir("Results")
    # Get our data.
    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="train",
        categories=[
            "alt.atheism",
            "talk.religion.misc",
            "comp.graphics",
            "sci.space",
            "rec.sport.hockey",
            "soc.religion.christian",
        ],
        shuffle=True,
        random_state=random_state,
    )
    # Process the data.
    number_of_data_points = 300
    indices = np.arange(len(dataset.target))
    np.random.shuffle(indices)
    original = np.array(dataset.data)[indices[:number_of_data_points]]
    labels = dataset.target[indices[:number_of_data_points]]
    original = np.array(original)
    labels = np.array(labels)
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    true_k = unique_labels.shape[0]
    # Save a sample file.
    with open(os.path.join("Results", f"Raw Sample.txt"), "w") as file:
        file.write(str(original[0]))
    # Vectorize the data.
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=4, stop_words="english")
    t0 = time()
    x_tfidf = vectorizer.fit_transform(original)
    vectorization_time = time() - t0
    data = np.array(x_tfidf.todense())
    n_samples, n_features = data.shape
    print(
        f"{true_k} Categories | {n_samples} Samples | {n_features} Features | Vectorization done in "
        f"{vectorization_time:.3f} s"
    )

    count = Counter(labels).items()
    count = map(lambda x: (int(x[0]), x[1]), count)
    print(f"Number of data points in each category: \n{sorted(count)}\n")
    print(f"Name of each category:\n{dataset.target_names}\n")

    # Save dataset statistics.
    with open(os.path.join("Results", f"Dataset.csv"), "w") as file:
        file.write(
            f"Attribute,Value\n"
            f"Categories,{true_k}\n"
            f"Samples,{n_samples}\n"
            f"Features,{n_features}"
        )
    # Plot data.
    plt.figure(dpi=200)
    plt.imshow(data, cmap="gray")
    plt.ylabel("Documents")
    plt.xlabel("Features")
    # Perform ISA.
    t0 = time()
    # sample_vec, feature_vec = isa(
    #     data,
    #     n_initial=2000,
    #     n_updates=20,
    #     thresh_feature=1.6,
    #     thresh_sample=1.1,
    #     fusion_similarity_threshold=0.7,
    # )
    isa_time = time() - t0
    print(f"ISA done in {isa_time:.3f} s")
    # display(sample_vec, feature_vec, "ISA", vectorizer, data, labels, n_features)

    # Test hyperparameters for ISA

    ISA_hyperparameter_test(
        data=data,
        vectorizer=vectorizer,
        n_features=n_features,
        labels=labels,
        ts_feature=[0.5, 0.8, 1],
        ts_sample=[0.9, 1.5, 1.8, 2.2],
        fs_ts=[0.6],
    )

    # Perform K-Means.
    kmeans = KMeans(n_clusters=true_k, random_state=random_state)
    t0 = time()
    kmeans.fit(data)
    kmeans_time = time() - t0
    print(f"K-Means done in {kmeans_time:.3f} s")
    # Construct sample_vec.
    sample_vec = []
    for cluster_label in range(true_k):
        sample_indices = np.where(kmeans.labels_ == cluster_label)[0]
        sample_vec.append(set(sample_indices))
    # Construct feature_vec from cluster centroids.
    n_top_features = 50
    feature_vec = []
    for cluster_label in range(true_k):
        centroid = kmeans.cluster_centers_[cluster_label]
        top_feature_indices = np.argsort(centroid)[-n_top_features:]
        feature_vec.append(set(top_feature_indices))
    display(sample_vec, feature_vec, "K-Means", vectorizer, data, labels, n_features)
    # Perform spectral clustering.
    spectral = SpectralClustering(
        n_clusters=true_k, random_state=random_state, affinity="nearest_neighbors"
    )
    t0 = time()
    spectral_labels = spectral.fit_predict(data)
    spectral_time = time() - t0
    print(f"Spectral Clustering done in {spectral_time:.3f} s")
    # Construct sample_vec from the spectral clustering assignments.
    sample_vec = []
    for cluster_label in range(true_k):
        sample_indices = np.where(spectral_labels == cluster_label)[0]
        sample_vec.append(set(sample_indices))
    # Construct feature_vec by computing a centroid for each cluster.
    n_top_features = 50
    feature_vec = []
    for cluster_label in range(true_k):
        cluster_data = data[list(sample_vec[cluster_label])]
        # Compute the mean TF-IDF vector for the cluster.
        centroid = np.mean(cluster_data, axis=0)
        top_feature_indices = np.argsort(centroid)[-n_top_features:]
        feature_vec.append(set(top_feature_indices))
    # Display results for Spectral Clustering.
    display(sample_vec, feature_vec, "Spectral", vectorizer, data, labels, n_features)
    # Save computation times statistics.
    with open(os.path.join("Results", f"Times.csv"), "w") as file:
        file.write(
            f"Model,Time (seconds)\n"
            f"Vectorization,{vectorization_time:.3f}\n"
            f"ISA,{isa_time:.3f}\n"
            f"K-Means,{kmeans_time:.3f}\n"
            f"Spectral,{spectral_time:.3f}"
        )


if __name__ == "__main__":
    main()
