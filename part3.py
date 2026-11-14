
import csv
import math
import os
import numpy as np

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

with open("data/records.csv", "r", newline="") as f:
    records = list(csv.DictReader(f))

feature_names = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score"
]

X = np.array(
    [[float(row[name]) for name in feature_names] for row in records],
    dtype=float
)

y = np.array([int(row["escalated"]) for row in records])
resolution = np.array(
    [float(row["resolution_minutes"]) for row in records]
)

n = len(X)
results = {}

# --------------------------------------------------
# TASK 3.1: STANDARDISATION
# --------------------------------------------------

mean_X = np.mean(X, axis=0)
sd_X = np.std(X, axis=0, ddof=1)

Z = (X - mean_X) / sd_X

# --------------------------------------------------
# TASK 3.1: K-MEANS FROM SCRATCH
# --------------------------------------------------

def kmeans(data, k):

    step = len(data) // k

    starting_indices = [
        i * step for i in range(k)
    ]

    centroids = data[starting_indices].copy()

    labels = np.full(len(data), -1, dtype=int)

    while True:

        distances = np.sum(
            (data[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        new_labels = np.argmin(distances, axis=1)

        # Stop at the first assignment pass with no changes
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels.copy()

        # Update centroids
        for cluster in range(k):

            members = data[labels == cluster]

            if len(members) > 0:
                centroids[cluster] = np.mean(members, axis=0)

    # Calculate WCSS
    wcss = 0.0

    for cluster in range(k):

        members = data[labels == cluster]

        if len(members) > 0:
            differences = members - centroids[cluster]

            wcss += np.sum(differences ** 2)

    return labels, centroids, wcss


cluster_results = {}

for k in [2, 3, 4, 5, 6]:

    labels, centroids, wcss = kmeans(Z, k)

    cluster_results[k] = (labels, centroids, wcss)

    results[f"wcss_k{k}"] = wcss


# --------------------------------------------------
# TASK 3.2: ELBOW METHOD
# --------------------------------------------------

elbow_k = None

for k in [3, 4, 5]:

    previous_drop = (
        cluster_results[k - 1][2] - cluster_results[k][2]
    )

    current_drop = (
        cluster_results[k][2] - cluster_results[k + 1][2]
    )

    if current_drop < 0.5 * previous_drop:

        elbow_k = k
        break

results["elbow_k"] = elbow_k


# --------------------------------------------------
# TASK 3.3: PROFILE K=3 SEGMENTS
# --------------------------------------------------

labels_3 = cluster_results[3][0]

cluster_mean_X3 = {}

for cluster in range(3):

    cluster_mean_X3[cluster] = np.mean(
        X[labels_3 == cluster, 2]
    )

# Sort clusters by ascending original X3 mean
ordered_clusters = sorted(
    range(3),
    key=lambda cluster: cluster_mean_X3[cluster]
)

segment_labels = np.empty(n, dtype=object)

segment_names = ["A", "B", "C"]

segment_rows = []

for segment, cluster in zip(segment_names, ordered_clusters):

    mask = labels_3 == cluster

    segment_labels[mask] = segment

    size = int(np.sum(mask))

    mean_values = np.mean(X[mask], axis=0)

    outcome_rate = np.mean(y[mask])

    mean_value = np.mean(resolution[mask])

    results[f"size_segment_{segment}"] = size

    segment_rows.append([
        segment,
        size,
        *mean_values,
        outcome_rate,
        mean_value
    ])


# Save segment CSV

os.makedirs("outputs", exist_ok=True)

with open("outputs/part3_segments.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "segment",
        "size",
        "mean_X1",
        "mean_X2",
        "mean_X3",
        "mean_X4",
        "mean_X5",
        "outcome_rate",
        "mean_value"
    ])

    for row in segment_rows:

        writer.writerow([
            row[0],
            row[1],
            *[f"{value:.4f}" for value in row[2:]]
        ])


# --------------------------------------------------
# TASK 3.4: HIERARCHICAL CLUSTERING
# --------------------------------------------------

distance_matrix = np.array([

    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]

])

def cluster_name(cluster):

    return "+".join(
        f"E{i + 1}" for i in sorted(cluster)
    )


def hierarchical_clustering(matrix, linkage):

    clusters = [[i] for i in range(len(matrix))]

    merge_rows = []
    heights = []

    for merge_no in range(1, len(matrix)):

        best_distance = float("inf")
        best_pair = None

        # Find the closest pair of clusters
        for i in range(len(clusters)):

            for j in range(i + 1, len(clusters)):

                distances = [
                    matrix[a, b]
                    for a in clusters[i]
                    for b in clusters[j]
                ]

                if linkage == "single":
                    distance = min(distances)

                elif linkage == "complete":
                    distance = max(distances)

                else:
                    distance = np.mean(distances)

                if distance < best_distance:

                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # Ensure cluster_a has the smaller minimum member
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        merge_rows.append([
            linkage,
            merge_no,
            cluster_name(cluster_a),
            cluster_name(cluster_b),
            best_distance
        ])

        heights.append(best_distance)

        merged_cluster = sorted(cluster_a + cluster_b)

        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]

        clusters.append(merged_cluster)

        clusters.sort(key=lambda cluster: min(cluster))

    # Find the largest gap between consecutive heights
    gaps = np.diff(heights)

    largest_gap_index = int(np.argmax(gaps))

    # Number of merges below the cut
    merges_below_cut = largest_gap_index + 1

    clusters_at_cut = 6 - merges_below_cut

    return merge_rows, heights, clusters_at_cut


hierarchical_rows = []

for linkage in ["single", "complete", "average"]:

    rows, heights, cut_count = hierarchical_clustering(
        distance_matrix,
        linkage
    )

    hierarchical_rows.extend(rows)

    results[f"{linkage}_clusters_at_cut"] = cut_count


# Save hierarchical clustering CSV

with open("outputs/part3_hierarchical.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ])

    for row in hierarchical_rows:

        writer.writerow([
            row[0],
            row[1],
            row[2],
            row[3],
            f"{row[4]:.4f}"
        ])


# --------------------------------------------------
# TASK 3.5: PRINCIPAL COMPONENT ANALYSIS
# --------------------------------------------------

# Step 1: Standardised data has a zero mean vector
# Step 2: Centre the data

Z_centered = Z - np.mean(Z, axis=0)

# Step 3: Covariance matrix
covariance_matrix = (
    Z_centered.T @ Z_centered
) / (n - 1)

# Step 4: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(
    covariance_matrix
)

# Step 5: Sort in descending eigenvalue order
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Step 6: Fix eigenvector signs
for j in range(eigenvectors.shape[1]):

    largest_index = np.argmax(
        np.abs(eigenvectors[:, j])
    )

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1

# Keep top two components
W = eigenvectors[:, :2]

# Step 7: Project data
projections = Z_centered @ W

eigenvalue_sum = np.sum(eigenvalues)

explained_ratios = eigenvalues / eigenvalue_sum

cumulative_ratios = np.cumsum(explained_ratios)

results["eigenvalue_sum"] = eigenvalue_sum

for i in range(5):

    results[f"eigenvalue_{i + 1}"] = eigenvalues[i]

    results[f"explained_ratio_{i + 1}"] = explained_ratios[i]

    results[f"cumulative_ratio_{i + 1}"] = cumulative_ratios[i]


# Minimum number of components for 80% variance
components_for_80 = int(
    np.argmax(cumulative_ratios >= 0.80) + 1
)

results["components_for_80pct"] = components_for_80


# PC1 and PC2 loadings

for i in range(5):

    results[f"pc1_loading_X{i + 1}"] = W[i, 0]

    results[f"pc2_loading_X{i + 1}"] = W[i, 1]


# First three row projections

for row_index in range(3):

    results[f"row{row_index + 1}_pc1"] = projections[row_index, 0]

    results[f"row{row_index + 1}_pc2"] = projections[row_index, 1]


# Mean PC1 score for each segment

for segment in segment_names:

    results[f"mean_pc1_segment_{segment}"] = np.mean(
        projections[segment_labels == segment, 0]
    )


# --------------------------------------------------
# SAVE PART 3 RESULTS
# --------------------------------------------------

with open("outputs/part3_results.txt", "w") as f:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted = str(value)

        elif isinstance(value, str):
            formatted = value

        else:
            formatted = f"{value:.4f}"

        f.write(f"{key} = {formatted}\n")


print("PART 3 COMPLETED SUCCESSFULLY")
print("Total results:", len(results))

print("\nGenerated files:")
print("outputs/part3_results.txt")
print("outputs/part3_segments.csv")
print("outputs/part3_hierarchical.csv")
