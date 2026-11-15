
import csv
import math
import os
import numpy as np

from model_helpers import (
    prepare_data,
    sigmoid,
    train_logistic,
    knn_predict,
    classification_metrics
)

# ==================================================
# LOAD THE SAME TRAINING AND TEST DATA AS PART 2
# ==================================================

(
    X_train, X_test, y_train, y_test,
    X_train_std, X_test_std, train_mean, train_sd
) = prepare_data()

results = {}
os.makedirs("outputs", exist_ok=True)


# ==================================================
# TASK 4.1: ENTROPY, GINI AND CANDIDATE SPLITS
# ==================================================

def entropy(labels):

    n = len(labels)

    if n == 0:
        return 0.0

    positives = np.sum(labels == 1)
    negatives = n - positives

    value = 0.0

    for count in [positives, negatives]:

        if count > 0:
            p = count / n
            value -= p * math.log2(p)

    return value


def gini(labels):

    n = len(labels)

    if n == 0:
        return 0.0

    positives = np.sum(labels == 1) / n
    negatives = 1 - positives

    return 1 - positives ** 2 - negatives ** 2


root_entropy = entropy(y_train)
root_gini = gini(y_train)

results["root_entropy"] = root_entropy
results["root_gini"] = root_gini

# Original, unstandardised training features
X_original = X_train

# Candidate splits: the condition defines the YES side
split_conditions = {
    "S1": X_original[:, 5] == 1,
    "S2": X_original[:, 6] == 1,
    "S3": X_original[:, 2] > 0.5,
    "S4": X_original[:, 0] <= 24
}

split_rows = []
split_gains = {}

for split_name, yes_mask in split_conditions.items():

    no_mask = ~yes_mask

    yes_labels = y_train[yes_mask]
    no_labels = y_train[no_mask]

    n_yes = len(yes_labels)
    n_no = len(no_labels)

    positives_yes = int(np.sum(yes_labels))
    positives_no = int(np.sum(no_labels))

    weighted_entropy = (
        (n_yes / len(y_train)) * entropy(yes_labels)
        + (n_no / len(y_train)) * entropy(no_labels)
    )

    weighted_gini = (
        (n_yes / len(y_train)) * gini(yes_labels)
        + (n_no / len(y_train)) * gini(no_labels)
    )

    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_gains[split_name] = {
        "info_gain": info_gain,
        "gini_gain": gini_gain
    }

    split_rows.append([
        split_name,
        n_yes,
        positives_yes,
        n_no,
        positives_no,
        info_gain,
        gini_gain
    ])

best_split_by_info_gain = max(
    split_gains,
    key=lambda name: split_gains[name]["info_gain"]
)

best_split_by_gini_gain = max(
    split_gains,
    key=lambda name: split_gains[name]["gini_gain"]
)

results["best_split_by_info_gain"] = best_split_by_info_gain
results["best_split_by_gini_gain"] = best_split_by_gini_gain

with open("outputs/part4_splits.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "split",
        "n_yes",
        "positives_yes",
        "n_no",
        "positives_no",
        "info_gain",
        "gini_gain"
    ])

    for row in split_rows:

        writer.writerow([
            *row[:5],
            f"{row[5]:.4f}",
            f"{row[6]:.4f}"
        ])


# ==================================================
# TASK 4.2: STUMP, LOGISTIC, KNN AND VOTING
# ==================================================

best_mask = split_conditions[best_split_by_info_gain]

stump_rate_yes = float(np.mean(y_train[best_mask]))
stump_rate_no = float(np.mean(y_train[~best_mask]))

results["stump_rate_yes"] = stump_rate_yes
results["stump_rate_no"] = stump_rate_no

# Logistic regression: same Task 2.2 model
weights, bias, _ = train_logistic(
    X_train_std,
    y_train,
    learning_rate=0.1,
    updates=1000
)

logistic_probabilities = sigmoid(
    X_test_std @ weights + bias
)

logistic_predictions = (
    logistic_probabilities >= 0.3
).astype(int)

# KNN: same Task 2.7 selected K
fold_ids = np.arange(len(y_train)) % 5

def calculate_f1_score(y_true, predictions):

    tp = np.sum((y_true == 1) & (predictions == 1))
    fp = np.sum((y_true == 0) & (predictions == 1))
    fn = np.sum((y_true == 1) & (predictions == 0))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )


cv_results = []

for k in range(1, 22, 2):

    fold_scores = []

    for fold in range(5):

        validation_mask = fold_ids == fold
        reference_mask = ~validation_mask

        predictions, _ = knn_predict(
            X_train_std[reference_mask],
            y_train[reference_mask],
            X_train_std[validation_mask],
            k
        )

        fold_scores.append(
            calculate_f1_score(
                y_train[validation_mask],
                predictions
            )
        )

    cv_results.append((k, np.mean(fold_scores)))

best_k = max(
    cv_results,
    key=lambda item: (item[1], -item[0])
)[0]

results["knn_best_k"] = best_k

knn_predictions, knn_probabilities = knn_predict(
    X_train_std,
    y_train,
    X_test_std,
    best_k
)

# Decision-tree stump predictions
test_yes_mask = best_mask_for_test = None

# Apply the selected split to original test features
if best_split_by_info_gain == "S1":
    test_yes_mask = X_test[:, 5] == 1
elif best_split_by_info_gain == "S2":
    test_yes_mask = X_test[:, 6] == 1
elif best_split_by_info_gain == "S3":
    test_yes_mask = X_test[:, 2] > 0.5
else:
    test_yes_mask = X_test[:, 0] <= 24

stump_probabilities = np.where(
    test_yes_mask,
    stump_rate_yes,
    stump_rate_no
)

stump_predictions = (
    stump_probabilities >= 0.3
).astype(int)

# Hard voting: at least two models predict positive
hard_votes = (
    logistic_predictions
    + knn_predictions
    + stump_predictions
)

hard_predictions = (hard_votes >= 2).astype(int)

# Soft voting: mean of three probabilities
soft_probabilities = np.mean(
    np.column_stack((
        logistic_probabilities,
        knn_probabilities,
        stump_probabilities
    )),
    axis=1
)

soft_predictions = (
    soft_probabilities >= 0.3
).astype(int)


# Save five model comparisons in required order
model_predictions = {
    "logistic": logistic_predictions,
    "knn": knn_predictions,
    "stump": stump_predictions,
    "hard_vote": hard_predictions,
    "soft_vote": soft_predictions
}

voting_rows = []

for model_name, predictions in model_predictions.items():

    tp = int(np.sum((y_test == 1) & (predictions == 1)))
    fp = int(np.sum((y_test == 0) & (predictions == 1)))
    tn = int(np.sum((y_test == 0) & (predictions == 0)))
    fn = int(np.sum((y_test == 1) & (predictions == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    voting_rows.append([
        model_name, tp, fp, tn, fn,
        precision, recall, f1
    ])

    results[f"{model_name}_tp"] = tp
    results[f"{model_name}_fp"] = fp
    results[f"{model_name}_tn"] = tn
    results[f"{model_name}_fn"] = fn
    results[f"{model_name}_precision"] = precision
    results[f"{model_name}_recall"] = recall
    results[f"{model_name}_f1"] = f1

with open("outputs/part4_voting.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "model", "tp", "fp", "tn", "fn",
        "precision", "recall", "f1"
    ])

    for row in voting_rows:

        writer.writerow([
            *row[:5],
            *[f"{value:.4f}" for value in row[5:]]
        ])


# Calculate Task 2.8 cost for all five models
for model_name, predictions in model_predictions.items():

    fp = int(np.sum((y_test == 0) & (predictions == 1)))
    fn = int(np.sum((y_test == 1) & (predictions == 0)))

    results[f"{model_name}_cost"] = 50 * fp + 400 * fn


# Read the logistic-regression cost calculated in Part 2
with open("outputs/part2_results.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(" = ", 1)

        if key == "cost_test_at_tau":
            results["logistic_task2_cost_comparator"] = int(value)
            break
    else:
        raise ValueError("cost_test_at_tau not found in part2_results.txt")
        
# ==================================================
# TASK 4.3: TWO ROUNDS OF ADABOOST BY HAND
# ==================================================

true_classes = np.array([
    1, 1, 1, -1, -1, -1, 1, -1
])

h1 = np.array([
    1, 1, -1, -1, -1, -1, -1, -1
])

h2 = np.array([
    -1, 1, 1, -1, 1, -1, 1, -1
])

sample_weights = np.full(8, 1 / 8)

adaboost_rows = []

alphas = []
round_errors = []

for round_number, predictions in [
    (1, h1),
    (2, h2)
]:

    weights_before = sample_weights.copy()

    weighted_error = float(
        np.sum(
            weights_before[predictions != true_classes]
        )
    )

    alpha = 0.5 * math.log(
        (1 - weighted_error) / weighted_error
    )

    unnormalised_weights = (
        weights_before
        * np.exp(-alpha * true_classes * predictions)
    )

    sample_weights = (
        unnormalised_weights
        / np.sum(unnormalised_weights)
    )

    if round_number == 1:
        round1_misclassified_weight_after = float(
            np.sum(sample_weights[predictions != true_classes])
        )

    round_errors.append(weighted_error)
    alphas.append(alpha)

    for i in range(8):

        adaboost_rows.append([
            round_number,
            i + 1,
            weights_before[i],
            unnormalised_weights[i],
            sample_weights[i]
        ])

results["round1_error"] = round_errors[0]
results["round1_alpha"] = alphas[0]

results["round2_error"] = round_errors[1]
results["round2_alpha"] = alphas[1]

ensemble_scores = alphas[0] * h1 + alphas[1] * h2

ensemble_predictions = np.sign(ensemble_scores)

ensemble_accuracy = np.mean(
    ensemble_predictions == true_classes
)

results["ensemble_accuracy"] = ensemble_accuracy

# Total weight of the two points misclassified in round 1
results["round1_misclassified_weight_after"] = (
    round1_misclassified_weight_after
)

with open("outputs/part4_adaboost.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "round",
        "point",
        "weight_before",
        "weight_unnormalised",
        "weight_after"
    ])

    for row in adaboost_rows:

        writer.writerow([
            row[0],
            row[1],
            *[f"{value:.4f}" for value in row[2:]]
        ])


# ==================================================
# TASK 4.4: OUT-OF-BAG FRACTION
# ==================================================

oob_fraction = (1 - 1 / 560) ** 560

results["oob_fraction"] = oob_fraction


# ==================================================
# TASK 4.5: NEURAL NETWORK ARCHITECTURE COUNTS
# ==================================================

architecture_rows = []

# MODEL A OPTION 1: FULLY CONNECTED NETWORK
mlp_hidden1 = 784 * 16 + 16
mlp_hidden2 = 16 * 16 + 16
mlp_output = 16 * 10 + 10

mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

architecture_rows.extend([
    ["mlp_hidden1", mlp_hidden1],
    ["mlp_hidden2", mlp_hidden2],
    ["mlp_output", mlp_output]
])

results["mlp_total"] = mlp_total


# MODEL A OPTION 2: CONVOLUTIONAL NETWORK

# Input: 28 x 28 x 1
# Conv1: 5 x 5, 6 filters, stride 1, padding 2

cnn_conv1 = (5 * 5 * 1 + 1) * 6

conv1_height = (28 - 5 + 2 * 2) / 1 + 1
conv1_width = (28 - 5 + 2 * 2) / 1 + 1

# Pool1: 2 x 2, stride 2

pool1_height = int((conv1_height - 2) / 2 + 1)
pool1_width = int((conv1_width - 2) / 2 + 1)

# Conv2: 5 x 5, 16 filters, stride 1, padding 0

cnn_conv2 = (5 * 5 * 6 + 1) * 16

conv2_height = int((pool1_height - 5) / 1 + 1)
conv2_width = int((pool1_width - 5) / 1 + 1)

# Pool2: 2 x 2, stride 2

pool2_height = int((conv2_height - 2) / 2 + 1)
pool2_width = int((conv2_width - 2) / 2 + 1)

flattened_features = (
    pool2_height * pool2_width * 16
)

cnn_dense1 = flattened_features * 32 + 32
cnn_dense2 = 32 * 10 + 10

cnn_total = (
    cnn_conv1
    + cnn_conv2
    + cnn_dense1
    + cnn_dense2
)

architecture_rows.extend([
    ["cnn_conv1", cnn_conv1],
    ["cnn_conv2", cnn_conv2],
    ["cnn_dense1", cnn_dense1],
    ["cnn_dense2", cnn_dense2]
])

results["cnn_total"] = cnn_total

# CNN output shapes for README
results["cnn_conv1_shape"] = "28x28x6"
results["cnn_pool1_shape"] = "14x14x6"
results["cnn_conv2_shape"] = "10x10x16"
results["cnn_pool2_shape"] = "5x5x16"
results["cnn_flattened_features"] = flattened_features


# Save architecture CSV in exact required order
with open("outputs/part4_architecture.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["layer", "parameters"])

    for row in architecture_rows:
        writer.writerow(row)


# Fully connected vs convolutional layer comparison
fc_layer_params = (128 * 128 * 3) * 1024 + 1024

conv_layer_params = (3 * 3 * 3 + 1) * 64

results["fc_layer_params"] = fc_layer_params
results["conv_layer_params"] = conv_layer_params


# MODEL B: RECURRENT NEURAL NETWORK

# Input dimension = 64
# Hidden dimension = 128
# Output classes = 16

W_xh = 128 * 64
W_hh = 128 * 128
W_hy = 16 * 128

b_h = 128
b_y = 16

rnn_total = W_xh + W_hh + W_hy + b_h + b_y

results["rnn_W_xh_shape"] = "128x64"
results["rnn_W_hh_shape"] = "128x128"
results["rnn_W_hy_shape"] = "16x128"

results["rnn_W_xh"] = W_xh
results["rnn_W_hh"] = W_hh
results["rnn_W_hy"] = W_hy
results["rnn_b_h"] = b_h
results["rnn_b_y"] = b_y

results["rnn_total"] = rnn_total


# ==================================================
# SAVE PART 4 RESULTS
# ==================================================

with open("outputs/part4_results.txt", "w") as f:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted = str(value)

        elif isinstance(value, str):
            formatted = value

        else:
            formatted = f"{value:.4f}"

        f.write(f"{key} = {formatted}\n")


print("PART 4 COMPLETED SUCCESSFULLY")

print("Total results:", len(results))

print("\nGenerated files:")
print("outputs/part4_results.txt")
print("outputs/part4_splits.csv")
print("outputs/part4_voting.csv")
print("outputs/part4_adaboost.csv")
print("outputs/part4_architecture.csv")
