
import csv
import math
import os
import numpy as np

from model_helpers import (
    prepare_data,
    sigmoid,
    binary_cross_entropy,
    train_logistic,
    knn_predict,
    classification_metrics,
    calculate_f1
)

# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

(
    X_train, X_test, y_train, y_test,
    X_train_std, X_test_std, train_mean, train_sd
) = prepare_data()

results = {}

results["train_rows"] = len(y_train)
results["train_positives"] = int(np.sum(y_train))
results["test_rows"] = len(y_test)
results["test_positives"] = int(np.sum(y_test))

n = len(y_train)

# --------------------------------------------------
# TASK 2.2: LOGISTIC REGRESSION
# --------------------------------------------------

main_checkpoints = [0, 1, 10, 100, 1000]

weights, bias, history = train_logistic(
    X_train_std,
    y_train,
    learning_rate=0.1,
    updates=1000,
    checkpoints=main_checkpoints
)

for epoch in main_checkpoints:
    results[f"loss_epoch_{epoch}"] = history[epoch]

test_probabilities = sigmoid(
    X_test_std @ weights + bias
)

results["test_bce"] = binary_cross_entropy(
    y_test,
    test_probabilities
)

for i, weight in enumerate(weights, start=1):
    results[f"weight_X{i}"] = weight

results["bias"] = bias

# --------------------------------------------------
# TASK 2.3: LEARNING RATE COMPARISON
# --------------------------------------------------

for eta in [0.01, 0.1, 1.0]:

    _, _, lr_history = train_logistic(
        X_train_std,
        y_train,
        learning_rate=eta,
        updates=200,
        checkpoints=[200]
    )

    results[f"loss_epoch_200_eta_{eta}"] = lr_history[200]

# --------------------------------------------------
# TASK 2.4: ODDS RATIOS
# --------------------------------------------------

for i, weight in enumerate(weights, start=1):
    results[f"odds_ratio_X{i}"] = math.exp(weight)

# --------------------------------------------------
# TASK 2.5: CLASSIFICATION METRICS
# --------------------------------------------------

for tau in [0.5, 0.3]:

    metrics = classification_metrics(
        y_test,
        test_probabilities,
        tau
    )

    for key, value in metrics.items():
        results[f"tau_{tau}_{key}"] = value

results["blind_accuracy"] = np.mean(y_test == 0)

# --------------------------------------------------
# TASK 2.6: THRESHOLD SWEEP
# --------------------------------------------------

threshold_rows = []

for tau in np.arange(0.1, 1.0, 0.1):

    metrics = classification_metrics(
        y_test,
        test_probabilities,
        float(tau)
    )

    tpr = metrics["recall"]

    fpr = (
        metrics["fp"] / (metrics["fp"] + metrics["tn"])
        if metrics["fp"] + metrics["tn"] else 0.0
    )

    threshold_rows.append([
        float(tau),
        metrics["tp"],
        metrics["fp"],
        metrics["tn"],
        metrics["fn"],
        tpr,
        fpr
    ])

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_thresholds.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "tau", "tp", "fp", "tn", "fn", "tpr", "fpr"
    ])

    for row in threshold_rows:

        writer.writerow([
            f"{row[0]:.4f}",
            row[1],
            row[2],
            row[3],
            row[4],
            f"{row[5]:.4f}",
            f"{row[6]:.4f}"
        ])

# --------------------------------------------------
# TASK 2.7: KNN CROSS-VALIDATION
# --------------------------------------------------

fold_ids = np.arange(len(y_train)) % 5

knn_cv_rows = []

for k in range(1, 22, 2):

    fold_f1_scores = []

    for fold in range(5):

        validation_mask = fold_ids == fold
        reference_mask = ~validation_mask

        X_reference = X_train_std[reference_mask]
        y_reference = y_train[reference_mask]

        X_validation = X_train_std[validation_mask]
        y_validation = y_train[validation_mask]

        predictions, _ = knn_predict(
            X_reference,
            y_reference,
            X_validation,
            k
        )

        fold_f1 = calculate_f1(
            y_validation,
            predictions
        )

        fold_f1_scores.append(fold_f1)

    mean_cv_f1 = float(np.mean(fold_f1_scores))

    knn_cv_rows.append([k, mean_cv_f1])

best_row = max(
    knn_cv_rows,
    key=lambda row: (row[1], -row[0])
)

best_k = best_row[0]

results["knn_best_k"] = best_k

with open("outputs/part2_knn_cv.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["K", "mean_cv_f1"])

    for k, score in knn_cv_rows:
        writer.writerow([k, f"{score:.4f}"])

# --------------------------------------------------
# KNN TEST EVALUATION
# --------------------------------------------------

knn_test_predictions, knn_test_probabilities = knn_predict(
    X_train_std,
    y_train,
    X_test_std,
    best_k
)

knn_metrics = classification_metrics(
    y_test,
    knn_test_probabilities,
    0.5
)

for key in [
    "tp", "fp", "tn", "fn",
    "accuracy", "precision", "recall", "f1"
]:

    results[f"knn_{key}"] = knn_metrics[key]

# --------------------------------------------------
# TASK 2.8: COST-BASED THRESHOLD
# --------------------------------------------------

def calculate_cost(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    return 50 * fp + 400 * fn


train_probabilities = sigmoid(
    X_train_std @ weights + bias
)

threshold_candidates = [
    k / 20 for k in range(1, 20)
]

training_costs = [
    calculate_cost(
        y_train,
        train_probabilities,
        tau
    )
    for tau in threshold_candidates
]

best_position = int(np.argmin(training_costs))

cost_tau = threshold_candidates[best_position]
cost_train = training_costs[best_position]

results["cost_tau"] = cost_tau
results["cost_train_at_tau"] = cost_train

results["cost_test_at_tau"] = calculate_cost(
    y_test,
    test_probabilities,
    cost_tau
)

results["cost_test_at_0.5"] = calculate_cost(
    y_test,
    test_probabilities,
    0.5
)

results["cost_test_flag_nobody"] = (
    400 * int(np.sum(y_test))
)

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

with open("outputs/part2_results.txt", "w") as f:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted = str(value)

        elif isinstance(value, str):
            formatted = value

        else:
            formatted = f"{value:.4f}"

        f.write(f"{key} = {formatted}\n")


print("PART 2 COMPLETED SUCCESSFULLY")
print("Total results:", len(results))

print("\nGenerated files:")
print("outputs/part2_results.txt")
print("outputs/part2_thresholds.csv")
print("outputs/part2_knn_cv.csv")
