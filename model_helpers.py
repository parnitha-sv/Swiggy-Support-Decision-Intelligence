
import csv
import numpy as np


def load_data():
    with open("data/records.csv", "r", newline="") as f:
        records = list(csv.DictReader(f))

    feature_names = [
        "customer_tenure_months",
        "earlier_ticket_count_90d",
        "sla_consumed_ratio",
        "account_value_inr_thousands",
        "satisfaction_score",
        "reopened_flag",
        "self_service_first_flag"
    ]

    X = np.array([
        [float(row[name]) for name in feature_names]
        for row in records
    ], dtype=float)

    y = np.array([
        int(row["escalated"]) for row in records
    ], dtype=int)

    return X, y


def prepare_data():
    X, y = load_data()

    negative = np.where(y == 0)[0]
    positive = np.where(y == 1)[0]

    n0 = 7 * len(negative) // 10
    n1 = 7 * len(positive) // 10

    train_idx = np.sort(np.concatenate((
        negative[:n0], positive[:n1]
    )))

    test_idx = np.sort(np.concatenate((
        negative[n0:], positive[n1:]
    )))

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    mean = np.mean(X_train, axis=0)
    sd = np.std(X_train, axis=0, ddof=1)

    X_train_std = (X_train - mean) / sd
    X_test_std = (X_test - mean) / sd

    return (
        X_train, X_test, y_train, y_test,
        X_train_std, X_test_std, mean, sd
    )


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, probabilities):
    probabilities = np.clip(
        probabilities, 1e-15, 1 - 1e-15
    )

    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


def train_logistic(X, y, learning_rate=0.1,
                   updates=1000, checkpoints=None):

    if checkpoints is None:
        checkpoints = []

    n = len(y)
    weights = np.zeros(X.shape[1])
    bias = 0.0
    history = {}

    def current_loss():
        return binary_cross_entropy(
            y, sigmoid(X @ weights + bias)
        )

    if 0 in checkpoints:
        history[0] = current_loss()

    for epoch in range(1, updates + 1):

        probabilities = sigmoid(X @ weights + bias)
        error = probabilities - y

        grad_w = (X.T @ error) / n
        grad_b = np.mean(error)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if epoch in checkpoints:
            history[epoch] = current_loss()

    return weights, bias, history


def knn_predict(X_reference, y_reference, X_query, k):

    predictions = []
    probabilities = []

    for query in X_query:

        distances = np.sqrt(
            np.sum((X_reference - query) ** 2, axis=1)
        )

        nearest = np.argsort(
            distances, kind="stable"
        )[:k]

        positive_count = int(
            np.sum(y_reference[nearest])
        )

        probabilities.append(positive_count / k)

        predictions.append(
            int(positive_count >= (k + 1) // 2)
        )

    return (
        np.array(predictions, dtype=int),
        np.array(probabilities, dtype=float)
    )


def classification_metrics(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }


def calculate_f1(y_true, predictions):

    tp = np.sum((y_true == 1) & (predictions == 1))
    fp = np.sum((y_true == 0) & (predictions == 1))
    fn = np.sum((y_true == 1) & (predictions == 0))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
