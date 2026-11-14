
import csv
import math
import os
import numpy as np

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

with open("data/records.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    records = list(reader)

def column(name):
    return np.array([float(row[name]) for row in records], dtype=float)

x1 = column("customer_tenure_months")
x2 = column("earlier_ticket_count_90d")
x3 = column("sla_consumed_ratio")
x4 = column("account_value_inr_thousands")
x5 = column("satisfaction_score")
x6 = column("reopened_flag")
x7 = column("self_service_first_flag")
y = column("resolution_minutes")

n = len(y)
results = {}

# --------------------------------------------------
# TASK 1.1: ESTIMATION AND 95% CONFIDENCE INTERVAL
# --------------------------------------------------

mean_y = np.mean(y)
sd_y = np.std(y, ddof=1)
se_y = sd_y / math.sqrt(n)

ci_low = mean_y - 1.96 * se_y
ci_high = mean_y + 1.96 * se_y

results["n"] = n
results["mean_value"] = mean_y
results["sd_value"] = sd_y
results["se_value"] = se_y
results["ci95_low"] = ci_low
results["ci95_high"] = ci_high

# --------------------------------------------------
# TASK 1.2: TWO-TAILED Z-TEST
# --------------------------------------------------

z_stat = (mean_y - 8.0) / se_y
reject_h0 = abs(z_stat) > 1.96

results["z_stat"] = z_stat
results["reject_h0"] = "yes" if reject_h0 else "no"

# --------------------------------------------------
# TASK 1.3: REQUIRED SAMPLE SIZE
# --------------------------------------------------

margin_error = 0.5
n_required = math.ceil((1.96 * sd_y / margin_error) ** 2)

results["n_required"] = n_required

# --------------------------------------------------
# REGRESSION HELPER: CLOSED-FORM SIMPLE REGRESSION
# --------------------------------------------------

def simple_regression(x, target):

    x_mean = np.mean(x)
    y_mean = np.mean(target)

    covariance = np.sum((x - x_mean) * (target - y_mean)) / (len(x) - 1)
    variance = np.sum((x - x_mean) ** 2) / (len(x) - 1)

    slope = covariance / variance
    intercept = y_mean - slope * x_mean

    fitted = intercept + slope * x

    tss = np.sum((target - y_mean) ** 2)
    ess = np.sum((fitted - y_mean) ** 2)
    rss = np.sum((target - fitted) ** 2)

    r2 = 1 - rss / tss
    rmse = math.sqrt(rss / len(target))
    mae = np.mean(np.abs(target - fitted))

    return slope, intercept, fitted, tss, ess, rss, r2, rmse, mae


# --------------------------------------------------
# TASK 1.4: SIMPLE LINEAR REGRESSION
# --------------------------------------------------

reg = simple_regression(x4, y)

reg_slope = reg[0]
reg_intercept = reg[1]
fitted = reg[2]
tss = reg[3]
ess = reg[4]
rss = reg[5]
r2 = reg[6]
rmse = reg[7]
mae = reg[8]

results["reg_slope"] = reg_slope
results["reg_intercept"] = reg_intercept
results["tss"] = tss
results["ess"] = ess
results["rss"] = rss

results["tss_equals_ess_plus_rss"] = (
    "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
)

results["r2"] = r2
results["rmse"] = rmse
results["mae"] = mae

# --------------------------------------------------
# TASK 1.5: GRADIENT DESCENT
# --------------------------------------------------

x_mean = np.mean(x4)
x_sd = np.std(x4, ddof=1)

z = (x4 - x_mean) / x_sd

m = 0.0
c = 0.0
eta = 0.1

def calculate_cost(m, c):
    prediction = m * z + c
    error = y - prediction
    return np.mean(error ** 2)

results["gd_J0"] = calculate_cost(m, c)

for iteration in range(1, 501):

    prediction = m * z + c
    error = y - prediction

    dm = (-2 / n) * np.sum(z * error)
    dc = (-2 / n) * np.sum(error)

    m = m - eta * dm
    c = c - eta * dc

    if iteration in [1, 10, 100, 500]:
        results[f"gd_J{iteration}"] = calculate_cost(m, c)

gd_slope = m / x_sd
gd_intercept = c - m * x_mean / x_sd

results["gd_slope"] = gd_slope
results["gd_intercept"] = gd_intercept
results["gd_slope_gap"] = abs(gd_slope - reg_slope)

# --------------------------------------------------
# TASK 1.6: RESIDUAL ANALYSIS AND LOG TRANSFORMATION
# --------------------------------------------------

# Raw residuals: stable sort by fitted values

raw_residuals = y - fitted

raw_order = np.argsort(fitted, kind="stable")

raw_lower = raw_residuals[raw_order[:400]]
raw_upper = raw_residuals[raw_order[400:]]

raw_sd_lower = np.std(raw_lower, ddof=1)
raw_sd_upper = np.std(raw_upper, ddof=1)

results["resid_sd_lower_raw"] = raw_sd_lower
results["resid_sd_upper_raw"] = raw_sd_upper
results["resid_sd_ratio_raw"] = raw_sd_upper / raw_sd_lower

# Log-log regression

log_x4 = np.log(x4)
log_y = np.log(y)

log_reg = simple_regression(log_x4, log_y)

log_slope = log_reg[0]
log_intercept = log_reg[1]
log_fitted = log_reg[2]
log_r2 = log_reg[6]
log_rmse = log_reg[7]

results["loglog_intercept"] = log_intercept
results["loglog_slope"] = log_slope
results["loglog_r2"] = log_r2
results["loglog_rmse"] = log_rmse

# Log-model residual analysis

log_residuals = log_y - log_fitted

log_order = np.argsort(log_fitted, kind="stable")

log_lower = log_residuals[log_order[:400]]
log_upper = log_residuals[log_order[400:]]

log_sd_lower = np.std(log_lower, ddof=1)
log_sd_upper = np.std(log_upper, ddof=1)

results["resid_sd_lower_log"] = log_sd_lower
results["resid_sd_upper_log"] = log_sd_upper
results["resid_sd_ratio_log"] = log_sd_upper / log_sd_lower

# --------------------------------------------------
# TASK 1.7: MULTIPLE LINEAR REGRESSION
# --------------------------------------------------

X = np.column_stack((
    np.ones(n),
    np.log(x4),
    x1,
    x2,
    x3,
    x5,
    x6,
    x7
))

Y = np.log(y)

# Normal equation using a linear solve
beta = np.linalg.solve(X.T @ X, X.T @ Y)

multiple_fitted = X @ beta

mr_tss = np.sum((Y - np.mean(Y)) ** 2)
mr_rss = np.sum((Y - multiple_fitted) ** 2)

mr_r2 = 1 - mr_rss / mr_tss

p = 7
mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)

loglog_adj_r2 = 1 - (1 - log_r2) * (n - 1) / (n - 1 - 1)

for i in range(8):
    results[f"mr_beta_{i}"] = beta[i]

results["mr_r2"] = mr_r2
results["mr_adj_r2"] = mr_adj_r2
results["loglog_adj_r2"] = loglog_adj_r2

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

with open("outputs/part1_results.txt", "w") as f:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted = str(value)

        elif isinstance(value, str):
            formatted = value

        else:
            formatted = f"{value:.4f}"

        f.write(f"{key} = {formatted}\n")

print("PART 1 COMPLETED SUCCESSFULLY")
print("Total results:", len(results))
print("\nGenerated file: outputs/part1_results.txt")
