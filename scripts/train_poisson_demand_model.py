
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
TRAIN_FILE = "data/processed/train_demand.csv"
TEST_FILE = "data/processed/test_demand.csv"

# Save the new model separately.
MODEL_PATH = Path("models/demand_model_lagged.joblib")

VALIDATION_OUTPUT = "data/processed/alpha_validation_lagged.csv"
DIAGNOSTICS_OUTPUT = "data/processed/product_diagnostics_lagged.csv"


# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

train["hour"] = pd.to_datetime(train["hour"], utc=True)
test["hour"] = pd.to_datetime(test["hour"], utc=True)

train["product_id"] = train["product_id"].astype(str)
test["product_id"] = test["product_id"].astype(str)

print("========== DATA ==========")
print(f"Training rows: {len(train):,}")
print(f"Testing rows: {len(test):,}")

required_columns = [
    "product_id",
    "hour",
    "log_price",
    "log_previous_hour_views",
    "hour_of_day",
    "day_of_week",
    "purchases",
]

for column in required_columns:
    if column not in train.columns or column not in test.columns:
        raise ValueError(
            f"Required column '{column}' is missing. "
            "Run scripts/prepare_model_data.py first."
        )

if train[required_columns].isna().any().any():
    raise ValueError("Training data contains missing required values.")

if test[required_columns].isna().any().any():
    raise ValueError("Testing data contains missing required values.")

if (train["purchases"] < 0).any() or (test["purchases"] < 0).any():
    raise ValueError("Purchases cannot be negative.")


# ---------------------------------------------------------
# 3. Create cyclical time features
# ---------------------------------------------------------
for df in [train, test]:
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour_of_day"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour_of_day"] / 24
    )


# ---------------------------------------------------------
# 4. Create product-specific price effects
# ---------------------------------------------------------
all_products = pd.concat(
    [train["product_id"], test["product_id"]],
    ignore_index=True,
)

product_dummies = pd.get_dummies(
    all_products,
    prefix="product",
    drop_first=True,
    dtype=float,
)

n_train = len(train)

train_dummies = product_dummies.iloc[:n_train].copy()
test_dummies = product_dummies.iloc[n_train:].copy()

train_dummies.index = train.index
test_dummies.index = test.index

# Product-specific log-price interactions
train_interactions = train_dummies.mul(
    train["log_price"], axis=0
)

test_interactions = test_dummies.mul(
    test["log_price"], axis=0
)

train_interactions.columns = [
    f"{column}_x_log_price"
    for column in train_interactions.columns
]

test_interactions.columns = train_interactions.columns


# ---------------------------------------------------------
# 5. Prepare model features
# ---------------------------------------------------------
BASE_FEATURES = [
    "log_price",
    "log_previous_hour_views",
    "hour_sin",
    "hour_cos",
    "day_of_week",
]

X_train = pd.concat(
    [train[BASE_FEATURES], train_dummies, train_interactions],
    axis=1,
)

X_test = pd.concat(
    [test[BASE_FEATURES], test_dummies, test_interactions],
    axis=1,
)

# Ensure identical feature columns and ordering
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

y_train = train["purchases"]
y_test = test["purchases"]

print("\n========== FEATURES ==========")
print(f"Base features: {len(BASE_FEATURES)}")
print(f"Product indicators: {train_dummies.shape[1]}")
print(f"Product-price interactions: {train_interactions.shape[1]}")
print(f"Total features: {X_train.shape[1]}")
print(f"Traffic feature: log_previous_hour_views")


# ---------------------------------------------------------
# 6. Chronological validation and alpha selection
# ---------------------------------------------------------
print("\n========== TIME-BASED VALIDATION ==========")

unique_hours = np.sort(train["hour"].unique())

if len(unique_hours) < 5:
    raise ValueError(
        "Not enough unique training hours for validation."
    )

split_index = int(len(unique_hours) * 0.80)
validation_start = unique_hours[split_index]

fit_mask = train["hour"] < validation_start
validation_mask = train["hour"] >= validation_start

X_fit = X_train.loc[fit_mask]
y_fit = y_train.loc[fit_mask]

X_validation = X_train.loc[validation_mask]
y_validation = y_train.loc[validation_mask]

print(f"Fit rows: {len(X_fit):,}")
print(f"Validation rows: {len(X_validation):,}")
print(f"Validation starts: {pd.Timestamp(validation_start)}")

alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]
validation_results = []

for alpha in alpha_values:
    candidate_model = PoissonRegressor(
        alpha=alpha,
        max_iter=2000,
    )

    candidate_model.fit(X_fit, y_fit)
    validation_predictions = candidate_model.predict(X_validation)

    validation_mae = mean_absolute_error(
        y_validation, validation_predictions
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation, validation_predictions
        )
    )

    validation_results.append({
        "alpha": alpha,
        "MAE": validation_mae,
        "RMSE": validation_rmse,
    })

validation_results = pd.DataFrame(validation_results)

print("\nValidation results:")
print(
    validation_results.to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
        },
    )
)

best_alpha = float(
    validation_results.loc[
        validation_results["RMSE"].idxmin(), "alpha"
    ]
)

print(f"\nSelected alpha: {best_alpha}")

validation_results.to_csv(VALIDATION_OUTPUT, index=False)
print(f"Validation results saved to: {VALIDATION_OUTPUT}")


# ---------------------------------------------------------
# 7. Train final model on all training data
# ---------------------------------------------------------
print("\n========== FINAL MODEL TRAINING ==========")

model = PoissonRegressor(
    alpha=best_alpha,
    max_iter=2000,
)

model.fit(X_train, y_train)

print("Lagged-views model trained successfully.")

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# 8. Evaluate on the chronological test set
# ---------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

print("\n========== TEST PERFORMANCE ==========")
print(f"Selected alpha: {best_alpha}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Minimum prediction: {y_pred.min():.4f}")
print(f"Maximum prediction: {y_pred.max():.4f}")


# ---------------------------------------------------------
# 9. Analyze zero-purchase and active-purchase rows
# ---------------------------------------------------------
print("\n========== PREDICTION ANALYSIS ==========")

zero_mask = y_test == 0
nonzero_mask = y_test > 0

for name, mask in [
    ("Zero-purchase hours", zero_mask),
    ("Non-zero-purchase hours", nonzero_mask),
]:
    actual = y_test[mask]
    predicted = y_pred[mask]

    print(f"\n--- {name} ---")
    print(f"Rows: {int(mask.sum()):,}")

    if len(actual) > 0:
        print(
            f"MAE: {mean_absolute_error(actual, predicted):.4f}"
        )
        print(
            f"RMSE: "
            f"{np.sqrt(mean_squared_error(actual, predicted)):.4f}"
        )
        print(f"Actual mean: {actual.mean():.4f}")
        print(f"Predicted mean: {predicted.mean():.4f}")


# ---------------------------------------------------------
# 10. Sample predictions
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "product_id": test["product_id"].values,
    "hour": test["hour"].values,
    "actual": y_test.values,
    "predicted": y_pred,
})

print("\n========== SAMPLE PREDICTIONS ==========")
print(comparison.head(20).to_string(index=False))


# ---------------------------------------------------------
# 11. Product-wise diagnostics
# ---------------------------------------------------------
diagnostics = comparison.copy()

product_results = []

for product_id, group in diagnostics.groupby("product_id"):
    actual = group["actual"]
    predicted = group["predicted"]

    product_results.append({
        "product_id": product_id,
        "records": len(group),
        "actual_mean": actual.mean(),
        "predicted_mean": predicted.mean(),
        "underprediction": actual.mean() - predicted.mean(),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
    })

product_results = pd.DataFrame(product_results)
product_results = product_results.sort_values(
    "underprediction",
    ascending=False,
)

print("\n========== PRODUCT-WISE DIAGNOSTICS ==========")
print(
    product_results.to_string(
        index=False,
        formatters={
            "actual_mean": "{:.4f}".format,
            "predicted_mean": "{:.4f}".format,
            "underprediction": "{:.4f}".format,
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
        },
    )
)

product_results.to_csv(DIAGNOSTICS_OUTPUT, index=False)
print(f"Diagnostics saved to: {DIAGNOSTICS_OUTPUT}")


# ---------------------------------------------------------
# 12. Inspect price coefficients
# ---------------------------------------------------------
coefficients = pd.Series(
    model.coef_,
    index=X_train.columns,
)

print("\n========== PRICE-RELATED COEFFICIENTS ==========")
print(f"Reference log-price coefficient: {coefficients['log_price']:.6f}")

interactions = coefficients[
    coefficients.index.str.endswith("_x_log_price")
]

print("\nProduct-specific price interactions:")
print(interactions.to_string())

print(f"\nIntercept: {model.intercept_:.6f}")


# ---------------------------------------------------------
# 13. Save the lagged model separately
# ---------------------------------------------------------
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

model_artifact = {
    "model": model,
    "alpha": best_alpha,
    "feature_columns": X_train.columns.tolist(),
    "base_features": BASE_FEATURES,
    "product_ids": sorted(train["product_id"].unique().tolist()),
    "traffic_feature": "log_previous_hour_views",
    "model_version": "lagged_views_v1",
    "evaluation": {
        "mae": float(mae),
        "rmse": float(rmse),
        "test_rows": int(len(test)),
    },
}

joblib.dump(model_artifact, MODEL_PATH)

print("\n========== MODEL SAVED ==========")
print(f"Model saved to: {MODEL_PATH}")
print(f"Selected alpha: {best_alpha}")
print(f"Saved feature count: {len(X_train.columns)}")
print("Existing demand_model.joblib was not overwritten.")
