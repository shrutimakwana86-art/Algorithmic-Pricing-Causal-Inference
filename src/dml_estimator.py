import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML

# 1. Load dataset
df = pd.read_csv("data/regional_pricing_data.csv")

Y = df["revenue"].values               # Outcome variable
T = df["treatment"].values             # Treatment indicator (0 or 1)
X = df[["demand_index"]].values        # Effect modifiers
W = df[["competitor_price"]].values    # Confounding variables

# 2. Fit Double Machine Learning (DML) estimator
dml = LinearDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestRegressor(n_estimators=100, random_state=42),
    random_state=42
)

dml.fit(Y, T, X=X, W=W)

# 3. Estimate Average Treatment Effect (ATE) and 95% Confidence Interval
ate_dml = dml.ate(X=X)
ate_interval = dml.ate_interval(X=X, alpha=0.05)

print("--- Double Machine Learning (DML) Results ---")
print(f"Estimated Average Treatment Effect (ATE): +{ate_dml:.2f} revenue units/day")
print(f"95% Confidence Interval: [{ate_interval[0]:.2f}, {ate_interval[1]:.2f}]")