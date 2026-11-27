import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 1. Load dataset and pivot into time-series matrix (Rows = day, Columns = city)
df = pd.read_csv("data/regional_pricing_data.csv")
pivot_rev = df.pivot(index="day", columns="city", values="revenue")

treated_city = "Vadodara"
donor_cities = ["Ahmedabad", "Surat", "Rajkot"]

# Pre-treatment period (Days 1 to 70)
Y_pre_treated = pivot_rev.loc[1:70, treated_city].values
X_pre_donors = pivot_rev.loc[1:70, donor_cities].values

# 2. Synthetic Control Optimization (weights sum to 1, non-negative)
def loss_function(weights):
    synthetic_pre = X_pre_donors @ weights
    return np.mean((Y_pre_treated - synthetic_pre) ** 2)

init_weights = np.ones(len(donor_cities)) / len(donor_cities)
bounds = [(0, 1) for _ in donor_cities]
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

result = minimize(loss_function, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
optimal_weights = result.x

print("--- Optimal Donor Pool Weights ---")
for city, w in zip(donor_cities, optimal_weights):
    print(f"{city}: {w:.4f}")

# 3. Compute Synthetic Vadodara across all 100 days
pivot_rev["Synthetic_Vadodara"] = pivot_rev[donor_cities].values @ optimal_weights
pivot_rev["Causal_Lift_Diff"] = pivot_rev[treated_city] - pivot_rev["Synthetic_Vadodara"]

# 4. Calculate Pre-Treatment Mean Squared Prediction Error (MSPE)
pre_mspe = np.mean((pivot_rev.loc[1:70, treated_city] - pivot_rev.loc[1:70, "Synthetic_Vadodara"]) ** 2)
post_lift_avg = pivot_rev.loc[71:100, "Causal_Lift_Diff"].mean()

print(f"\nPre-Treatment MSPE (Days 1-70): {pre_mspe:.2f}")
print(f"Average Daily Revenue Lift Post-Treatment (Days 71-100): +{post_lift_avg:.2f}")

# Save synthetic control output table
pivot_rev.to_csv("data/synthetic_control_results.csv")

# 5. Plot Actual vs Counterfactual Revenue Trajectory
plt.figure(figsize=(10, 5))
plt.plot(pivot_rev.index, pivot_rev[treated_city], label="Actual Vadodara (Treated)", color="blue", linewidth=2)
plt.plot(pivot_rev.index, pivot_rev["Synthetic_Vadodara"], label="Synthetic Vadodara (Counterfactual)", color="orange", linestyle="--", linewidth=2)
plt.axvline(x=70, color="red", linestyle=":", label="Day 70: Dynamic Price Hike")
plt.title("Synthetic Control: Actual vs Counterfactual Revenue (Vadodara)")
plt.xlabel("Day")
plt.ylabel("Daily Revenue")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/synthetic_control_fit.png")
print("Graph saved at: outputs/synthetic_control_fit.png")