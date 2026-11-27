import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dowhy import CausalModel

# Create outputs directory if it does not exist
os.makedirs("outputs", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("data/regional_pricing_data.csv")

# 2. Separate Pre-Treatment (Days 1-70) and Post-Treatment (Days 71-100) periods
pre_df = df[df["day"] <= 70].copy()
post_df = df[df["day"] > 70].copy()

pre_df.to_csv("data/pre_treatment.csv", index=False)
post_df.to_csv("data/post_treatment.csv", index=False)
print(f"Data split complete: Pre-treatment rows = {len(pre_df)}, Post-treatment rows = {len(post_df)}")

# 3. Construct Causal Directed Acyclic Graph (DAG) using DoWhy
causal_graph = """
digraph {
    demand_index -> treatment;
    demand_index -> revenue;
    competitor_price -> treatment;
    competitor_price -> revenue;
    treatment -> our_price;
    our_price -> revenue;
    treatment -> revenue;
}
"""

model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="revenue",
    graph=causal_graph
)

# 4. Identify causal estimand and confounders
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print("\n--- Identified Causal Estimand & Confounders ---")
print(identified_estimand)

# 5. Create a Clean, Non-Overlapping Enterprise Causal DAG
G = nx.DiGraph()
edges = [
    ("demand_index", "treatment"),
    ("demand_index", "revenue"),
    ("competitor_price", "treatment"),
    ("competitor_price", "revenue"),
    ("treatment", "our_price"),
    ("our_price", "revenue"),
    ("treatment", "revenue")
]
G.add_edges_from(edges)

# Exact coordinates for each node
pos = {
    "demand_index": (0.28, 0.82),
    "competitor_price": (0.72, 0.82),
    "treatment": (0.12, 0.45),
    "our_price": (0.50, 0.16),
    "revenue": (0.88, 0.45)
}

labels = {
    "demand_index": "Market Demand\n(demand_index)",
    "competitor_price": "Competitor Price\n(competitor_price)",
    "treatment": "Dynamic Pricing\n(treatment)",
    "our_price": "Product Price\n(our_price)",
    "revenue": "Daily Revenue\n(revenue)"
}

# Exact color mapping per node name
color_map = {
    "demand_index": "#E67E22",      # Orange (Confounder)
    "competitor_price": "#E67E22",  # Orange (Confounder)
    "treatment": "#2980B9",         # Blue (Treatment)
    "our_price": "#8E44AD",         # Purple (Mediator)
    "revenue": "#27AE60"            # Green (Outcome)
}

fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

# Draw directed arrows first (behind boxes)
for u, v in edges:
    is_main = (u == "treatment" and v == "revenue")
    ax.annotate(
        "",
        xy=pos[v], xytext=pos[u],
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.8,head_width=0.45",
            color="#2980B9" if is_main else "#566573",
            lw=2.8 if is_main else 1.8,
            shrinkA=42, shrinkB=42,
            connectionstyle="arc3,rad=0.06" if not is_main else "arc3,rad=0.0"
        ),
        zorder=1
    )

# Draw clean rounded boxes for each node so text never overflows
for node, (x, y) in pos.items():
    ax.text(
        x, y, labels[node],
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="white",
        bbox=dict(
            boxstyle="round,pad=0.75,rounding_size=0.2",
            facecolor=color_map[node],
            edgecolor="#1C2833",
            linewidth=1.5
        ),
        zorder=2
    )

# Place Legend cleanly BELOW the graph (no overlap)
legend_handles = [
    mpatches.Patch(color="#2980B9", label="Treatment (Intervention)"),
    mpatches.Patch(color="#27AE60", label="Outcome (Target KPI)"),
    mpatches.Patch(color="#E67E22", label="Confounders (Adjusted via DoWhy & DML)"),
    mpatches.Patch(color="#8E44AD", label="Mediator (Price Mechanism)")
]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    frameon=True,
    facecolor="white",
    edgecolor="#BDC3C7",
    fontsize=10
)

plt.title(
    "Causal Directed Acyclic Graph (DAG): Algorithmic Pricing Impact",
    fontsize=14,
    fontweight="bold",
    color="#1C2833",
    pad=15
)
plt.xlim(-0.02, 1.02)
plt.ylim(0.02, 0.98)
plt.axis("off")
plt.tight_layout()

plt.savefig("outputs/causal_dag.png", dpi=300, bbox_inches="tight")
print("Clean Causal DAG saved at: outputs/causal_dag.png")
plt.show()