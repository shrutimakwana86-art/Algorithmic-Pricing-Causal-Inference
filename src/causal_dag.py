import pandas as pd
from dowhy import CausalModel
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. Load dataset and validate Causal Model with DoWhy
df = pd.read_csv("data/regional_pricing_data.csv")

causal_graph = """
digraph {
    demand_index -> treatment;
    demand_index -> revenue;
    competitor_price -> treatment;
    competitor_price -> revenue;
    treatment -> our_price;
    treatment -> revenue;
    our_price -> revenue;
}
"""

model = CausalModel(
    data=df,
    treatment="treatment",
    outcome="revenue",
    graph=causal_graph
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
print(identified_estimand)

# 2. Generate exact Color-Coded Causal DAG Diagram
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Title
plt.title("Causal Directed Acyclic Graph (DAG): Algorithmic Pricing Impact", fontsize=13, fontweight="bold", pad=12)

# Node coordinates (x, y) and styling matching your document figure
nodes = {
    "demand_index": {
        "pos": (2.2, 5.7),
        "label": "Market Demand\n(demand_index)",
        "color": "#C85A32"  # Confounder (Rust/Orange)
    },
    "competitor_price": {
        "pos": (6.8, 5.2),
        "label": "Competitor Price\n(competitor_price)",
        "color": "#C85A32"  # Confounder (Rust/Orange)
    },
    "treatment": {
        "pos": (1.2, 3.7),
        "label": "Dynamic Pricing\n(treatment)",
        "color": "#2B7BBA"  # Treatment (Blue)
    },
    "our_price": {
        "pos": (5.0, 2.1),
        "label": "Product Price\n(our_price)",
        "color": "#6A5ACD"  # Mediator (Purple)
    },
    "revenue": {
        "pos": (8.6, 3.3),
        "label": "Daily Revenue\n(revenue)",
        "color": "#3B8E5B"  # Outcome (Green)
    }
}

# Draw directed edges (arrows)
edges = [
    ("demand_index", "treatment", (1.9, 5.3), (1.3, 4.15)),
    ("demand_index", "revenue", (2.8, 5.5), (8.1, 3.6)),
    ("competitor_price", "treatment", (6.1, 5.1), (1.8, 3.95)),
    ("competitor_price", "revenue", (7.1, 4.8), (8.4, 3.75)),
    ("treatment", "our_price", (1.6, 3.3), (4.4, 2.3)),
    ("treatment", "revenue", (1.9, 3.65), (7.95, 3.35)),
    ("our_price", "revenue", (5.6, 2.25), (8.0, 3.05))
]

for start_node, end_node, start_xy, end_xy in edges:
    is_main_effect = (start_node == "treatment" and end_node == "revenue")
    ax.annotate(
        "",
        xy=end_xy,
        xytext=start_xy,
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.7,head_width=0.4",
            color="#2B7BBA" if is_main_effect else "#555555",
            lw=2.4 if is_main_effect else 1.4,
            shrinkA=0,
            shrinkB=0
        ),
        zorder=2
    )

# Draw rectangular boxes
for key, info in nodes.items():
    x, y = info["pos"]
    ax.text(
        x, y, info["label"],
        ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.6", facecolor=info["color"], edgecolor="#333333", lw=1),
        zorder=3
    )

# Add Legend at the bottom
legend_elements = [
    patches.Patch(facecolor="#2B7BBA", label="Treatment (Intervention)"),
    patches.Patch(facecolor="#3B8E5B", label="Outcome (Target KPI)"),
    patches.Patch(facecolor="#C85A32", label="Confounders (Adjusted via DoWhy & DML)"),
    patches.Patch(facecolor="#6A5ACD", label="Mediator (Price Mechanism)")
]
ax.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False,
    fontsize=9
)

plt.tight_layout()
plt.savefig("outputs/causal_dag.png", dpi=300, bbox_inches="tight")
print("Updated Causal DAG saved to 'outputs/causal_dag.png'!")