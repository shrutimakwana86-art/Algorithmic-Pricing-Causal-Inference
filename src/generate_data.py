import os
import numpy as np
import pandas as pd
# Check or create the data directory to save the dataset
os.makedirs("data", exist_ok=True)
np.random.seed(42)
days = np.arange(1, 101)
cities = ["Vadodara", "Ahmedabad", "Surat", "Rajkot"]
records = []
for day in days:
    # General market demand and weekend seasonality effect (Confounder)
    base_demand = 100 + 15 * np.sin(day / 7.0) + np.random.normal(0, 3)
    competitor_price = 50 + 0.05 * day + np.random.normal(0, 1.5)
    for city in cities:
        # Regional baseline scale for each city
        city_multiplier = {
            "Vadodara": 1.0,
            "Ahmedabad": 1.4,
            "Surat": 1.25,
            "Rajkot": 0.9
        }[city]
        # 15% price hike applied only in Vadodara starting from Day 71 (Treatment)
        treatment = 1 if (city == "Vadodara" and day > 70) else 0
        our_price = 52.0 if treatment == 0 else 60.0
        # Calculate revenue (true +12% causal lift in Vadodara post-treatment)
        lift = 1.12 if treatment == 1 else 1.0
        revenue = (
            (base_demand * city_multiplier * 25)
            + (competitor_price * 10)
            - (our_price * 8)
            + np.random.normal(0, 40)
        ) * lift
        period = "Pre-Treatment" if day <= 70 else "Post-Treatment"
        records.append({
            "day": day,
            "city": city,
            "period": period,
            "treatment": treatment,
            "our_price": round(our_price, 2),
            "competitor_price": round(competitor_price, 2),
            "demand_index": round(base_demand, 2),
            "revenue": round(revenue, 2)
        })
df = pd.DataFrame(records)
df.to_csv("data/regional_pricing_data.csv", index=False)
print("Dataset successfully created at: data/regional_pricing_data.csv")
print(df.head())