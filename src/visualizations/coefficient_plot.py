import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read coefficients
model_type = "complexity"
experiment_name = "linear-complexity/v5/"
path = f"models/experiments/{model_type}/{experiment_name}"
df = pl.read_csv(f"{path}/coefficients.csv")

# Convert to pandas for easier plotting
df_pd = df.to_pandas()

# Sort by absolute coefficient value
df_pd["abs_coefficient"] = abs(df_pd["coefficient"])
df_pd = df_pd.sort_values("abs_coefficient", ascending=True)

# Get top 20 positive and negative coefficients
top_pos = df_pd[df_pd["coefficient"] > 0].tail(20)
top_neg = df_pd[df_pd["coefficient"] < 0].tail(20)

# Combine and sort for visualization
plot_data = pd.concat([top_neg, top_pos])
plot_data = plot_data.sort_values("coefficient")

# Create color palette
colors = ["#FF6B6B" if c < 0 else "#4ECDC4" for c in plot_data["coefficient"]]

# Set up the plot style
plt.style.use("seaborn-v0_8-darkgrid")
plt.figure(figsize=(12, 10))
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14

# Create horizontal bar plot
bars = plt.barh(range(len(plot_data)), plot_data["coefficient"], color=colors)

# Customize the plot
plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)
plt.title("Top 20 Most Influential Features for Predicting Game Success", pad=20)
plt.xlabel(
    "Coefficient Value\n(Negative = Less Likely to Reach 25 Ratings, Positive = More Likely)",
    labelpad=10,
)

# Clean up feature names for display
feature_names = plot_data["feature"].str.replace("_", " ").str.title()
plt.yticks(range(len(plot_data)), feature_names)

# Add value labels on the bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    label_x = width + 0.01 if width >= 0 else width - 0.01
    ha = "left" if width >= 0 else "right"
    plt.text(label_x, i, f"{width:.3f}", va="center", ha=ha, color="black")

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig(f"{path}/feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
