import xgboost as xgb
import matplotlib.pyplot as plt

# Load saved model
model = xgb.Booster()
model.load_model("xgb_final_model_no_dc.json")

# Plot feature importances
fig, ax = plt.subplots(figsize=(10, 8))

xgb.plot_importance(
    model,
    ax=ax,
    importance_type="total_gain",   # "weight", "gain", "cover", "total_gain", "total_cover"
    show_values=False
)

# Remove y-axis title
ax.set_ylabel("")

# Add faint vertical gridlines
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

# Make bars slightly wider (taller, since they're horizontal)
for bar in ax.patches:
    current_height = bar.get_height()
    new_height = current_height * 1.25  # Increase by 25%
    bar.set_y(bar.get_y() - (new_height - current_height) / 2)
    bar.set_height(new_height)

ax.set_title("XGBoost Feature Importances no DC")
plt.tight_layout()
plt.savefig('./final_plots/feature_importances_no_dc.png', dpi=300)
plt.show()