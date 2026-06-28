import os
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
csv_file = "yearly_band_means_raw.csv"
df = pd.read_csv(csv_file)

# Remove accidental index column if present
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# ------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------
out_dir = "band_timeseries_csv_plots_raw"
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------
# Plot each band
# ------------------------------------------------------------------
for band in sorted(df["band"].unique()):

    band_df = df[df["band"] == band].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot observed
    obs = band_df[band_df["series"] == "Observed"].sort_values("year")

    #if band == 'total_precipitation_sum':
    #    obs["value"] = obs["value"] / 1000000

    if not obs.empty:
        ax.plot(
            obs["year"],
            obs["value"],
            color="black",
            linewidth=3,
            label="Observed"
        )

    # Plot future scenarios
    future_series = [
        s for s in band_df["series"].unique()
        if s != "Observed"
    ]

    for scenario in sorted(future_series):
        scen = (
            band_df[band_df["series"] == scenario]
            .sort_values("year")
        )

        ax.plot(
            scen["year"],
            scen["value"],
            linewidth=1.5,
            alpha=0.8,
            label=scenario
        )

    ax.set_title(f"{band}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Value")
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8
    )

    plt.tight_layout()

    outfile = os.path.join(out_dir, f"{band}_annual_mean_raw.png")
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Plots saved to: {out_dir}")