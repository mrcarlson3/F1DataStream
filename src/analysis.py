import warnings
from pathlib import Path
from typing import Dict

import yaml
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility: Add np.NaN as alias for np.nan
if not hasattr(np, "NaN"):
    np.NaN = np.nan

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Simple color palette for key teams
TEAM_COLORS = {
    "Mercedes": "#00D2BE",
    "Red Bull Racing": "#0600EF",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Alpine": "#0090FF",
    "Aston Martin": "#006F62",
}


# =============================================================================
# CONFIG & DATA
# =============================================================================

def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Normalize paths
    for key in ["cache_dir", "db_path", "figures_dir", "log_dir"]:
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    return cfg


def load_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """Load all analytical tables from DuckDB into DataFrames."""
    db_path = config["db_path"]
    print(f"Loading data from {db_path}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        data = {
            "team_metrics": con.execute("SELECT * FROM team_race_metrics").df(),
            "driver_metrics": con.execute("SELECT * FROM driver_race_metrics").df(),
            "stint_metrics": con.execute("SELECT * FROM stint_metrics").df(),
            "clean_results": con.execute("SELECT * FROM clean_results").df(),
        }
    finally:
        con.close()

    print("Loaded tables:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows")

    return data


# =============================================================================
# ANALYSIS 1: TEAM PACE EVOLUTION
# =============================================================================

def plot_team_pace_evolution(data: Dict, config: Dict) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Team Pace Evolution")
    print("=" * 60)

    team_metrics = data["team_metrics"]

    # Calculate seasonal averages
    seasonal_pace = (
        team_metrics.groupby(["Year", "team_name"])
        .agg({"pace_delta": "mean", "team_points": "sum"})
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Pace delta over time
    for team in seasonal_pace["team_name"].unique():
        team_data = seasonal_pace[seasonal_pace["team_name"] == team]
        color = TEAM_COLORS.get(team, None)
        ax1.plot(
            team_data["Year"],
            team_data["pace_delta"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=team,
            color=color,
        )

    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax1.axvline(x=2022, color="red", linestyle=":", alpha=0.5, linewidth=2, label="2022 Reg Change")
    ax1.set_xlabel("Season", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Pace Delta (s)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "F1 Team Race Pace Evolution\nRelative to Fastest Team per Race",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="best", frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Championship points over time
    for team in seasonal_pace["team_name"].unique():
        team_data = seasonal_pace[seasonal_pace["team_name"] == team]
        color = TEAM_COLORS.get(team, None)
        ax2.plot(
            team_data["Year"],
            team_data["team_points"],
            marker="s",
            linewidth=2.5,
            markersize=8,
            label=team,
            color=color,
        )

    ax2.axvline(x=2022, color="red", linestyle=":", alpha=0.5, linewidth=2)
    ax2.set_xlabel("Season", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Total Points", fontsize=12, fontweight="bold")
    ax2.set_title("Championship Points by Season", fontsize=14, fontweight="bold", pad=15)
    ax2.legend(loc="best", frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "pace_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # Simple textual insights
    print("\nKey Insights:")
    pre_2022 = seasonal_pace[seasonal_pace["Year"] < 2022].groupby("team_name")["pace_delta"].mean()
    post_2022 = seasonal_pace[seasonal_pace["Year"] >= 2022].groupby("team_name")["pace_delta"].mean()

    if len(pre_2022) > 0:
        print(f"  Pre-2022 fastest: {pre_2022.idxmin()} (Δ = {pre_2022.min():.3f}s)")
    if len(post_2022) > 0:
        print(f"  Post-2022 fastest: {post_2022.idxmin()} (Δ = {post_2022.min():.3f}s)")

        # Only compute change where teams exist in both eras
        common_teams = pre_2022.index.intersection(post_2022.index)
        if len(common_teams) > 0:
            pace_change = (post_2022[common_teams] - pre_2022[common_teams]).sort_values()
            print(f"  Biggest improvement: {pace_change.index[0]} ({pace_change.iloc[0]:.3f}s faster)")
            print(f"  Biggest decline: {pace_change.index[-1]} (+{pace_change.iloc[-1]:.3f}s slower)")

    plt.show()


# =============================================================================
# ANALYSIS 2: STRATEGY PATTERNS
# =============================================================================

def plot_strategy_patterns(data: Dict, config: Dict) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Strategy Patterns")
    print("=" * 60)

    driver_metrics = data["driver_metrics"]

    strategy = (
        driver_metrics.groupby("team_name")
        .agg(
            {
                "pit_stop_count": "mean",
                "first_stop_lap": "mean",
                "fraction_soft": "mean",
                "fraction_medium": "mean",
                "fraction_hard": "mean",
            }
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Average pit stops
    ax1 = axes[0, 0]
    ax1.bar(
        strategy["team_name"],
        strategy["pit_stop_count"],
        color=[TEAM_COLORS.get(t, "gray") for t in strategy["team_name"]],
    )
    ax1.set_ylabel("Average Pit Stops per Race", fontweight="bold")
    ax1.set_title("Pit Stop Frequency", fontweight="bold", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # First stop timing
    ax2 = axes[0, 1]
    ax2.bar(
        strategy["team_name"],
        strategy["first_stop_lap"],
        color=[TEAM_COLORS.get(t, "gray") for t in strategy["team_name"]],
    )
    ax2.set_ylabel("Average First Stop Lap", fontweight="bold")
    ax2.set_title("Pit Stop Timing", fontweight="bold", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(axis="y", alpha=0.3)

    # Compound usage
    ax3 = axes[1, 0]
    x = np.arange(len(strategy["team_name"]))
    width = 0.6
    p1 = ax3.bar(x, strategy["fraction_soft"], width, label="Soft", color="#FF4444")
    p2 = ax3.bar(
        x,
        strategy["fraction_medium"],
        width,
        bottom=strategy["fraction_soft"],
        label="Medium",
        color="#FFD700",
    )
    p3 = ax3.bar(
        x,
        strategy["fraction_hard"],
        width,
        bottom=strategy["fraction_soft"] + strategy["fraction_medium"],
        label="Hard",
        color="#FFFFFF",
        edgecolor="black",
    )

    ax3.set_ylabel("Proportion of Laps", fontweight="bold")
    ax3.set_title("Tyre Compound Usage", fontweight="bold", fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy["team_name"], rotation=45, ha="right")
    ax3.legend(loc="upper right")
    ax3.grid(axis="y", alpha=0.3)

    # Strategy profile scatter
    ax4 = axes[1, 1]
    for _, row in strategy.iterrows():
        team = row["team_name"]
        color = TEAM_COLORS.get(team, "gray")
        ax4.scatter(
            row["first_stop_lap"],
            row["pit_stop_count"],
            s=300,
            color=color,
            alpha=0.7,
            edgecolors="black",
            linewidth=2,
        )
        ax4.text(
            row["first_stop_lap"],
            row["pit_stop_count"],
            team.split()[0],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    ax4.set_xlabel("First Stop Lap", fontweight="bold")
    ax4.set_ylabel("Pit Stops per Race", fontweight="bold")
    ax4.set_title("Strategy Profile", fontweight="bold", fontsize=12)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("F1 Race Strategy Patterns by Team", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "strategy_patterns.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    print("\nKey Insights:")
    if len(strategy) > 0:
        print(
            f"  Most pit stops: "
            f"{strategy.loc[strategy['pit_stop_count'].idxmax(), 'team_name']} "
            f"({strategy['pit_stop_count'].max():.2f})"
        )
        print(
            f"  Earliest stopper: "
            f"{strategy.loc[strategy['first_stop_lap'].idxmin(), 'team_name']} "
            f"(lap {strategy['first_stop_lap'].min():.1f})"
        )
        print(
            f"  Most soft usage: "
            f"{strategy.loc[strategy['fraction_soft'].idxmax(), 'team_name']} "
            f"({strategy['fraction_soft'].max():.1%})"
        )

    plt.show()


# =============================================================================
# ANALYSIS 3: DEGRADATION VS RESULTS
# =============================================================================

def plot_degradation_analysis(data: Dict, config: Dict) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS 3: Degradation vs Results")
    print("=" * 60)

    stint_metrics = data["stint_metrics"]
    driver_metrics = data["driver_metrics"]

    # Average degradation per driver per race
    driver_deg = (
        stint_metrics.groupby(["Year", "Round", "DriverNumber"])
        .agg(
            {
                "degradation_slope": "mean",
                "stint_length_laps": "mean",
            }
        )
        .reset_index()
    )

    merged = driver_deg.merge(
        driver_metrics[
            ["Year", "Round", "DriverNumber", "team_name", "finish_position", "points"]
        ],
        on=["Year", "Round", "DriverNumber"],
        how="inner",
    )

    merged = merged[merged["finish_position"].notna()]
    if len(merged) < 10:
        print(f"Insufficient data for degradation analysis ({len(merged)} records). Skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Degradation vs finish position
    ax1 = axes[0]
    for team in merged["team_name"].unique():
        team_data = merged[merged["team_name"] == team]
        color = TEAM_COLORS.get(team, "gray")
        ax1.scatter(
            team_data["degradation_slope"],
            team_data["finish_position"],
            alpha=0.6,
            s=50,
            color=color,
            label=team,
            edgecolors="black",
            linewidth=0.5,
        )

    valid_data = merged[
        merged["degradation_slope"].notna() & merged["finish_position"].notna()
    ]
    if len(valid_data) >= 2:
        z = np.polyfit(
            valid_data["degradation_slope"], valid_data["finish_position"], 1
        )
        p = np.poly1d(z)
        x_trend = np.linspace(
            valid_data["degradation_slope"].min(),
            valid_data["degradation_slope"].max(),
            100,
        )
        corr = np.corrcoef(
            valid_data["degradation_slope"], valid_data["finish_position"]
        )[0, 1]
        ax1.plot(
            x_trend,
            p(x_trend),
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"Trend (r={corr:.2f})",
        )

    ax1.set_xlabel("Average Degradation (s/lap)", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Finish Position", fontweight="bold", fontsize=11)
    ax1.set_title("Tyre Degradation vs Race Result", fontweight="bold", fontsize=12)
    ax1.invert_yaxis()
    ax1.legend(loc="best", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Degradation by compound
    ax2 = axes[1]
    compound_deg = stint_metrics[
        stint_metrics["compound"].isin(["SOFT", "MEDIUM", "HARD"])
    ].copy()

    compounds = ["SOFT", "MEDIUM", "HARD"]
    colors = ["#FF4444", "#FFD700", "#FFFFFF"]
    positions = [
        compound_deg[compound_deg["compound"] == c]["degradation_slope"].values
        for c in compounds
    ]

    bp = ax2.boxplot(
        positions,
        labels=compounds,
        patch_artist=True,
        showmeans=True,
        meanline=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.5)

    ax2.set_ylabel("Degradation Rate (s/lap)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Tyre Compound", fontweight="bold", fontsize=11)
    ax2.set_title("Degradation by Compound", fontweight="bold", fontsize=12)
    ax2.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Tyre Degradation Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "degradation_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    print("\nKey Insights:")
    for compound in compounds:
        comp_data = compound_deg[compound_deg["compound"] == compound][
            "degradation_slope"
        ]
        if len(comp_data) > 0:
            print(
                f"  {compound}: mean={comp_data.mean():.4f} s/lap, "
                f"median={comp_data.median():.4f} s/lap, "
                f"std={comp_data.std():.4f}"
            )
        else:
            print(f"  {compound}: no data available")

    if len(valid_data) >= 2:
        corr = merged["degradation_slope"].corr(merged["finish_position"])
        print(f"  Correlation (deg vs finish): {corr:.3f}")

    plt.show()


# =============================================================================
# ANALYSIS 4: ANOMALY DETECTION
# =============================================================================

def detect_and_plot_anomalies(data: Dict, config: Dict) -> None:
    print("\n" + "=" * 60)
    print("ANALYSIS 4: Anomaly Detection")
    print("=" * 60)

    team_metrics = data["team_metrics"].copy()

    # Z-scores per team
    def z_transform(x: pd.Series) -> pd.Series:
        if x.std() == 0 or len(x) < 2:
            return pd.Series([0] * len(x), index=x.index)
        return (x - x.mean()) / x.std()

    team_metrics["pace_z_score"] = team_metrics.groupby("team_name")["pace_delta"].transform(
        z_transform
    )

    anomalies = team_metrics[np.abs(team_metrics["pace_z_score"]) > 2].copy()
    anomalies = anomalies.sort_values("pace_z_score", ascending=False)

    print(f"Found {len(anomalies)} anomalous race performances")
    if len(anomalies) == 0:
        print("No anomalies detected. Skipping case study.")
        return

    # Show top 10 anomalies
    for _, row in anomalies.head(10).iterrows():
        direction = "slower" if row["pace_z_score"] > 0 else "faster"
        print(
            f"  {row['Year']} {row['EventName']}: {row['team_name']} "
            f"({row['pace_z_score']:.2f}σ {direction})"
        )

    # Case study: most extreme anomaly
    case_study = anomalies.iloc[0]
    print(
        f"\nCase Study: {case_study['Year']} {case_study['EventName']} - {case_study['team_name']}"
    )

    race_data = team_metrics[
        (team_metrics["Year"] == case_study["Year"])
        & (team_metrics["Round"] == case_study["Round"])
    ].copy()
    race_data = race_data.sort_values("team_avg_pace")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pace comparison
    ax1 = axes[0]
    colors = [TEAM_COLORS.get(t, "gray") for t in race_data["team_name"]]
    bars = ax1.barh(race_data["team_name"], race_data["pace_delta"], color=colors)

    anomaly_idx = race_data["team_name"].tolist().index(case_study["team_name"])
    bars[anomaly_idx].set_edgecolor("red")
    bars[anomaly_idx].set_linewidth(3)

    ax1.set_xlabel("Pace Delta (s)", fontweight="bold")
    ax1.set_title(
        f'{case_study["EventName"]} {case_study["Year"]}\nTeam Race Pace',
        fontweight="bold",
        fontsize=12,
    )
    ax1.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax1.grid(axis="x", alpha=0.3)

    # Strategy comparison
    ax2 = axes[1]
    x = np.arange(len(race_data))
    width = 0.35

    ax2.bar(x - width / 2, race_data["team_avg_pit_stops"], width, label="Pit Stops", alpha=0.7)
    ax2.bar(
        x + width / 2,
        race_data["team_avg_first_stop"] / 10,
        width,
        label="1st Stop Lap (/10)",
        alpha=0.7,
    )

    ax2.set_ylabel("Count", fontweight="bold")
    ax2.set_title("Strategy Profile", fontweight="bold", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(race_data["team_name"], rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "anomaly_case_study.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = load_config()
    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("F1 DATA ANALYSIS")
    print("=" * 60)

    data = load_data(config)

    plot_team_pace_evolution(data, config)
    plot_strategy_patterns(data, config)
    plot_degradation_analysis(data, config)
    detect_and_plot_anomalies(data, config)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {figures_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
