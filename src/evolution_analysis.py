import warnings
from pathlib import Path
from typing import Dict

import yaml
import pandas as pd
import numpy as np
import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Team colors for consistency
TEAM_COLORS = {
    "Mercedes": "#00D2BE",
    "Red Bull Racing": "#0600EF",
    "Ferrari": "#DC0000",
    "McLaren": "#FF8700",
    "Alpine": "#0090FF",
    "Aston Martin": "#006F62",
}


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for key in ["cache_dir", "db_path", "figures_dir", "log_dir"]:
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    return cfg


def load_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """Load analytical tables from DuckDB."""
    db_path = config["db_path"]
    print(f"Loading data from {db_path}")

    con = duckdb.connect(db_path, read_only=True)
    try:
        data = {
            "stint_metrics": con.execute("SELECT * FROM stint_metrics").df(),
            "clean_laps": con.execute("SELECT * FROM clean_laps").df(),
            "driver_metrics": con.execute("SELECT * FROM driver_race_metrics").df(),
        }
    finally:
        con.close()

    print("Loaded tables:")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows")

    return data


def plot_track_evolution(data: Dict, config: Dict) -> None:
    """
    Analyze and visualize track evolution - how the track gets faster
    throughout the race as rubber is laid down.
    """
    print("\n" + "=" * 60)
    print("TRACK EVOLUTION ANALYSIS")
    print("=" * 60)

    stint_metrics = data["stint_metrics"].copy()
    clean_laps = data["clean_laps"].copy()

    # Filter to only valid laps with lap times
    clean_laps = clean_laps[clean_laps["LapTime_s"].notna()].copy()
    
    # Calculate stint start lap for each driver/race/stint combination
    # This is the minimum lap number for each stint
    stint_starts = (
        clean_laps.groupby(["Year", "Round", "DriverNumber", "Stint"])["LapNumber"]
        .min()
        .reset_index()
        .rename(columns={"LapNumber": "stint_start_lap"})
    )
    
    # Merge stint info with laps
    laps_with_stint = clean_laps.merge(
        stint_starts,
        on=["Year", "Round", "DriverNumber", "Stint"],
        how="left"
    )
    
    # Also merge compound from stint_metrics
    stint_compound = stint_metrics[["Year", "Round", "DriverNumber", "Stint", "compound"]].copy()
    laps_with_stint = laps_with_stint.merge(
        stint_compound,
        on=["Year", "Round", "DriverNumber", "Stint"],
        how="left",
        suffixes=("", "_stint")
    )

    # Calculate lap age within stint
    laps_with_stint["lap_age"] = (
        laps_with_stint["LapNumber"] - laps_with_stint["stint_start_lap"]
    )

    # Filter to only fresh tires (first 3 laps of stint) to control for tire age
    fresh_laps = laps_with_stint[
        (laps_with_stint["lap_age"] >= 0) & 
        (laps_with_stint["lap_age"] <= 2)
    ].copy()

    print(f"Analyzing {len(fresh_laps):,} fresh tire laps across {len(fresh_laps['EventName'].unique())} events")

    # =============================================================================
    # ANALYSIS 1: Track Evolution by Circuit
    # =============================================================================
    print("\nAnalyzing track evolution by circuit...")

    evolution_by_circuit = []

    for event in fresh_laps["EventName"].unique():
        event_laps = fresh_laps[fresh_laps["EventName"] == event]
        
        # Early race (first 15 laps)
        early = event_laps[event_laps["LapNumber"] <= 15]
        # Late race (lap 40+)
        late = event_laps[event_laps["LapNumber"] >= 40]
        
        if len(early) > 5 and len(late) > 5:  # Need enough data
            early_time = early["LapTime_s"].median()
            late_time = late["LapTime_s"].median()
            
            # Extract year and event name
            year = event_laps["Year"].iloc[0]
            
            evolution_by_circuit.append({
                "Event": event,
                "Year": int(year),
                "Early_Time": early_time,
                "Late_Time": late_time,
                "Evolution_Seconds": early_time - late_time,
                "Evolution_Percent": ((early_time - late_time) / early_time) * 100,
                "Sample_Size": len(early) + len(late)
            })

    evolution_df = pd.DataFrame(evolution_by_circuit)
    
    if len(evolution_df) == 0:
        print("Insufficient data for track evolution analysis.")
        return

    evolution_df = evolution_df.sort_values("Evolution_Seconds", ascending=False)

    # =============================================================================
    # ANALYSIS 2: Evolution Throughout Race
    # =============================================================================
    print("Analyzing pace progression throughout races...")

    # Bin laps into race phases
    fresh_laps["race_phase"] = pd.cut(
        fresh_laps["LapNumber"],
        bins=[0, 15, 30, 45, 100],
        labels=["Early (1-15)", "Mid-Early (16-30)", "Mid-Late (31-45)", "Late (46+)"]
    )

    phase_evolution = (
        fresh_laps.groupby("race_phase")["LapTime_s"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )

    # =============================================================================
    # ANALYSIS 3: Team Exploitation of Track Evolution
    # =============================================================================
    print("Analyzing which teams exploit track evolution best...")

    driver_metrics = data["driver_metrics"].copy()
    
    # Merge team names into fresh_laps
    team_map = driver_metrics[["Year", "Round", "DriverNumber", "team_name"]].drop_duplicates()
    fresh_laps_teams = fresh_laps.merge(
        team_map,
        on=["Year", "Round", "DriverNumber"],
        how="left"
    )

    team_evolution = []
    for team in fresh_laps_teams["team_name"].dropna().unique():
        team_laps = fresh_laps_teams[fresh_laps_teams["team_name"] == team]
        
        early = team_laps[team_laps["LapNumber"] <= 20]
        late = team_laps[team_laps["LapNumber"] >= 40]
        
        if len(early) > 10 and len(late) > 10:
            early_time = early["LapTime_s"].median()
            late_time = late["LapTime_s"].median()
            
            team_evolution.append({
                "Team": team,
                "Early_Pace": early_time,
                "Late_Pace": late_time,
                "Evolution_Gain": early_time - late_time,
                "Improvement_Pct": ((early_time - late_time) / early_time) * 100
            })

    team_evolution_df = pd.DataFrame(team_evolution)
    team_evolution_df = team_evolution_df.sort_values("Evolution_Gain", ascending=False)

    # =============================================================================
    # VISUALIZATION
    # =============================================================================

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Track Evolution by Circuit (Top 15)
    ax1 = fig.add_subplot(gs[0, :])
    top_circuits = evolution_df.head(15)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_circuits)))
    bars = ax1.barh(
        range(len(top_circuits)),
        top_circuits["Evolution_Seconds"],
        color=colors,
        edgecolor="black",
        linewidth=1.5
    )
    
    ax1.set_yticks(range(len(top_circuits)))
    ax1.set_yticklabels([f"{row['Event']} '{int(row['Year']) % 100}" 
                          for _, row in top_circuits.iterrows()])
    ax1.set_xlabel("Track Evolution (seconds faster from early to late race)", 
                   fontweight="bold", fontsize=11)
    ax1.set_title(
        "Track Evolution by Circuit\n"
        "How much faster is the track in late race vs early race? (Fresh tire comparison)",
        fontweight="bold",
        fontsize=13,
        pad=15
    )
    ax1.axvline(x=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax1.grid(axis="x", alpha=0.3)
    
    # Add values on bars
    for i, (_, row) in enumerate(top_circuits.iterrows()):
        ax1.text(
            row["Evolution_Seconds"] + 0.05,
            i,
            f"{row['Evolution_Seconds']:.2f}s ({row['Evolution_Percent']:.1f}%)",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    # Plot 2: Race Phase Evolution
    ax2 = fig.add_subplot(gs[1, 0])
    
    phases = phase_evolution["race_phase"].astype(str)
    means = phase_evolution["mean"]
    
    ax2.plot(phases, means, marker="o", linewidth=3, markersize=12, 
             color="#FF6B6B", label="Mean Lap Time")
    ax2.fill_between(range(len(phases)), means, alpha=0.3, color="#FF6B6B")
    
    ax2.set_ylabel("Lap Time (seconds)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Race Phase", fontweight="bold", fontsize=11)
    ax2.set_title("Pace Improvement Throughout Race", fontweight="bold", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add percentage improvement annotation
    if len(means) >= 2:
        first_time = means.iloc[0]
        last_time = means.iloc[-1]
        improvement = ((first_time - last_time) / first_time) * 100
        ax2.annotate(
            f"Overall: {improvement:.1f}% faster",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            ha="center",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7)
        )

    # Plot 3: Team Evolution Exploitation
    ax3 = fig.add_subplot(gs[1, 1])
    
    if len(team_evolution_df) > 0:
        team_colors_list = [TEAM_COLORS.get(team, "gray") 
                           for team in team_evolution_df["Team"]]
        
        bars = ax3.barh(
            team_evolution_df["Team"],
            team_evolution_df["Evolution_Gain"],
            color=team_colors_list,
            edgecolor="black",
            linewidth=1.5
        )
        
        ax3.set_xlabel("Evolution Gain (seconds)", fontweight="bold", fontsize=11)
        ax3.set_title("Team Track Evolution Exploitation", fontweight="bold", fontsize=12)
        ax3.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax3.grid(axis="x", alpha=0.3)
        
        # Add percentage labels
        for i, (_, row) in enumerate(team_evolution_df.iterrows()):
            ax3.text(
                row["Evolution_Gain"] + 0.05,
                i,
                f"{row['Improvement_Pct']:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold"
            )

    # Plot 4: Scatter - Early vs Late Pace by Circuit
    ax4 = fig.add_subplot(gs[2, 0])
    
    ax4.scatter(
        evolution_df["Early_Time"],
        evolution_df["Late_Time"],
        s=evolution_df["Sample_Size"] / 2,
        alpha=0.6,
        c=evolution_df["Evolution_Seconds"],
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=1
    )
    
    # Diagonal line (no evolution)
    min_time = min(evolution_df["Early_Time"].min(), evolution_df["Late_Time"].min())
    max_time = max(evolution_df["Early_Time"].max(), evolution_df["Late_Time"].max())
    ax4.plot([min_time, max_time], [min_time, max_time], 
             "k--", alpha=0.5, linewidth=2, label="No Evolution")
    
    ax4.set_xlabel("Early Race Pace (seconds)", fontweight="bold", fontsize=11)
    ax4.set_ylabel("Late Race Pace (seconds)", fontweight="bold", fontsize=11)
    ax4.set_title("Early vs Late Race Pace\n(Points below line = track got faster)", 
                  fontweight="bold", fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn",
        norm=plt.Normalize(
            vmin=evolution_df["Evolution_Seconds"].min(),
            vmax=evolution_df["Evolution_Seconds"].max()
        )
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax4)
    cbar.set_label("Evolution (seconds)", fontweight="bold")

    # Plot 5: Distribution of Evolution
    ax5 = fig.add_subplot(gs[2, 1])
    
    ax5.hist(
        evolution_df["Evolution_Seconds"],
        bins=20,
        color="#4ECDC4",
        edgecolor="black",
        alpha=0.7
    )
    ax5.axvline(
        evolution_df["Evolution_Seconds"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {evolution_df['Evolution_Seconds'].median():.2f}s"
    )
    ax5.set_xlabel("Track Evolution (seconds)", fontweight="bold", fontsize=11)
    ax5.set_ylabel("Number of Races", fontweight="bold", fontsize=11)
    ax5.set_title("Distribution of Track Evolution", fontweight="bold", fontsize=12)
    ax5.legend()
    ax5.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "F1 Track Evolution Analysis: How Circuits Get Faster During Races",
        fontsize=16,
        fontweight="bold",
        y=0.995
    )

    # Save
    figures_dir = Path(config["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "track_evolution_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {output_path}")

    # Print insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    if len(evolution_df) > 0:
        print(f"\n📊 Circuit Analysis ({len(evolution_df)} races):")
        print(f"  • Biggest evolution: {evolution_df.iloc[0]['Event']} "
              f"({evolution_df.iloc[0]['Evolution_Seconds']:.2f}s / "
              f"{evolution_df.iloc[0]['Evolution_Percent']:.1f}%)")
        print(f"  • Smallest evolution: {evolution_df.iloc[-1]['Event']} "
              f"({evolution_df.iloc[-1]['Evolution_Seconds']:.2f}s / "
              f"{evolution_df.iloc[-1]['Evolution_Percent']:.1f}%)")
        print(f"  • Average evolution: {evolution_df['Evolution_Seconds'].mean():.2f}s "
              f"({evolution_df['Evolution_Percent'].mean():.1f}%)")
        print(f"  • Median evolution: {evolution_df['Evolution_Seconds'].median():.2f}s")

    if len(phase_evolution) > 0:
        print(f"\n⏱️ Race Phase Progression:")
        for _, row in phase_evolution.iterrows():
            print(f"  • {row['race_phase']}: {row['mean']:.2f}s avg "
                  f"({int(row['count'])} laps)")

    if len(team_evolution_df) > 0:
        print(f"\n🏎️ Team Evolution Exploitation:")
        print(f"  • Best: {team_evolution_df.iloc[0]['Team']} "
              f"({team_evolution_df.iloc[0]['Evolution_Gain']:.2f}s / "
              f"{team_evolution_df.iloc[0]['Improvement_Pct']:.1f}%)")
        print(f"  • Worst: {team_evolution_df.iloc[-1]['Team']} "
              f"({team_evolution_df.iloc[-1]['Evolution_Gain']:.2f}s / "
              f"{team_evolution_df.iloc[-1]['Improvement_Pct']:.1f}%)")

    print("\n💡 What This Means:")
    print("  • Qualifying laps are set on a 'green' track with less rubber")
    print("  • By race end, the track can be 1-3 seconds faster")
    print("  • Late pit stops get fresh tires + faster track = double advantage")
    print("  • High-degradation circuits show more evolution (more rubber laid)")

    # plt.show()  # Uncomment if you want interactive display


def main():
    config = load_config()
    
    print("=" * 60)
    print("F1 TRACK EVOLUTION ANALYSIS")
    print("=" * 60)
    
    data = load_data(config)
    plot_track_evolution(data, config)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()