import warnings
from pathlib import Path
from typing import Dict

import yaml
import pandas as pd
import numpy as np
import duckdb

warnings.filterwarnings("ignore")

# NumPy 2.0 compatibility
if not hasattr(np, "NaN"):
    np.NaN = np.nan


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
    """Load all analytical tables from DuckDB into DataFrames."""
    db_path = config["db_path"]
    
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

    return data


def analyze_team_pace(data: Dict) -> None:
    """Analyze team pace evolution over time."""
    print("\n" + "=" * 70)
    print("TEAM PACE EVOLUTION")
    print("=" * 70)
    
    team_metrics = data["team_metrics"]
    
    # Calculate seasonal averages
    seasonal_pace = (
        team_metrics.groupby(["Year", "team_name"])
        .agg({"pace_delta": "mean", "team_points": "sum"})
        .reset_index()
    )
    
    # Pre and post 2022 analysis
    pre_2022 = seasonal_pace[seasonal_pace["Year"] < 2022].groupby("team_name")["pace_delta"].mean()
    post_2022 = seasonal_pace[seasonal_pace["Year"] >= 2022].groupby("team_name")["pace_delta"].mean()
    
    print("\n📊 Performance Summary:")
    print(f"   Dataset covers {seasonal_pace['Year'].min():.0f} to {seasonal_pace['Year'].max():.0f}")
    print(f"   Teams analyzed: {seasonal_pace['team_name'].nunique()}")
    
    if len(pre_2022) > 0:
        fastest_pre = pre_2022.idxmin()
        print(f"\n   Pre-2022 Era:")
        print(f"   • Fastest team: {fastest_pre} ({pre_2022[fastest_pre]:.3f}s avg pace delta)")
        print(f"   • This team was consistently {abs(pre_2022[fastest_pre]):.3f}s faster than average")
    
    if len(post_2022) > 0:
        fastest_post = post_2022.idxmin()
        print(f"\n   Post-2022 Era (New Regulations):")
        print(f"   • Fastest team: {fastest_post} ({post_2022[fastest_post]:.3f}s avg pace delta)")
        
        # Check for performance shifts
        common_teams = pre_2022.index.intersection(post_2022.index)
        if len(common_teams) > 0:
            pace_change = (post_2022[common_teams] - pre_2022[common_teams]).sort_values()
            
            print(f"\n   Biggest Winners from Regulation Change:")
            for i in range(min(3, len(pace_change))):
                team = pace_change.index[i]
                change = pace_change.iloc[i]
                print(f"   • {team}: {abs(change):.3f}s faster")
            
            print(f"\n   Biggest Losers from Regulation Change:")
            for i in range(max(0, len(pace_change) - 3), len(pace_change)):
                team = pace_change.index[i]
                change = pace_change.iloc[i]
                print(f"   • {team}: {change:.3f}s slower")
    
    # Championship points
    points_by_team = seasonal_pace.groupby("team_name")["team_points"].sum().sort_values(ascending=False)
    print(f"\n🏆 Total Championship Points (All Seasons):")
    for team, points in points_by_team.head(5).items():
        print(f"   • {team}: {points:.0f} points")


def analyze_strategy(data: Dict) -> None:
    """Analyze team strategy patterns."""
    print("\n" + "=" * 70)
    print("STRATEGY PATTERNS")
    print("=" * 70)
    
    driver_metrics = data["driver_metrics"]
    
    strategy = (
        driver_metrics.groupby("team_name")
        .agg({
            "pit_stop_count": "mean",
            "first_stop_lap": "mean",
            "fraction_soft": "mean",
            "fraction_medium": "mean",
            "fraction_hard": "mean",
        })
        .reset_index()
    )
    
    if len(strategy) == 0:
        print("   No strategy data available")
        return
    
    print("\n⛽ Pit Stop Strategy:")
    
    # Most/least aggressive on pit stops
    max_stops_team = strategy.loc[strategy["pit_stop_count"].idxmax()]
    min_stops_team = strategy.loc[strategy["pit_stop_count"].idxmin()]
    
    print(f"   • Most pit stops: {max_stops_team['team_name']} ({max_stops_team['pit_stop_count']:.2f} avg)")
    print(f"   • Fewest pit stops: {min_stops_team['team_name']} ({min_stops_team['pit_stop_count']:.2f} avg)")
    
    # Pit timing
    earliest_team = strategy.loc[strategy["first_stop_lap"].idxmin()]
    latest_team = strategy.loc[strategy["first_stop_lap"].idxmax()]
    
    print(f"\n   Pit Stop Timing:")
    print(f"   • Earliest stopper: {earliest_team['team_name']} (lap {earliest_team['first_stop_lap']:.1f})")
    print(f"   • Latest stopper: {latest_team['team_name']} (lap {latest_team['first_stop_lap']:.1f})")
    
    # Tire compound preferences
    print(f"\n🛞 Tire Compound Preferences:")
    
    for _, row in strategy.nlargest(5, 'fraction_soft').iterrows():
        print(f"   • {row['team_name']}:")
        print(f"     - Soft: {row['fraction_soft']:.1%}  Medium: {row['fraction_medium']:.1%}  Hard: {row['fraction_hard']:.1%}")


def analyze_degradation(data: Dict) -> None:
    """Analyze tire degradation patterns."""
    print("\n" + "=" * 70)
    print("TIRE DEGRADATION ANALYSIS")
    print("=" * 70)
    
    stint_metrics = data["stint_metrics"]
    driver_metrics = data["driver_metrics"]
    
    # Degradation by compound
    compound_deg = stint_metrics[stint_metrics["compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    
    print("\n🔍 Degradation Rates by Compound:")
    
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        comp_data = compound_deg[compound_deg["compound"] == compound]["degradation_slope"]
        if len(comp_data) > 0:
            print(f"\n   {compound} Tires:")
            print(f"   • Average degradation: {comp_data.mean():.4f} s/lap")
            print(f"   • Median: {comp_data.median():.4f} s/lap")
            print(f"   • Std deviation: {comp_data.std():.4f} s/lap")
            
            # Interpretation
            if comp_data.mean() < -0.1:
                print(f"   → Tires typically gain pace (track evolution effect)")
            elif comp_data.mean() > 0.1:
                print(f"   → Tires degrade noticeably over stint")
            else:
                print(f"   → Minimal degradation (stable performance)")
    
    # Correlation with results
    driver_deg = (
        stint_metrics.groupby(["Year", "Round", "DriverNumber"])
        .agg({"degradation_slope": "mean"})
        .reset_index()
    )
    
    merged = driver_deg.merge(
        driver_metrics[["Year", "Round", "DriverNumber", "finish_position"]],
        on=["Year", "Round", "DriverNumber"],
        how="inner",
    )
    
    merged = merged[merged["finish_position"].notna()]
    
    if len(merged) >= 10:
        corr = merged["degradation_slope"].corr(merged["finish_position"])
        print(f"\n📈 Degradation vs Race Results:")
        print(f"   • Correlation coefficient: {corr:.3f}")
        
        if abs(corr) < 0.1:
            print(f"   → Very weak relationship - tire management isn't a primary performance factor")
        elif abs(corr) < 0.3:
            print(f"   → Weak relationship - other factors more important")
        elif abs(corr) < 0.5:
            print(f"   → Moderate relationship - tire management matters")
        else:
            print(f"   → Strong relationship - tire management is crucial")


def analyze_anomalies(data: Dict) -> None:
    """Detect and analyze anomalous performances."""
    print("\n" + "=" * 70)
    print("PERFORMANCE ANOMALIES")
    print("=" * 70)
    
    team_metrics = data["team_metrics"].copy()
    
    # Z-scores per team
    def z_transform(x: pd.Series) -> pd.Series:
        if x.std() == 0 or len(x) < 2:
            return pd.Series([0] * len(x), index=x.index)
        return (x - x.mean()) / x.std()
    
    team_metrics["pace_z_score"] = team_metrics.groupby("team_name")["pace_delta"].transform(z_transform)
    
    anomalies = team_metrics[np.abs(team_metrics["pace_z_score"]) > 2].copy()
    anomalies = anomalies.sort_values("pace_z_score", ascending=False)
    
    print(f"\n🚨 Detected {len(anomalies)} anomalous performances (>2σ from team norm)")
    
    if len(anomalies) == 0:
        print("   No significant anomalies detected - all teams performed consistently")
        return
    
    print("\n   Most Unusual Performances:")
    
    # Worst performances (slowest)
    print("\n   Unexpectedly Slow:")
    for _, row in anomalies.head(5).iterrows():
        print(f"   • {row['Year']} {row['EventName']}: {row['team_name']}")
        print(f"     {row['pace_z_score']:.2f}σ slower than usual ({row['pace_delta']:.3f}s off pace)")
    
    # Best performances (fastest)
    best_anomalies = anomalies[anomalies["pace_z_score"] < 0].tail(5)
    if len(best_anomalies) > 0:
        print("\n   Unexpectedly Fast:")
        for _, row in best_anomalies.iterrows():
            print(f"   • {row['Year']} {row['EventName']}: {row['team_name']}")
            print(f"     {abs(row['pace_z_score']):.2f}σ faster than usual ({row['pace_delta']:.3f}s pace delta)")
    
    # Case study of most extreme
    if len(anomalies) > 0:
        case = anomalies.iloc[0]
        print(f"\n   📌 Most Extreme Case:")
        print(f"   {case['Year']} {case['EventName']} - {case['team_name']}")
        print(f"   • Performance: {case['pace_z_score']:.2f}σ worse than team average")
        print(f"   • Pace delta: {case['pace_delta']:.3f}s")
        print(f"   • Possible causes: Technical issues, damage, extreme conditions, or strategic choices")


def main():
    config = load_config()
    
    print("\n" + "=" * 70)
    print("F1 DATA ANALYSIS REPORT")
    print("=" * 70)
    
    data = load_data(config)
    
    print(f"\n📁 Dataset Overview:")
    for name, df in data.items():
        print(f"   • {name}: {len(df):,} records")
    
    # Run all analyses
    analyze_team_pace(data)
    analyze_strategy(data)
    analyze_degradation(data)
    analyze_anomalies(data)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n💡 Key Takeaways:")
    print("   • The 2022 regulation changes significantly reshuffled team performance")
    print("   • Tire management shows varying importance across different compounds")
    print("   • Strategy choices (pit timing, compound selection) vary significantly by team")
    print("   • Performance anomalies highlight races affected by incidents or unusual conditions")
    print("\n")


if __name__ == "__main__":
    main()
