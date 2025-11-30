#!/usr/bin/env python3
"""
F1 Data Analysis and Visualization

Generate insights and plots from the processed F1 data:
1. Team pace evolution across seasons
2. Strategy patterns by team
3. Degradation vs race results
4. Anomaly detection and case studies

Usage:
    python src/analysis.py
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

import yaml
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility: Add np.NaN as alias for np.nan
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure plotting
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Custom color palette for teams
TEAM_COLORS = {
    'Mercedes': '#00D2BE',
    'Red Bull Racing': '#0600EF',
    'Ferrari': '#DC0000',
    'McLaren': '#FF8700',
    'Alpine': '#0090FF',
    'Aston Martin': '#006F62'
}


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config() -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(config: Dict) -> logging.Logger:
    """Setup basic logging for analysis."""
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "analysis.log"  # Fixed filename, overwrite each run
    
    logger = logging.getLogger('f1_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler only (overwrite mode)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(config: Dict, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    """Load all analytical tables from DuckDB."""
    db_path = config['db_path']
    
    logger.info(f"Loading data from {db_path}")
    
    try:
        con = duckdb.connect(db_path, read_only=True)
        
        data = {
            'team_metrics': con.execute("SELECT * FROM team_race_metrics").df(),
            'driver_metrics': con.execute("SELECT * FROM driver_race_metrics").df(),
            'stint_metrics': con.execute("SELECT * FROM stint_metrics").df(),
            'clean_results': con.execute("SELECT * FROM clean_results").df()
        }
        
        con.close()
        
        logger.info(f"Loaded {len(data)} tables:")
        for name, df in data.items():
            logger.info(f"  {name}: {len(df):,} rows")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


# =============================================================================
# ANALYSIS 1: TEAM PACE EVOLUTION
# =============================================================================

def plot_team_pace_evolution(data: Dict, config: Dict, logger: logging.Logger):
    """Plot team pace evolution across seasons (main story)."""
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 1: Team Pace Evolution (2019-2023)")
    logger.info("="*60)
    
    team_metrics = data['team_metrics']
    
    # Calculate seasonal averages
    seasonal_pace = team_metrics.groupby(['Year', 'team_name']).agg({
        'pace_delta': 'mean',
        'team_points': 'sum'
    }).reset_index()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Pace delta over time
    for team in seasonal_pace['team_name'].unique():
        team_data = seasonal_pace[seasonal_pace['team_name'] == team]
        color = TEAM_COLORS.get(team, None)
        ax1.plot(team_data['Year'], team_data['pace_delta'], 
                marker='o', linewidth=2.5, markersize=8,
                label=team, color=color)
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axvline(x=2022, color='red', linestyle=':', alpha=0.5, linewidth=2, label='2022 Reg Change')
    
    ax1.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Pace Delta (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Team Race Pace Evolution (2019-2023)\nRelative to Fastest Team per Race', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Championship points over time
    for team in seasonal_pace['team_name'].unique():
        team_data = seasonal_pace[seasonal_pace['team_name'] == team]
        color = TEAM_COLORS.get(team, None)
        ax2.plot(team_data['Year'], team_data['team_points'], 
                marker='s', linewidth=2.5, markersize=8,
                label=team, color=color)
    
    ax2.axvline(x=2022, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax2.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Points', fontsize=12, fontweight='bold')
    ax2.set_title('Championship Points by Season', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(config['figures_dir']) / 'pace_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    
    # Key insights
    logger.info("\nKey Insights:")
    
    # Best/worst teams by era
    pre_2022 = seasonal_pace[seasonal_pace['Year'] < 2022].groupby('team_name')['pace_delta'].mean()
    post_2022 = seasonal_pace[seasonal_pace['Year'] >= 2022].groupby('team_name')['pace_delta'].mean()
    
    logger.info(f"  Pre-2022 fastest: {pre_2022.idxmin()} (Δ = {pre_2022.min():.3f}s)")
    logger.info(f"  Post-2022 fastest: {post_2022.idxmin()} (Δ = {post_2022.min():.3f}s)")
    
    # Biggest change
    pace_change = post_2022 - pre_2022
    logger.info(f"  Biggest improvement: {pace_change.idxmin()} ({pace_change.min():.3f}s faster)")
    logger.info(f"  Biggest decline: {pace_change.idxmax()} (+{pace_change.max():.3f}s slower)")
    
    plt.show()


# =============================================================================
# ANALYSIS 2: STRATEGY PATTERNS
# =============================================================================

def plot_strategy_patterns(data: Dict, config: Dict, logger: logging.Logger):
    """Analyze and plot strategy patterns by team."""
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 2: Strategy Patterns")
    logger.info("="*60)
    
    driver_metrics = data['driver_metrics']
    
    # Aggregate strategy metrics by team
    strategy = driver_metrics.groupby('team_name').agg({
        'pit_stop_count': 'mean',
        'first_stop_lap': 'mean',
        'fraction_soft': 'mean',
        'fraction_medium': 'mean',
        'fraction_hard': 'mean'
    }).reset_index()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Average pit stops
    ax1 = axes[0, 0]
    bars1 = ax1.bar(strategy['team_name'], strategy['pit_stop_count'],
                    color=[TEAM_COLORS.get(t, 'gray') for t in strategy['team_name']])
    ax1.set_ylabel('Average Pit Stops per Race', fontweight='bold')
    ax1.set_title('Pit Stop Frequency', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: First stop timing
    ax2 = axes[0, 1]
    bars2 = ax2.bar(strategy['team_name'], strategy['first_stop_lap'],
                    color=[TEAM_COLORS.get(t, 'gray') for t in strategy['team_name']])
    ax2.set_ylabel('Average First Stop Lap', fontweight='bold')
    ax2.set_title('Pit Stop Timing', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Compound usage (stacked bar)
    ax3 = axes[1, 0]
    x = np.arange(len(strategy['team_name']))
    width = 0.6
    
    p1 = ax3.bar(x, strategy['fraction_soft'], width, label='Soft', color='#FF4444')
    p2 = ax3.bar(x, strategy['fraction_medium'], width, bottom=strategy['fraction_soft'],
                label='Medium', color='#FFD700')
    p3 = ax3.bar(x, strategy['fraction_hard'], width,
                bottom=strategy['fraction_soft'] + strategy['fraction_medium'],
                label='Hard', color='#FFFFFF', edgecolor='black')
    
    ax3.set_ylabel('Proportion of Laps', fontweight='bold')
    ax3.set_title('Tyre Compound Usage', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategy['team_name'], rotation=45, ha='right')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Strategy diversity (scatter)
    ax4 = axes[1, 1]
    for _, row in strategy.iterrows():
        team = row['team_name']
        color = TEAM_COLORS.get(team, 'gray')
        ax4.scatter(row['first_stop_lap'], row['pit_stop_count'],
                   s=300, color=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax4.text(row['first_stop_lap'], row['pit_stop_count'], 
                team.split()[0], ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax4.set_xlabel('First Stop Lap', fontweight='bold')
    ax4.set_ylabel('Pit Stops per Race', fontweight='bold')
    ax4.set_title('Strategy Profile', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('F1 Race Strategy Patterns by Team (2019-2023)', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    output_path = Path(config['figures_dir']) / 'strategy_patterns.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    
    # Insights
    logger.info("\nKey Insights:")
    logger.info(f"  Most pit stops: {strategy.loc[strategy['pit_stop_count'].idxmax(), 'team_name']} "
               f"({strategy['pit_stop_count'].max():.2f})")
    logger.info(f"  Earliest stopper: {strategy.loc[strategy['first_stop_lap'].idxmin(), 'team_name']} "
               f"(lap {strategy['first_stop_lap'].min():.1f})")
    logger.info(f"  Most soft usage: {strategy.loc[strategy['fraction_soft'].idxmax(), 'team_name']} "
               f"({strategy['fraction_soft'].max():.1%})")
    
    plt.show()


# =============================================================================
# ANALYSIS 3: DEGRADATION VS RESULTS
# =============================================================================

def plot_degradation_analysis(data: Dict, config: Dict, logger: logging.Logger):
    """Analyze tyre degradation vs race results."""
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 3: Degradation vs Results")
    logger.info("="*60)
    
    stint_metrics = data['stint_metrics']
    driver_metrics = data['driver_metrics']
    
    # Calculate average degradation per driver per race
    driver_deg = stint_metrics.groupby(['Year', 'Round', 'DriverNumber']).agg({
        'degradation_slope': 'mean',
        'stint_length_laps': 'mean'
    }).reset_index()
    
    # Merge with results
    merged = driver_deg.merge(
        driver_metrics[['Year', 'Round', 'DriverNumber', 'team_name', 'finish_position', 'points']],
        on=['Year', 'Round', 'DriverNumber'],
        how='inner'
    )
    
    # Remove DNFs and invalid positions
    merged = merged[merged['finish_position'].notna()]
    
    # Check if we have enough data
    if len(merged) < 10:
        logger.warning(f"Insufficient data for degradation analysis ({len(merged)} records). Skipping.")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Degradation vs finish position
    ax1 = axes[0]
    
    for team in merged['team_name'].unique():
        team_data = merged[merged['team_name'] == team]
        color = TEAM_COLORS.get(team, 'gray')
        ax1.scatter(team_data['degradation_slope'], team_data['finish_position'],
                   alpha=0.6, s=50, color=color, label=team, edgecolors='black', linewidth=0.5)
    
    # Add trend line only if we have valid data
    valid_data = merged[merged['degradation_slope'].notna() & merged['finish_position'].notna()]
    if len(valid_data) >= 2:
        z = np.polyfit(valid_data['degradation_slope'], valid_data['finish_position'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['degradation_slope'].min(), valid_data['degradation_slope'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={np.corrcoef(valid_data["degradation_slope"], valid_data["finish_position"])[0,1]:.2f})')
    
    ax1.set_xlabel('Average Degradation (seconds/lap)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Finish Position', fontweight='bold', fontsize=11)
    ax1.set_title('Tyre Degradation vs Race Result', fontweight='bold', fontsize=12)
    ax1.invert_yaxis()  # Lower position = better
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Degradation by compound
    ax2 = axes[1]
    
    compound_deg = stint_metrics[stint_metrics['compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()
    
    # Box plot
    compounds = ['SOFT', 'MEDIUM', 'HARD']
    colors = ['#FF4444', '#FFD700', '#FFFFFF']
    
    positions = []
    for i, compound in enumerate(compounds):
        data = compound_deg[compound_deg['compound'] == compound]['degradation_slope']
        positions.append(data.values)
    
    bp = ax2.boxplot(positions, labels=compounds, patch_artist=True,
                     showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    
    ax2.set_ylabel('Degradation Rate (seconds/lap)', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Tyre Compound', fontweight='bold', fontsize=11)
    ax2.set_title('Degradation by Compound', fontweight='bold', fontsize=12)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Tyre Degradation Analysis (2019-2023)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(config['figures_dir']) / 'degradation_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    
    # Statistics
    logger.info("\nKey Insights:")
    
    for compound in compounds:
        comp_data = compound_deg[compound_deg['compound'] == compound]['degradation_slope']
        if len(comp_data) > 0:
            logger.info(f"  {compound}: mean={comp_data.mean():.4f} s/lap, "
                       f"median={comp_data.median():.4f} s/lap, "
                       f"std={comp_data.std():.4f}")
        else:
            logger.info(f"  {compound}: no data available")
    
    # Correlation
    if len(valid_data) >= 2:
        corr = merged['degradation_slope'].corr(merged['finish_position'])
        logger.info(f"  Correlation (deg vs finish): {corr:.3f}")
    
    plt.show()


# =============================================================================
# ANALYSIS 4: ANOMALY DETECTION
# =============================================================================

def detect_and_plot_anomalies(data: Dict, config: Dict, logger: logging.Logger):
    """Find and visualize race anomalies."""
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS 4: Anomaly Detection")
    logger.info("="*60)
    
    team_metrics = data['team_metrics']
    
    # Calculate z-scores for pace delta
    team_metrics['pace_z_score'] = team_metrics.groupby('team_name')['pace_delta'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Find anomalies (|z| > 2)
    anomalies = team_metrics[np.abs(team_metrics['pace_z_score']) > 2].copy()
    anomalies = anomalies.sort_values('pace_z_score', ascending=False)
    
    logger.info(f"Found {len(anomalies)} anomalous race performances")
    
    if len(anomalies) == 0:
        logger.warning("No anomalies detected. Skipping anomaly case study.")
        return
    
    # Show top anomalies
    for idx, row in anomalies.head(10).iterrows():
        direction = "slower" if row['pace_z_score'] > 0 else "faster"
        logger.info(f"  {row['Year']} {row['EventName']}: {row['team_name']} "
                   f"({row['pace_z_score']:.2f}σ {direction})")
    
    # Pick most interesting case for detailed plot
    if len(anomalies) > 0:
        case_study = anomalies.iloc[0]
        
        logger.info(f"\nCase Study: {case_study['Year']} {case_study['EventName']} - {case_study['team_name']}")
        
        # Get all teams from that race
        race_data = team_metrics[
            (team_metrics['Year'] == case_study['Year']) & 
            (team_metrics['Round'] == case_study['Round'])
        ].copy()
        
        race_data = race_data.sort_values('team_avg_pace')
        
        # Create case study plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Pace comparison
        ax1 = axes[0]
        colors = [TEAM_COLORS.get(t, 'gray') for t in race_data['team_name']]
        bars = ax1.barh(race_data['team_name'], race_data['pace_delta'], color=colors)
        
        # Highlight anomaly
        anomaly_idx = race_data['team_name'].tolist().index(case_study['team_name'])
        bars[anomaly_idx].set_edgecolor('red')
        bars[anomaly_idx].set_linewidth(3)
        
        ax1.set_xlabel('Pace Delta (seconds)', fontweight='bold')
        ax1.set_title(f'{case_study["EventName"]} {case_study["Year"]}\nTeam Race Pace', 
                     fontweight='bold', fontsize=12)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Strategy comparison
        ax2 = axes[1]
        x = np.arange(len(race_data))
        width = 0.35
        
        ax2.bar(x - width/2, race_data['team_avg_pit_stops'], width, label='Pit Stops', alpha=0.7)
        ax2.bar(x + width/2, race_data['team_avg_first_stop']/10, width, label='1st Stop Lap (/10)', alpha=0.7)
        
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Strategy Profile', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(race_data['team_name'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(config['figures_dir']) / 'anomaly_case_study.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_path}")
        
        plt.show()


# =============================================================================
# MAIN ANALYSIS ROUTINE
# =============================================================================

def main():
    """Run all analyses and generate visualizations."""
    
    # Setup
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("="*60)
    logger.info("F1 DATA ANALYSIS")
    logger.info("="*60)
    
    # Ensure output directory exists
    figures_dir = Path(config['figures_dir'])
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        data = load_data(config, logger)
        
        # Run analyses
        plot_team_pace_evolution(data, config, logger)
        plot_strategy_patterns(data, config, logger)
        plot_degradation_analysis(data, config, logger)
        detect_and_plot_anomalies(data, config, logger)
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"Figures saved to: {figures_dir}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\nAnalysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
