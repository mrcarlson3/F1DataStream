import sys
import logging
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings

import yaml
import pandas as pd
import numpy as np

# NumPy 2.0 compatibility: Add np.NaN as alias for np.nan
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import duckdb
import fastf1
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from tqdm import tqdm

# Suppress FastF1 warnings
warnings.filterwarnings('ignore', category=FutureWarning)
fastf1.Cache.set_disabled()  # We'll enable it in setup


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging with file handler only."""
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "pipeline.log"  # Fixed filename, overwrite each run
    
    # Create logger
    logger = logging.getLogger('f1_pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler (overwrite mode)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# =============================================================================
# CONFIGURATION
# =============================================================================

@task(name="load-config", retries=0)
def load_config() -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure paths are Path objects
        for key in ['cache_dir', 'db_path', 'log_dir', 'figures_dir']:
            if key in config:
                config[key] = str(Path(config[key]).resolve())
        
        return config
    except Exception as e:
        print(f"ERROR: Failed to load config from {config_path}: {e}")
        raise


@task(name="setup-environment", retries=0)
def setup_environment(config: Dict) -> logging.Logger:
    """Initialize FastF1 cache and logging."""
    logger = setup_logging(config)
    
    # Setup FastF1 cache
    cache_dir = Path(config['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        fastf1.Cache.enable_cache(str(cache_dir))
        logger.info(f"FastF1 cache enabled at: {cache_dir}")
    except Exception as e:
        logger.error(f"Failed to enable FastF1 cache: {e}")
        raise
    
    # Ensure database directory exists
    db_path = Path(config['db_path'])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    return logger


# =============================================================================
# STAGE 1: DATA INGESTION
# =============================================================================

@task(name="load-session", retries=0)  # Handle retries manually for more control
def load_session(year: int, round_num: int, event_name: str, config: Dict, logger: logging.Logger, max_retries: int = 3) -> Optional[Dict]:
    """Load a single race session with retry logic and cache clearing."""
    prefect_logger = get_run_logger()
    cache_dir = Path(config['cache_dir'])
    
    # Add delay between races to avoid rate limiting (skip for first round)
    if round_num > 1:
        time.sleep(3)  # 3 second delay between consecutive race loads
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"  Retry attempt {attempt + 1}/{max_retries} for {year} R{round_num}")
                prefect_logger.info(f"Retrying {year} R{round_num} (attempt {attempt + 1})")
                
                # Clear cache for this specific session on retry
                if cache_dir.exists():
                    pattern = f"*{year}*{round_num}*"
                    for cache_file in cache_dir.glob(pattern):
                        try:
                            cache_file.unlink()
                            logger.info(f"    Cleared cache: {cache_file.name}")
                        except Exception as e:
                            logger.warning(f"    Could not clear cache {cache_file.name}: {e}")
                
                # Longer exponential backoff for retries
                time.sleep(3 * (2 ** attempt))  # 6s, 12s, 24s
            
            logger.info(f"Loading {year} Round {round_num}: {event_name}")
            if attempt == 0:
                prefect_logger.info(f"Loading {year} R{round_num}: {event_name}")
            
            # Get session
            session = fastf1.get_session(year, round_num, 'R')
            
            # Load with appropriate parameters
            if year >= 2022:
                session.load(laps=True, telemetry=False, weather=False, messages=False)
            else:
                session.load(telemetry=False, weather=False, messages=False)
            
            # Validate session loaded
            if not hasattr(session, 'laps') or session.laps is None:
                raise ValueError("Session laps attribute missing")
            
            laps = session.laps
            if len(laps) == 0:
                raise ValueError("No laps data available")
            
            results = session.results
            if len(results) == 0:
                raise ValueError("No results data available")
            
            # Add metadata
            laps_df = laps.copy()
            laps_df['Year'] = year
            laps_df['Round'] = round_num
            laps_df['EventName'] = event_name
            
            results_df = results.copy()
            results_df['Year'] = year
            results_df['Round'] = round_num
            results_df['EventName'] = event_name
            
            logger.info(f"  ✓ Loaded {len(laps_df)} laps, {len(results_df)} drivers")
            prefect_logger.info(f"✓ {year} R{round_num}: {len(laps_df)} laps")
            
            return {'laps': laps_df, 'results': results_df}
            
        except Exception as e:
            error_msg = str(e)
            
            # Check for unrecoverable errors
            if "Failed to load any schedule data" in error_msg:
                logger.error(f"  ✗ Schedule unavailable for {year} R{round_num}")
                prefect_logger.error(f"Schedule unavailable: {year} R{round_num}")
                return None  # Don't retry
            
            if attempt < max_retries - 1:
                logger.warning(f"  ⚠ Attempt {attempt + 1} failed: {error_msg}")
                continue  # Try again
            else:
                logger.error(f"  ✗ Failed to load {year} R{round_num} after {max_retries} attempts: {error_msg}")
                prefect_logger.error(f"✗ {year} R{round_num} failed after {max_retries} attempts")
                return None


@task(name="ingest-season", retries=1)
def ingest_season(year: int, config: Dict, logger: logging.Logger) -> Dict:
    """Ingest all races from a single season with detailed tracking."""
    prefect_logger = get_run_logger()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"INGESTING SEASON: {year}")
    logger.info(f"{'='*60}")
    prefect_logger.info(f"Starting ingestion for season {year}")
    
    all_data = []
    failed_races = []
    
    try:
        # Get event schedule
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule['EventFormat'] != 'testing']
        
        logger.info(f"Found {len(races)} events for {year}")
        prefect_logger.info(f"Season {year}: {len(races)} races to load")
        
        # Load each race
        for idx, event in races.iterrows():
            round_num = event['RoundNumber']
            event_name = event['EventName']
            
            session_data = load_session(year, round_num, event_name, config, logger)
            
            if session_data:
                all_data.append(session_data)
            else:
                failed_races.append(f"R{round_num} - {event_name}")
        
        logger.info(f"Successfully loaded {len(all_data)}/{len(races)} races for {year}")
        prefect_logger.info(f"Season {year} complete: {len(all_data)}/{len(races)} races loaded")
        
        if failed_races:
            logger.warning(f"Failed races for {year}:")
            for race in failed_races:
                logger.warning(f"  - {race}")
        
        return {
            'year': year,
            'data': all_data,
            'successful': len(all_data),
            'total': len(races),
            'failed': failed_races
        }
        
    except Exception as e:
        logger.error(f"Failed to get schedule for {year}: {e}")
        prefect_logger.error(f"Season {year} failed: {e}")
        return {
            'year': year,
            'data': [],
            'successful': 0,
            'total': 0,
            'failed': []
        }


@task(name="store-raw-data", retries=2, retry_delay_seconds=2)
def store_raw_data(all_season_data: List[Dict], config: Dict, logger: logging.Logger):
    """Store raw laps and results in DuckDB."""
    logger.info(f"\n{'='*60}")
    logger.info("STORING RAW DATA TO DUCKDB")
    logger.info(f"{'='*60}")
    
    db_path = Path(config['db_path'])
    
    # Backup old database if it exists
    if db_path.exists():
        backup_path = db_path.with_suffix('.duckdb.backup')
        if backup_path.exists():
            backup_path.unlink()  # Remove old backup
        db_path.rename(backup_path)
        logger.info(f"Backed up existing database to {backup_path}")
    
    try:
        # Flatten all session data
        all_laps = []
        all_results = []
        
        for season_info in all_season_data:
            for session_data in season_info['data']:
                if session_data:
                    all_laps.append(session_data['laps'])
                    all_results.append(session_data['results'])
        
        if not all_laps:
            logger.error("No data to store!")
            return
        
        # Concatenate all data
        laps_df = pd.concat(all_laps, ignore_index=True)
        results_df = pd.concat(all_results, ignore_index=True)
        
        logger.info(f"Total laps: {len(laps_df):,}")
        logger.info(f"Total results: {len(results_df):,}")
        
        # Connect to DuckDB (create new database)
        con = duckdb.connect(str(db_path))
        
        # Store raw laps
        con.execute("DROP TABLE IF EXISTS raw_laps")
        con.execute("""
            CREATE TABLE raw_laps AS 
            SELECT * FROM laps_df
        """)
        logger.info("✓ Stored raw_laps table")
        
        # Store raw results
        con.execute("DROP TABLE IF EXISTS raw_results")
        con.execute("""
            CREATE TABLE raw_results AS 
            SELECT * FROM results_df
        """)
        logger.info("✓ Stored raw_results table")
        
        # Show summary
        lap_count = con.execute("SELECT COUNT(*) FROM raw_laps").fetchone()[0]
        result_count = con.execute("SELECT COUNT(*) FROM raw_results").fetchone()[0]
        
        logger.info(f"\nDatabase summary:")
        logger.info(f"  - raw_laps: {lap_count:,} rows")
        logger.info(f"  - raw_results: {result_count:,} rows")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to store raw data: {e}")
        raise


@task(name="ingest-all-sessions")
def ingest_all_sessions(config: Dict, logger: logging.Logger):
    """Main ingestion task - fetch all seasons with comprehensive tracking."""
    prefect_logger = get_run_logger()
    
    logger.info(f"\n{'#'*60}")
    logger.info("STAGE 1: DATA INGESTION")
    logger.info(f"{'#'*60}")
    
    seasons = config['seasons']
    logger.info(f"Ingesting {len(seasons)} seasons: {seasons}")
    prefect_logger.info(f"Starting ingestion: {len(seasons)} seasons")
    
    # Ingest each season
    all_season_data = []
    for year in seasons:
        season_data = ingest_season(year, config, logger)
        all_season_data.append(season_data)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("INGESTION SUMMARY")
    logger.info(f"{'='*60}")
    
    total_successful = 0
    total_races = 0
    
    for season_info in all_season_data:
        success_rate = (season_info['successful'] / season_info['total'] * 100) if season_info['total'] > 0 else 0
        logger.info(f"  {season_info['year']}: {season_info['successful']}/{season_info['total']} races ({success_rate:.1f}%)")
        total_successful += season_info['successful']
        total_races += season_info['total']
    
    overall_rate = (total_successful / total_races * 100) if total_races > 0 else 0
    logger.info(f"\n  Overall: {total_successful}/{total_races} races ({overall_rate:.1f}%)")
    prefect_logger.info(f"Ingestion complete: {total_successful}/{total_races} races ({overall_rate:.1f}%)")
    
    # Store all data
    store_raw_data(all_season_data, config, logger)
    
    logger.info("\n✓ Stage 1 complete: All data ingested")


# =============================================================================
# STAGE 2: DATA CLEANING
# =============================================================================

@task(name="create-dimension-tables", retries=2)
def create_dimension_tables(config: Dict, logger: logging.Logger):
    """Create driver and team dimension tables."""
    logger.info("Creating dimension tables...")
    
    db_path = config['db_path']
    team_mapping = config.get('team_mapping', {})
    
    try:
        con = duckdb.connect(db_path)
        
        # Create dim_driver
        con.execute("""
            CREATE OR REPLACE TABLE dim_driver AS
            SELECT DISTINCT
                DriverNumber AS driver_number,
                Abbreviation AS driver_code,
                FullName AS driver_name
            FROM raw_results
            WHERE DriverNumber IS NOT NULL
        """)
        logger.info("  ✓ Created dim_driver")
        
        # Create dim_team with mapping
        con.execute("""
            CREATE OR REPLACE TABLE dim_team AS
            SELECT DISTINCT TeamName AS team_name
            FROM raw_results
            WHERE TeamName IS NOT NULL
        """)
        
        # Apply team name mapping
        for old_name, new_name in team_mapping.items():
            con.execute("""
                UPDATE dim_team 
                SET team_name = ?
                WHERE team_name = ?
            """, [new_name, old_name])
        
        logger.info("  ✓ Created dim_team with name mapping")
        
        driver_count = con.execute("SELECT COUNT(*) FROM dim_driver").fetchone()[0]
        team_count = con.execute("SELECT COUNT(*) FROM dim_team").fetchone()[0]
        
        logger.info(f"  Dimensions: {driver_count} drivers, {team_count} teams")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to create dimension tables: {e}")
        raise


@task(name="clean-results", retries=2)
def clean_results(config: Dict, logger: logging.Logger):
    """Clean and standardize results table."""
    logger.info("Cleaning results...")
    
    db_path = config['db_path']
    teams_of_interest = config.get('teams_of_interest', [])
    team_mapping = config.get('team_mapping', {})
    
    try:
        con = duckdb.connect(db_path)
        
        # Build team filter
        all_team_names = teams_of_interest + list(team_mapping.keys())
        team_filter = "', '".join(all_team_names)
        
        con.execute(f"""
            CREATE OR REPLACE TABLE clean_results AS
            SELECT 
                Year,
                Round,
                EventName,
                DriverNumber AS driver_number,
                Abbreviation AS driver_code,
                FullName AS driver_name,
                TeamName AS team_name,
                CAST(Position AS INTEGER) AS finish_position,
                CAST(Points AS DOUBLE) AS points,
                Status,
                GridPosition AS grid_position
            FROM raw_results
            WHERE TeamName IN ('{team_filter}')
                AND DriverNumber IS NOT NULL
        """)
        
        # Apply team name mapping
        for old_name, new_name in team_mapping.items():
            con.execute("""
                UPDATE clean_results 
                SET team_name = ?
                WHERE team_name = ?
            """, [new_name, old_name])
        
        result_count = con.execute("SELECT COUNT(*) FROM clean_results").fetchone()[0]
        logger.info(f"  ✓ Cleaned {result_count:,} result records (top 6 teams only)")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to clean results: {e}")
        raise


@task(name="clean-laps", retries=2)
def clean_laps(config: Dict, logger: logging.Logger):
    """Clean and standardize laps table with data quality checks."""
    logger.info("Cleaning laps...")
    
    db_path = config['db_path']
    max_lap_time = config.get('max_lap_time_seconds', 200)
    min_lap_time = config.get('min_lap_time_seconds', 60)
    
    try:
        con = duckdb.connect(db_path)
        
        # Get laps data
        laps_df = con.execute("SELECT * FROM raw_laps").df()
        
        logger.info(f"  Processing {len(laps_df):,} raw laps...")
        
        # Convert time columns to seconds
        time_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
        
        for col in time_cols:
            if col in laps_df.columns:
                # Handle Timedelta objects
                laps_df[f'{col}_s'] = laps_df[col].apply(
                    lambda x: x.total_seconds() if pd.notna(x) and hasattr(x, 'total_seconds') else None
                )
        
        # Create pit lap flags
        laps_df['is_pit_lap'] = (
            laps_df['PitInTime'].notna() | laps_df['PitOutTime'].notna()
        )
        laps_df['is_pit_in_lap'] = laps_df['PitInTime'].notna()
        laps_df['is_pit_out_lap'] = laps_df['PitOutTime'].notna()
        
        # Data quality: filter lap times
        initial_count = len(laps_df)
        laps_df = laps_df[
            (laps_df['LapTime_s'].notna()) &
            (laps_df['LapTime_s'] >= min_lap_time) &
            (laps_df['LapTime_s'] <= max_lap_time)
        ]
        filtered_count = initial_count - len(laps_df)
        
        if filtered_count > 0:
            logger.info(f"  ⚠ Filtered {filtered_count:,} laps with invalid times")
        
        # Select final columns
        clean_laps = laps_df[[
            'Year', 'Round', 'EventName',
            'DriverNumber', 'LapNumber', 'Stint',
            'LapTime_s', 'Sector1Time_s', 'Sector2Time_s', 'Sector3Time_s',
            'Compound', 'TyreLife', 'TrackStatus',
            'is_pit_lap', 'is_pit_in_lap', 'is_pit_out_lap'
        ]].copy()
        
        # Store to DuckDB
        con.execute("DROP TABLE IF EXISTS clean_laps")
        con.execute("""
            CREATE TABLE clean_laps AS 
            SELECT * FROM clean_laps
        """)
        
        lap_count = con.execute("SELECT COUNT(*) FROM clean_laps").fetchone()[0]
        logger.info(f"  ✓ Cleaned {lap_count:,} laps")
        
        # Show compound distribution
        compounds = con.execute("""
            SELECT Compound, COUNT(*) as count
            FROM clean_laps
            WHERE Compound IS NOT NULL
            GROUP BY Compound
            ORDER BY count DESC
        """).df()
        
        logger.info("  Tyre compound distribution:")
        for _, row in compounds.iterrows():
            logger.info(f"    {row['Compound']}: {row['count']:,}")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to clean laps: {e}")
        raise


@task(name="clean-and-standardize")
def clean_and_standardize(config: Dict, logger: logging.Logger):
    """Main cleaning task."""
    logger.info(f"\n{'#'*60}")
    logger.info("STAGE 2: DATA CLEANING & STANDARDIZATION")
    logger.info(f"{'#'*60}")
    
    create_dimension_tables(config, logger)
    clean_results(config, logger)
    clean_laps(config, logger)
    
    logger.info("\n✓ Stage 2 complete: Data cleaned and standardized")


# =============================================================================
# STAGE 3: FEATURE ENGINEERING
# =============================================================================

@task(name="build-stint-metrics", retries=2)
def build_stint_metrics(config: Dict, logger: logging.Logger):
    """Calculate stint-level metrics including degradation slopes."""
    logger.info("Building stint metrics...")
    
    db_path = config['db_path']
    
    try:
        con = duckdb.connect(db_path)
        
        # Get clean laps (exclude pit laps for pace analysis)
        laps_df = con.execute("""
            SELECT 
                Year, Round, EventName,
                DriverNumber, LapNumber, Stint,
                LapTime_s, Compound, TyreLife,
                is_pit_lap
            FROM clean_laps
            WHERE is_pit_lap = FALSE
                AND LapTime_s IS NOT NULL
                AND Compound IS NOT NULL
        """).df()
        
        logger.info(f"  Analyzing {len(laps_df):,} racing laps...")
        
        # Calculate stint metrics
        stint_metrics = []
        
        grouped = laps_df.groupby(['Year', 'Round', 'DriverNumber', 'Stint'])
        
        for (year, round_num, driver, stint), group in grouped:
            if len(group) < 3:  # Skip very short stints
                continue
            
            # Sort by lap number
            group = group.sort_values('LapNumber')
            
            # Stint metadata
            compound = group['Compound'].iloc[0]
            stint_laps = len(group)
            
            # Create stint lap index (0, 1, 2, ...)
            group['stint_lap_index'] = range(len(group))
            
            # Calculate degradation slope (linear regression)
            if stint_laps >= 5:
                try:
                    # Simple linear regression: y = mx + b
                    x = group['stint_lap_index'].values
                    y = group['LapTime_s'].values
                    
                    # Remove outliers (beyond 2 std)
                    mean_time = y.mean()
                    std_time = y.std()
                    mask = np.abs(y - mean_time) <= 2 * std_time
                    
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) >= 3:
                        slope, intercept = np.polyfit(x_clean, y_clean, 1)
                    else:
                        slope = 0.0
                        intercept = mean_time
                except:
                    slope = 0.0
                    intercept = group['LapTime_s'].mean()
            else:
                slope = 0.0
                intercept = group['LapTime_s'].mean()
            
            stint_metrics.append({
                'Year': year,
                'Round': round_num,
                'DriverNumber': driver,
                'Stint': stint,
                'compound': compound,
                'stint_length_laps': stint_laps,
                'stint_mean_lap_time': group['LapTime_s'].mean(),
                'stint_median_lap_time': group['LapTime_s'].median(),
                'stint_best_lap': group['LapTime_s'].min(),
                'stint_std': group['LapTime_s'].std(),
                'degradation_slope': slope,  # seconds per lap
                'stint_base_pace': intercept  # intercept at lap 0
            })
        
        stint_df = pd.DataFrame(stint_metrics)
        
        # Store to DuckDB
        con.execute("DROP TABLE IF EXISTS stint_metrics")
        con.execute("""
            CREATE TABLE stint_metrics AS 
            SELECT * FROM stint_df
        """)
        
        stint_count = con.execute("SELECT COUNT(*) FROM stint_metrics").fetchone()[0]
        logger.info(f"  ✓ Created {stint_count:,} stint records")
        
        # Show degradation summary
        avg_deg = con.execute("""
            SELECT 
                compound,
                AVG(degradation_slope) as avg_degradation,
                COUNT(*) as stint_count
            FROM stint_metrics
            GROUP BY compound
            ORDER BY avg_degradation DESC
        """).df()
        
        logger.info("  Average degradation by compound (s/lap):")
        for _, row in avg_deg.iterrows():
            logger.info(f"    {row['compound']}: {row['avg_degradation']:.3f} ({row['stint_count']} stints)")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to build stint metrics: {e}")
        raise


@task(name="build-driver-race-metrics", retries=2)
def build_driver_race_metrics(config: Dict, logger: logging.Logger):
    """Calculate driver-level race metrics."""
    logger.info("Building driver race metrics...")
    
    db_path = config['db_path']
    
    try:
        con = duckdb.connect(db_path)
        
        # Get clean laps for pace calculation
        pace_metrics = con.execute("""
            SELECT 
                l.Year, l.Round, l.EventName,
                l.DriverNumber,
                r.driver_name,
                r.team_name,
                r.finish_position,
                r.points,
                -- Race pace (median of non-pit laps)
                MEDIAN(l.LapTime_s) as avg_race_pace,
                -- Pace consistency
                STDDEV(l.LapTime_s) as pace_std,
                MAD(l.LapTime_s) as pace_mad,
                -- Lap counts
                COUNT(*) as total_laps,
                -- Pit stops
                SUM(CASE WHEN l.is_pit_lap THEN 1 ELSE 0 END) as pit_stop_count
            FROM clean_laps l
            JOIN clean_results r 
                ON l.Year = r.Year 
                AND l.Round = r.Round 
                AND l.DriverNumber = r.driver_number
            WHERE l.is_pit_lap = FALSE
                AND l.LapTime_s IS NOT NULL
            GROUP BY 
                l.Year, l.Round, l.EventName,
                l.DriverNumber, r.driver_name, r.team_name,
                r.finish_position, r.points
        """).df()
        
        # Get pit stop details
        pit_details = con.execute("""
            SELECT 
                Year, Round, DriverNumber,
                MIN(LapNumber) as first_stop_lap,
                MAX(LapNumber) as last_stop_lap
            FROM clean_laps
            WHERE is_pit_lap = TRUE
            GROUP BY Year, Round, DriverNumber
        """).df()
        
        # Merge
        driver_metrics = pace_metrics.merge(
            pit_details,
            on=['Year', 'Round', 'DriverNumber'],
            how='left'
        )
        
        # Get compound usage
        compound_usage = con.execute("""
            SELECT 
                Year, Round, DriverNumber,
                SUM(CASE WHEN Compound = 'SOFT' THEN 1 ELSE 0 END) as soft_laps,
                SUM(CASE WHEN Compound = 'MEDIUM' THEN 1 ELSE 0 END) as medium_laps,
                SUM(CASE WHEN Compound = 'HARD' THEN 1 ELSE 0 END) as hard_laps,
                COUNT(*) as total_compound_laps
            FROM clean_laps
            WHERE Compound IS NOT NULL
                AND is_pit_lap = FALSE
            GROUP BY Year, Round, DriverNumber
        """).df()
        
        # Calculate fractions
        compound_usage['fraction_soft'] = compound_usage['soft_laps'] / compound_usage['total_compound_laps']
        compound_usage['fraction_medium'] = compound_usage['medium_laps'] / compound_usage['total_compound_laps']
        compound_usage['fraction_hard'] = compound_usage['hard_laps'] / compound_usage['total_compound_laps']
        
        # Merge compound data
        driver_metrics = driver_metrics.merge(
            compound_usage[['Year', 'Round', 'DriverNumber', 'fraction_soft', 'fraction_medium', 'fraction_hard']],
            on=['Year', 'Round', 'DriverNumber'],
            how='left'
        )
        
        # Fill NaN values
        driver_metrics = driver_metrics.fillna({
            'pit_stop_count': 0,
            'first_stop_lap': 0,
            'fraction_soft': 0,
            'fraction_medium': 0,
            'fraction_hard': 0
        })
        
        # Store to DuckDB
        con.execute("DROP TABLE IF EXISTS driver_race_metrics")
        con.execute("""
            CREATE TABLE driver_race_metrics AS 
            SELECT * FROM driver_metrics
        """)
        
        metric_count = con.execute("SELECT COUNT(*) FROM driver_race_metrics").fetchone()[0]
        logger.info(f"  ✓ Created {metric_count:,} driver-race records")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to build driver race metrics: {e}")
        raise


@task(name="build-team-race-metrics", retries=2)
def build_team_race_metrics(config: Dict, logger: logging.Logger):
    """Calculate team-level race metrics with relative pace."""
    logger.info("Building team race metrics...")
    
    db_path = config['db_path']
    
    try:
        con = duckdb.connect(db_path)
        
        # Aggregate driver metrics to team level
        team_metrics = con.execute("""
            SELECT 
                Year, Round, EventName, team_name,
                -- Pace metrics
                AVG(avg_race_pace) as team_avg_pace,
                AVG(pace_std) as team_pace_consistency,
                -- Strategy
                AVG(pit_stop_count) as team_avg_pit_stops,
                AVG(first_stop_lap) as team_avg_first_stop,
                AVG(fraction_soft) as team_fraction_soft,
                AVG(fraction_medium) as team_fraction_medium,
                AVG(fraction_hard) as team_fraction_hard,
                -- Results
                AVG(finish_position) as team_avg_finish,
                SUM(points) as team_points,
                COUNT(*) as drivers_finished
            FROM driver_race_metrics
            GROUP BY Year, Round, EventName, team_name
        """).df()
        
        # Calculate relative pace (delta from best team in each race)
        team_metrics['best_pace_in_race'] = team_metrics.groupby(['Year', 'Round'])['team_avg_pace'].transform('min')
        team_metrics['pace_delta'] = team_metrics['team_avg_pace'] - team_metrics['best_pace_in_race']
        
        # Store to DuckDB
        con.execute("DROP TABLE IF EXISTS team_race_metrics")
        con.execute("""
            CREATE TABLE team_race_metrics AS 
            SELECT * FROM team_metrics
        """)
        
        metric_count = con.execute("SELECT COUNT(*) FROM team_race_metrics").fetchone()[0]
        logger.info(f"  ✓ Created {metric_count:,} team-race records")
        
        # Show summary by team
        team_summary = con.execute("""
            SELECT 
                team_name,
                COUNT(*) as races,
                AVG(pace_delta) as avg_pace_delta,
                AVG(team_points) as avg_points_per_race
            FROM team_race_metrics
            GROUP BY team_name
            ORDER BY avg_pace_delta
        """).df()
        
        logger.info("  Team performance summary:")
        for _, row in team_summary.iterrows():
            logger.info(f"    {row['team_name']}: {row['races']} races, "
                       f"pace Δ: {row['avg_pace_delta']:.3f}s, "
                       f"avg pts: {row['avg_points_per_race']:.1f}")
        
        con.close()
        
    except Exception as e:
        logger.error(f"Failed to build team race metrics: {e}")
        raise


@task(name="build-features")
def build_features(config: Dict, logger: logging.Logger):
    """Main feature engineering task."""
    logger.info(f"\n{'#'*60}")
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info(f"{'#'*60}")
    
    build_stint_metrics(config, logger)
    build_driver_race_metrics(config, logger)
    build_team_race_metrics(config, logger)
    
    logger.info("\n✓ Stage 3 complete: All features built")


# =============================================================================
# MAIN PIPELINE FLOW
# =============================================================================

@flow(name="f1-data-pipeline", log_prints=True)
def f1_data_pipeline():
    """Main Prefect flow orchestrating the complete F1 data pipeline."""
    
    # Load configuration
    config = load_config()
    
    # Setup environment and logging
    logger = setup_environment(config)
    
    start_time = datetime.now()
    logger.info(f"\n{'='*60}")
    logger.info(f"F1 DATA PIPELINE STARTED")
    logger.info(f"Timestamp: {start_time}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Stage 1: Ingest
        ingest_all_sessions(config, logger)
        
        # Stage 2: Clean
        clean_and_standardize(config, logger)
        
        # Stage 3: Features
        build_features(config, logger)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration}")
        logger.info(f"Database: {config['db_path']}")
        logger.info(f"{'='*60}\n")
        
        # Show final table counts
        db_path = config['db_path']
        con = duckdb.connect(db_path)
        
        tables = ['raw_laps', 'raw_results', 'clean_laps', 'clean_results',
                 'stint_metrics', 'driver_race_metrics', 'team_race_metrics']
        
        logger.info("Final database summary:")
        for table in tables:
            try:
                count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                logger.info(f"  {table}: {count:,} rows")
            except:
                logger.warning(f"  {table}: not found")
        
        con.close()
        
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(f"{'='*60}\n")
        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run the flow directly
    print("Starting F1 Data Pipeline...")
    print("This will take 30-60 minutes for the first run.")
    print("Progress will be logged to data/logs/\n")
    
    f1_data_pipeline()
