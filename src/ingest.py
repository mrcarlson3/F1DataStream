import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import warnings
import yaml
import pandas as pd
import numpy as np
import duckdb
import fastf1

from fastf1.core import DataNotLoadedError, NoLapDataError
from fastf1 import RateLimitExceededError

from prefect import flow, task, get_run_logger

from transform import (  # new import
    create_dimension_tables,
    clean_results,
    clean_laps,
    build_stint_metrics,
    build_driver_race_metrics,
    build_team_race_metrics,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# NumPy 2.0 compatibility
if not hasattr(np, "NaN"):
    np.NaN = np.nan


def setup_logging(log_dir: str) -> None:
    """Configure logging to append Prefect logs to a local file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "pipeline.log"
    
    # Configure root logger to write to file (append mode)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to Prefect's logger
    prefect_logger = logging.getLogger("prefect")
    prefect_logger.addHandler(file_handler)
    prefect_logger.setLevel(logging.INFO)
    
    # Add a run separator
    separator = "\n" + "="*80
    separator += f"\nPipeline Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    separator += "\n" + "="*80 + "\n"
    with open(log_file, 'a') as f:
        f.write(separator)


# config 


# Task: Load the YAML configuration file and normalize key filesystem paths.
# This ensures later tasks receive an absolute-path config dictionary.
@task
def load_config() -> Dict:
    """Load configuration from config.yaml and normalize relevant paths."""
    logger = get_run_logger()
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    for key in ["cache_dir", "db_path", "figures_dir", "log_dir"]:
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser().resolve())

    logger.info(f"Loaded config from {config_path}")
    return cfg


# Task: Prepare the runtime environment such as FastF1 cache and database folder.
# This guarantees required directories exist and FastF1 caching is enabled once.
@task
def setup_environment(config: Dict) -> None:
    """Enable FastF1 cache and ensure DB directory exists."""
    logger = get_run_logger()

    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Recommended way from docs: enable cache once at startup
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info(f"FastF1 cache enabled at: {cache_dir}")

    db_path = Path(config["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"DuckDB database path: {db_path}")


#ingestion 

# Task: Load one race session (laps + results) from FastF1 with retries/backoff.
# It returns a dict of DataFrames for laps and results or None if data is unavailable.
@task
def load_session(
    year: int,
    round_number: int,
    event_name: str,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> Optional[Dict]:
    """
    Load a single race session using the patterns documented for FastF1 3.6.1.

    - Uses `fastf1.get_session(year, gp, identifier)` where
      `gp` = round number (int) and `identifier` = 'R' (race). 
    - Calls `Session.load(laps=True, telemetry=False, weather=False, messages=True)`.
      (laps=True so `.laps` is available; messages=True so race control-based
       lap corrections and results calculations work correctly.) 
    - Handles:
        * `DataNotLoadedError` / `NoLapDataError` → session has no usable timing/results
        * `RateLimitExceededError` → backoff and retry
        * Other exceptions → retry up to `max_retries`
    """

    logger = get_run_logger()

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Retry {attempt}/{max_retries-1} for {year} R{round_number} "
                    f"after {delay:.0f}s"
                )
                time.sleep(delay)

            logger.info(f"Loading {year} Round {round_number}: {event_name}")

            # Be gentle to backend / avoid hitting hard rate limits.
            time.sleep(1.0)

            # Per docs: get_session(year, gp, identifier) where gp can be round number
            # and identifier can be 'R' for Race.
            session = fastf1.get_session(year, round_number, "R")

            # Recommended to load all relevant data at once. Telemetry is optional for ETL.
            session.load(
                laps=True,
                telemetry=False,
                weather=False,
                messages=True,
            )

            laps = session.laps
            results = session.results

            if laps is None or len(laps) == 0:
                raise NoLapDataError("No usable laps data for this session.")
            if results is None or len(results) == 0:
                raise DataNotLoadedError(
                    "Results data not available for this session."
                )

            laps_df = laps.copy()
            laps_df["Year"] = year
            laps_df["Round"] = round_number
            laps_df["EventName"] = event_name

            results_df = results.copy()
            results_df["Year"] = year
            results_df["Round"] = round_number
            results_df["EventName"] = event_name

            logger.info(
                f"✓ {year} R{round_number}: {len(laps_df)} laps, {len(results_df)} results"
            )
            return {"laps": laps_df, "results": results_df}

        except RateLimitExceededError as e:
            logger.warning(
                f"Rate limit exceeded while loading {year} R{round_number}: {e}"
            )
            if attempt == max_retries - 1:
                logger.error(
                    f"Giving up on {year} R{round_number} due to rate limit."
                )
                return None

            delay = base_delay * (2 ** attempt) * 2
            logger.info(f"Backing off for {delay:.0f}s due to rate limit")
            time.sleep(delay)

        except (DataNotLoadedError, NoLapDataError) as e:
            logger.warning(
                f"No usable data for {year} R{round_number} ({event_name}): {e}"
            )
            return None

        except Exception as e:
            msg = str(e)

            if "Failed to load any schedule data" in msg:
                logger.error(
                    f"Schedule unavailable for {year} R{round_number} ({event_name})"
                )
                return None

            logger.warning(
                f"Attempt {attempt+1}/{max_retries} failed for {year} R{round_number}: {msg}"
            )
            if attempt == max_retries - 1:
                logger.error(f"Giving up on {year} R{round_number}: {msg}")
                return None

    return None


# Task: Iterate over all race events in a given year and call load_session for each.
# It aggregates per-race data and tracks which rounds failed or lacked data.
@task
def ingest_season(year: int) -> Dict:
    """Ingest all race events for a given season from FastF1."""
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info(f"INGESTING SEASON {year}")
    logger.info("=" * 60)

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        logger.error(f"Failed to get schedule for {year}: {e}")
        return {"year": year, "data": [], "successful": 0, "total": 0, "failed": []}

    races = schedule[schedule["EventFormat"] != "testing"].copy()
    races = races.sort_values("RoundNumber")
    logger.info(f"{year}: {len(races)} race events")

    all_data: List[Dict] = []
    failed: List[str] = []

    for _, ev in races.iterrows():
        rnd = int(ev["RoundNumber"])
        name = ev["EventName"]

        time.sleep(2.0)

        session_data = load_session(year, rnd, name)
        if session_data is None:
            failed.append(f"R{rnd} - {name}")
        else:
            all_data.append(session_data)

    logger.info(f"Season {year}: {len(all_data)}/{len(races)} races loaded")
    if failed:
        logger.warning("Failed or unavailable races:")
        for r in failed:
            logger.warning(f"  - {r}")

    return {
        "year": year,
        "data": all_data,
        "successful": len(all_data),
        "total": len(races),
        "failed": failed,
    }


# Task: Combine all seasons' raw laps/results and persist them to DuckDB tables.
# It backs up any existing database file before recreating the raw tables.
@task
def store_raw_data(all_season_data: List[Dict], db_path: str) -> None:
    """Concatenate and store raw laps & results into DuckDB."""
    logger = get_run_logger()
    db_path = Path(db_path)

    if db_path.exists():
        backup = db_path.with_suffix(".backup.duckdb")
        if backup.exists():
            backup.unlink()
        db_path.rename(backup)
        logger.info(f"Existing DB backed up to {backup}")

    all_laps = []
    all_results = []

    for season_info in all_season_data:
        for s in season_info["data"]:
            all_laps.append(s["laps"])
            all_results.append(s["results"])

    if not all_laps:
        logger.error("No laps collected – nothing to write.")
        return

    laps_df = pd.concat(all_laps, ignore_index=True)
    results_df = pd.concat(all_results, ignore_index=True)

    logger.info(f"Writing DuckDB: {len(laps_df):,} laps, {len(results_df):,} results")
    con = duckdb.connect(str(db_path))

    con.execute("DROP TABLE IF EXISTS raw_laps")
    con.execute("DROP TABLE IF EXISTS raw_results")

    con.execute("CREATE TABLE raw_laps AS SELECT * FROM laps_df")
    con.execute("CREATE TABLE raw_results AS SELECT * FROM results_df")

    n_laps = con.execute("SELECT COUNT(*) FROM raw_laps").fetchone()[0]
    n_res = con.execute("SELECT COUNT(*) FROM raw_results").fetchone()[0]
    logger.info(f"DB summary: raw_laps={n_laps:,}, raw_results={n_res:,}")

    con.close()


# Task: Run ingestion over all configured seasons and summarize success rates.
# It then calls store_raw_data to persist everything into the DuckDB database.
@task
def ingest_all_sessions(config: Dict) -> None:
    """Run ingestion for all seasons and store raw data."""
    logger = get_run_logger()
    seasons = config["seasons"]
    logger.info("#" * 60)
    logger.info(f"STAGE 1: INGESTION for seasons {seasons}")
    logger.info("#" * 60)

    all_season_data: List[Dict] = []
    total_ok = 0
    total_races = 0

    for year in seasons:
        sdata = ingest_season(year)
        all_season_data.append(sdata)
        total_ok += sdata["successful"]
        total_races += sdata["total"]

    rate = (100 * total_ok / total_races) if total_races else 0
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)
    for s in all_season_data:
        if s["total"] > 0:
            r = 100 * s["successful"] / s["total"]
            logger.info(f"  {s['year']}: {s['successful']}/{s['total']} races ({r:.1f}%)")
    logger.info(f"Overall: {total_ok}/{total_races} races ({rate:.1f}%)")

    store_raw_data(all_season_data, config["db_path"])
    logger.info("✓ Stage 1 complete")



@task
def run_clean_and_standardize(config: Dict) -> None:
    """Prefect task wrapper to run cleaning & standardization via transform.py."""
    logger = get_run_logger()
    logger.info("#" * 60)
    logger.info("STAGE 2: CLEANING & STANDARDIZATION")
    logger.info("#" * 60)

    db_path = config["db_path"]
    create_dimension_tables(db_path, config)
    clean_results(db_path, config)
    clean_laps(db_path, config)

    logger.info("✓ Stage 2 complete")


@task
def run_build_features(config: Dict) -> None:
    """Prefect task wrapper to run feature engineering via transform.py."""
    logger = get_run_logger()
    logger.info("#" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("#" * 60)

    db_path = config["db_path"]
    build_stint_metrics(db_path, config)
    build_driver_race_metrics(db_path, config)
    build_team_race_metrics(db_path, config)

    logger.info("✓ Stage 3 complete")

# Flow: End-to-end Prefect pipeline orchestrating ingestion, cleaning, and features.
# It logs timing, runs all stages, and prints final table row counts for sanity checks.
@flow(name="f1-data-pipeline", log_prints=True)
def f1_data_pipeline():
    logger = get_run_logger()

    config = load_config()
    setup_environment(config)

    start = datetime.now()
    logger.info("=" * 60)
    logger.info("F1 DATA PIPELINE STARTED")
    logger.info(f"Timestamp: {start}")
    logger.info("=" * 60)

    try:
        ingest_all_sessions(config)
        run_clean_and_standardize(config)
        run_build_features(config)

        end = datetime.now()
        duration = end - start
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration}")
        logger.info(f"Database: {config['db_path']}")
        logger.info("=" * 60)

        # Final table counts
        con = duckdb.connect(config["db_path"])
        tables = [
            "raw_laps",
            "raw_results",
            "clean_laps",
            "clean_results",
            "stint_metrics",
            "driver_race_metrics",
            "team_race_metrics",
        ]
        for t in tables:
            try:
                c = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                logger.info(f"{t}: {c:,} rows")
            except Exception:
                logger.warning(f"{t}: not found")
        con.close()

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    print("Starting F1 Data Pipeline with Prefect...")
    
    # Setup logging before running pipeline
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    log_dir = cfg.get('log_dir', './data/logs')
    setup_logging(log_dir)
    
    f1_data_pipeline()