import time
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

warnings.filterwarnings("ignore", category=FutureWarning)

# NumPy 2.0 compatibility
if not hasattr(np, "NaN"):
    np.NaN = np.nan


# ============================================================================
# CONFIG & ENV
# ============================================================================

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


@task
def setup_environment(config: Dict) -> None:
    """Enable FastF1 cache and ensure DB directory exists."""
    logger = get_run_logger()

    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Recommended way from docs: enable cache once at startup
    # fastf1.Cache.enable_cache('path/to/cache') 
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info(f"FastF1 cache enabled at: {cache_dir}")

    db_path = Path(config["db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"DuckDB database path: {db_path}")


# ============================================================================
# STAGE 1: INGESTION
# ============================================================================

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

            # Be gentle to backend / avoid hitting hard rate limits. ([docs.fastf1.dev](https://docs.fastf1.dev/_modules/fastf1/req.html?utm_source=openai))
            time.sleep(1.0)

            # Per docs: get_session(year, gp, identifier) where gp can be round number
            # and identifier can be 'R' for Race. ([docs.fastf1.dev](https://docs.fastf1.dev/_modules/fastf1/events.html?utm_source=openai))
            session = fastf1.get_session(year, round_number, "R")

            # Recommended to load all relevant data at once. Telemetry is optional for ETL. ([docs.fastf1.dev](https://docs.fastf1.dev/_modules/fastf1/core.html))
            session.load(
                laps=True,
                telemetry=False,
                weather=False,
                messages=True,
            )

            # After a successful `load`, `laps` and `results` must be
            # available, or DataNotLoadedError / NoLapDataError is raised.
            # These errors explicitly mean "no usable data". ([docs.fastf1.dev](https://docs.fastf1.dev/_modules/fastf1/core.html?utm_source=openai))
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
            # Hard rate limit: back off more aggressively, then retry. ([docs.fastf1.dev](https://docs.fastf1.dev/fastf1.html?utm_source=openai))
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
            # According to the docs, this explicitly means the API returned no
            # usable timing / results for this session. ([docs.fastf1.dev](https://docs.fastf1.dev/_modules/fastf1/core.html?utm_source=openai))
            logger.warning(
                f"No usable data for {year} R{round_number} ({event_name}): {e}"
            )
            # No point retrying: the underlying data is missing or unprocessable
            return None

        except Exception as e:
            msg = str(e)

            # Completely missing schedule (should be rare for 2018+). ([docs.fastf1.dev](https://docs.fastf1.dev/_modules/fastf1/events.html?utm_source=openai))
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


@task
def ingest_season(year: int) -> Dict:
    """Ingest all race events for a given season from FastF1."""
    logger = get_run_logger()
    logger.info("=" * 60)
    logger.info(f"INGESTING SEASON {year}")
    logger.info("=" * 60)

    try:
        # Per docs: get_event_schedule(year) returns a DataFrame-like schedule. ([docs.fastf1.dev](https://docs.fastf1.dev/fastf1.html?utm_source=openai))
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


# ============================================================================
# STAGE 2: CLEANING & DIMENSIONS
# ============================================================================

@task
def create_dimension_tables(config: Dict) -> None:
    logger = get_run_logger()
    db_path = config["db_path"]
    team_mapping = config.get("team_mapping", {})

    con = duckdb.connect(db_path)

    con.execute("""
        CREATE OR REPLACE TABLE dim_driver AS
        SELECT DISTINCT
            DriverNumber AS driver_number,
            Abbreviation AS driver_code,
            FullName AS driver_name
        FROM raw_results
        WHERE DriverNumber IS NOT NULL
    """)

    con.execute("""
        CREATE OR REPLACE TABLE dim_team AS
        SELECT DISTINCT TeamName AS team_name
        FROM raw_results
        WHERE TeamName IS NOT NULL
    """)

    for old_name, new_name in team_mapping.items():
        con.execute(
            "UPDATE dim_team SET team_name = ? WHERE team_name = ?",
            [new_name, old_name],
        )

    driver_count = con.execute("SELECT COUNT(*) FROM dim_driver").fetchone()[0]
    team_count = con.execute("SELECT COUNT(*) FROM dim_team").fetchone()[0]
    logger.info(f"dim_driver: {driver_count} rows, dim_team: {team_count} rows")

    con.close()


@task
def clean_results(config: Dict) -> None:
    logger = get_run_logger()
    db_path = config["db_path"]
    teams_of_interest = config.get("teams_of_interest", [])
    team_mapping = config.get("team_mapping", {})

    con = duckdb.connect(db_path)

    all_team_names = teams_of_interest + list(team_mapping.keys())
    placeholders = ", ".join(["?"] * len(all_team_names)) if all_team_names else "'__none__'"

    query = f"""
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
        {"WHERE TeamName IN (" + placeholders + ")" if all_team_names else ""}
          AND DriverNumber IS NOT NULL
    """
    if all_team_names:
        con.execute(query, all_team_names)
    else:
        con.execute(query)

    for old_name, new_name in team_mapping.items():
        con.execute(
            "UPDATE clean_results SET team_name = ? WHERE team_name = ?",
            [new_name, old_name],
        )

    count = con.execute("SELECT COUNT(*) FROM clean_results").fetchone()[0]
    logger.info(f"clean_results: {count:,} rows")

    con.close()


@task
def clean_laps(config: Dict) -> None:
    logger = get_run_logger()
    db_path = config["db_path"]
    max_lap_time = config.get("max_lap_time_seconds", 200)
    min_lap_time = config.get("min_lap_time_seconds", 60)

    con = duckdb.connect(db_path)
    laps_df = con.execute("SELECT * FROM raw_laps").df()
    logger.info(f"Processing {len(laps_df):,} raw laps")

    # Convert time columns to seconds
    time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for col in time_cols:
        if col in laps_df.columns:
            laps_df[f"{col}_s"] = laps_df[col].apply(
                lambda x: x.total_seconds()
                if (pd.notna(x) and hasattr(x, "total_seconds"))
                else None
            )

    # Pit flags
    laps_df["is_pit_lap"] = laps_df["PitInTime"].notna() | laps_df["PitOutTime"].notna()
    laps_df["is_pit_in_lap"] = laps_df["PitInTime"].notna()
    laps_df["is_pit_out_lap"] = laps_df["PitOutTime"].notna()

    initial = len(laps_df)
    laps_df = laps_df[
        (laps_df["LapTime_s"].notna())
        & (laps_df["LapTime_s"] >= min_lap_time)
        & (laps_df["LapTime_s"] <= max_lap_time)
    ]
    filtered = initial - len(laps_df)
    if filtered > 0:
        logger.info(f"Filtered {filtered:,} laps with invalid times")

    clean_laps_df = laps_df[
        [
            "Year",
            "Round",
            "EventName",
            "DriverNumber",
            "LapNumber",
            "Stint",
            "LapTime_s",
            "Sector1Time_s",
            "Sector2Time_s",
            "Sector3Time_s",
            "Compound",
            "TyreLife",
            "TrackStatus",
            "is_pit_lap",
            "is_pit_in_lap",
            "is_pit_out_lap",
        ]
    ].copy()

    con.execute("DROP TABLE IF EXISTS clean_laps")
    con.execute("CREATE TABLE clean_laps AS SELECT * FROM clean_laps_df")

    count = con.execute("SELECT COUNT(*) FROM clean_laps").fetchone()[0]
    logger.info(f"clean_laps: {count:,} rows")

    compounds = con.execute("""
        SELECT Compound, COUNT(*) AS cnt
        FROM clean_laps
        WHERE Compound IS NOT NULL
        GROUP BY Compound
        ORDER BY cnt DESC
    """).df()
    for _, row in compounds.iterrows():
        logger.info(f"Compound {row['Compound']}: {row['cnt']:,} laps")

    con.close()


@task
def clean_and_standardize(config: Dict) -> None:
    logger = get_run_logger()
    logger.info("#" * 60)
    logger.info("STAGE 2: CLEANING & STANDARDIZATION")
    logger.info("#" * 60)

    create_dimension_tables(config)
    clean_results(config)
    clean_laps(config)

    logger.info("✓ Stage 2 complete")


# ============================================================================
# STAGE 3: FEATURE ENGINEERING
# ============================================================================

@task
def build_stint_metrics(config: Dict) -> None:
    logger = get_run_logger()
    db_path = config["db_path"]

    con = duckdb.connect(db_path)
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

    logger.info(f"Building stint metrics from {len(laps_df):,} racing laps")

    stint_metrics = []
    grouped = laps_df.groupby(["Year", "Round", "DriverNumber", "Stint"])

    for (year, rnd, driver, stint), group in grouped:
        if len(group) < 3:
            continue

        group = group.sort_values("LapNumber").copy()
        group["stint_lap_index"] = range(len(group))

        compound = group["Compound"].iloc[0]
        stint_laps = len(group)
        y = group["LapTime_s"].values
        x = group["stint_lap_index"].values

        mean_time = y.mean()
        std_time = y.std() if len(y) > 1 else 0.0

        mask = np.ones_like(y, dtype=bool)
        if std_time > 0:
            mask = np.abs(y - mean_time) <= 2 * std_time

        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) >= 3:
            try:
                slope, intercept = np.polyfit(x_clean, y_clean, 1)
            except Exception:
                slope, intercept = 0.0, mean_time
        else:
            slope, intercept = 0.0, mean_time

        stint_metrics.append(
            {
                "Year": year,
                "Round": rnd,
                "DriverNumber": driver,
                "Stint": stint,
                "compound": compound,
                "stint_length_laps": stint_laps,
                "stint_mean_lap_time": group["LapTime_s"].mean(),
                "stint_median_lap_time": group["LapTime_s"].median(),
                "stint_best_lap": group["LapTime_s"].min(),
                "stint_std": group["LapTime_s"].std(),
                "degradation_slope": slope,
                "stint_base_pace": intercept,
            }
        )

    stint_df = pd.DataFrame(stint_metrics)
    con.execute("DROP TABLE IF EXISTS stint_metrics")
    con.execute("CREATE TABLE stint_metrics AS SELECT * FROM stint_df")

    count = con.execute("SELECT COUNT(*) FROM stint_metrics").fetchone()[0]
    logger.info(f"stint_metrics: {count:,} rows")

    avg_deg = con.execute("""
        SELECT
            compound,
            AVG(degradation_slope) AS avg_deg,
            COUNT(*) AS stint_count
        FROM stint_metrics
        GROUP BY compound
        ORDER BY avg_deg DESC
    """).df()
    for _, row in avg_deg.iterrows():
        logger.info(
            f"Compound {row['compound']}: avg degradation "
            f"{row['avg_deg']:.3f} s/lap over {row['stint_count']} stints"
        )

    con.close()


@task
def build_driver_race_metrics(config: Dict) -> None:
    logger = get_run_logger()
    db_path = config["db_path"]
    con = duckdb.connect(db_path)

    pace_metrics = con.execute("""
        SELECT
            l.Year, l.Round, l.EventName,
            l.DriverNumber,
            r.driver_name,
            r.team_name,
            r.finish_position,
            r.points,
            MEDIAN(l.LapTime_s) AS avg_race_pace,
            STDDEV(l.LapTime_s) AS pace_std,
            MAD(l.LapTime_s) AS pace_mad,
            COUNT(*) AS total_laps,
            SUM(CASE WHEN l.is_pit_lap THEN 1 ELSE 0 END) AS pit_stop_count
        FROM clean_laps l
        JOIN clean_results r
          ON l.Year = r.Year
         AND l.Round = r.Round
         AND l.DriverNumber = r.driver_number
        WHERE l.is_pit_lap = FALSE
          AND l.LapTime_s IS NOT NULL
        GROUP BY
            l.Year, l.Round, l.EventName,
            l.DriverNumber,
            r.driver_name,
            r.team_name,
            r.finish_position,
            r.points
    """).df()

    pit_details = con.execute("""
        SELECT
            Year, Round, DriverNumber,
            MIN(LapNumber) AS first_stop_lap,
            MAX(LapNumber) AS last_stop_lap
        FROM clean_laps
        WHERE is_pit_lap = TRUE
        GROUP BY Year, Round, DriverNumber
    """).df()

    driver_metrics = pace_metrics.merge(
        pit_details, on=["Year", "Round", "DriverNumber"], how="left"
    )

    compound_usage = con.execute("""
        SELECT
            Year, Round, DriverNumber,
            SUM(CASE WHEN Compound = 'SOFT' THEN 1 ELSE 0 END) AS soft_laps,
            SUM(CASE WHEN Compound = 'MEDIUM' THEN 1 ELSE 0 END) AS medium_laps,
            SUM(CASE WHEN Compound = 'HARD' THEN 1 ELSE 0 END) AS hard_laps,
            COUNT(*) AS total_compound_laps
        FROM clean_laps
        WHERE Compound IS NOT NULL
          AND is_pit_lap = FALSE
        GROUP BY Year, Round, DriverNumber
    """).df()

    compound_usage["fraction_soft"] = (
        compound_usage["soft_laps"] / compound_usage["total_compound_laps"]
    )
    compound_usage["fraction_medium"] = (
        compound_usage["medium_laps"] / compound_usage["total_compound_laps"]
    )
    compound_usage["fraction_hard"] = (
        compound_usage["hard_laps"] / compound_usage["total_compound_laps"]
    )

    driver_metrics = driver_metrics.merge(
        compound_usage[
            [
                "Year",
                "Round",
                "DriverNumber",
                "fraction_soft",
                "fraction_medium",
                "fraction_hard",
            ]
        ],
        on=["Year", "Round", "DriverNumber"],
        how="left",
    )

    driver_metrics = driver_metrics.fillna(
        {
            "pit_stop_count": 0,
            "first_stop_lap": 0,
            "fraction_soft": 0,
            "fraction_medium": 0,
            "fraction_hard": 0,
        }
    )

    con.execute("DROP TABLE IF EXISTS driver_race_metrics")
    con.execute("CREATE TABLE driver_race_metrics AS SELECT * FROM driver_metrics")

    count = con.execute("SELECT COUNT(*) FROM driver_race_metrics").fetchone()[0]
    logger.info(f"driver_race_metrics: {count:,} rows")

    con.close()


@task
def build_team_race_metrics(config: Dict) -> None:
    logger = get_run_logger()
    db_path = config["db_path"]
    con = duckdb.connect(db_path)

    team_metrics = con.execute("""
        SELECT
            Year, Round, EventName, team_name,
            AVG(avg_race_pace) AS team_avg_pace,
            AVG(pace_std) AS team_pace_consistency,
            AVG(pit_stop_count) AS team_avg_pit_stops,
            AVG(first_stop_lap) AS team_avg_first_stop,
            AVG(fraction_soft) AS team_fraction_soft,
            AVG(fraction_medium) AS team_fraction_medium,
            AVG(fraction_hard) AS team_fraction_hard,
            AVG(finish_position) AS team_avg_finish,
            SUM(points) AS team_points,
            COUNT(*) AS drivers_finished
        FROM driver_race_metrics
        GROUP BY Year, Round, EventName, team_name
    """).df()

    team_metrics["best_pace_in_race"] = team_metrics.groupby(
        ["Year", "Round"]
    )["team_avg_pace"].transform("min")
    team_metrics["pace_delta"] = (
        team_metrics["team_avg_pace"] - team_metrics["best_pace_in_race"]
    )

    con.execute("DROP TABLE IF EXISTS team_race_metrics")
    con.execute("CREATE TABLE team_race_metrics AS SELECT * FROM team_metrics")

    count = con.execute("SELECT COUNT(*) FROM team_race_metrics").fetchone()[0]
    logger.info(f"team_race_metrics: {count:,} rows")

    summary = con.execute("""
        SELECT
            team_name,
            COUNT(*) AS races,
            AVG(pace_delta) AS avg_pace_delta,
            AVG(team_points) AS avg_points_per_race
        FROM team_race_metrics
        GROUP BY team_name
        ORDER BY avg_pace_delta
    """).df()
    for _, row in summary.iterrows():
        logger.info(
            f"{row['team_name']}: {row['races']} races, "
            f"pace Δ={row['avg_pace_delta']:.3f}s, "
            f"avg pts={row['avg_points_per_race']:.1f}"
        )

    con.close()


@task
def build_features(config: Dict) -> None:
    logger = get_run_logger()
    logger.info("#" * 60)
    logger.info("STAGE 3: FEATURE ENGINEERING")
    logger.info("#" * 60)

    build_stint_metrics(config)
    build_driver_race_metrics(config)
    build_team_race_metrics(config)

    logger.info("✓ Stage 3 complete")


# ============================================================================
# MAIN FLOW
# ============================================================================

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
        clean_and_standardize(config)
        build_features(config)

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
    f1_data_pipeline()