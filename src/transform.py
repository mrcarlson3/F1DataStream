import duckdb
import pandas as pd
import numpy as np
from typing import Dict


# these are all the functions for the ingestion to clean and transform the data

def create_dimension_tables(db_path: str, config: Dict) -> None:
    """
    Build basic dimension tables for drivers and teams from raw_results.
    It also standardizes team names according to the configured mapping.
    """
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

    con.close()


def clean_results(db_path: str, config: Dict) -> None:
    """
    Create a cleaned race-results fact table filtered to teams of interest.
    It casts key fields to numeric types and applies team name standardization.
    """
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

    con.close()


def clean_laps(db_path: str, config: Dict) -> None:
    """
    Filter and enrich lap data, computing numeric time fields and pit flags.
    Writes a compact clean_laps table with only analytics-relevant columns.
    """
    max_lap_time = config.get("max_lap_time_seconds", 200)
    min_lap_time = config.get("min_lap_time_seconds", 60)

    con = duckdb.connect(db_path)
    laps_df = con.execute("SELECT * FROM raw_laps").df()

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

    laps_df = laps_df[
        (laps_df["LapTime_s"].notna())
        & (laps_df["LapTime_s"] >= min_lap_time)
        & (laps_df["LapTime_s"] <= max_lap_time)
    ]

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

    con.close()


# ============================================================================
# STAGE 3: FEATURE ENGINEERING (PURE FUNCTIONS)
# ============================================================================

def build_stint_metrics(db_path: str, config: Dict) -> None:
    """
    Derive lean per-stint metrics using clean_laps.
    Keeps stint length, compound, median pace and degradation slope.
    """
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
                slope, _ = np.polyfit(x_clean, y_clean, 1)
            except Exception:
                slope = 0.0
        else:
            slope = 0.0

        stint_metrics.append(
            {
                "Year": year,
                "Round": rnd,
                "DriverNumber": driver,
                "Stint": stint,
                "compound": compound,
                "stint_length_laps": stint_laps,
                "stint_median_lap_time": group["LapTime_s"].median(),
                "degradation_slope": slope,
            }
        )

    stint_df = pd.DataFrame(stint_metrics)
    con.execute("DROP TABLE IF EXISTS stint_metrics")
    con.execute("CREATE TABLE stint_metrics AS SELECT * FROM stint_df")

    con.close()


def build_driver_race_metrics(db_path: str, config: Dict) -> None:
    """
    Build minimal per-driver race-level metrics combining pace, pits and tyre usage.
    Stores median race pace, laps, pit count, first stop, and compound fractions.
    """
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
            MIN(LapNumber) AS first_stop_lap
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

    con.close()


def build_team_race_metrics(db_path: str, config: Dict) -> None:
    """
    Aggregate driver_race_metrics into lean per-team race metrics.
    Computes relative pace, strategy and points summaries at the team level.
    """
    con = duckdb.connect(db_path)

    team_metrics = con.execute("""
        WITH base AS (
            SELECT
                Year, Round, EventName, team_name,
                AVG(avg_race_pace) AS team_avg_pace,
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
        ),
        best AS (
            SELECT
                Year,
                Round,
                MIN(team_avg_pace) AS best_pace_in_race
            FROM base
            GROUP BY Year, Round
        )
        SELECT
            b.Year,
            b.Round,
            b.EventName,
            b.team_name,
            b.team_avg_pace,
            b.team_avg_pit_stops,
            b.team_avg_first_stop,
            b.team_fraction_soft,
            b.team_fraction_medium,
            b.team_fraction_hard,
            b.team_avg_finish,
            b.team_points,
            b.drivers_finished,
            (b.team_avg_pace - be.best_pace_in_race) AS pace_delta
        FROM base b
        JOIN best be
          ON b.Year = be.Year
         AND b.Round = be.Round
    """).df()

    con.execute("DROP TABLE IF EXISTS team_race_metrics")
    con.execute("CREATE TABLE team_race_metrics AS SELECT * FROM team_metrics")

    con.close()