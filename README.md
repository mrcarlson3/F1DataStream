# F1 Data Pipeline

A data engineering project that analyzes Formula 1 race performance using the FastF1 API. The pipeline ingests, cleans, and analyzes lap timing data to understand team pace evolution, tyre strategy patterns, and race performance trends.

The data comes from the **FastF1** library, which provides access to official Formula 1 timing data including:

- **Lap times**: Every lap completed by every driver in every race
- **Tyre compounds**: Which tyre type (Soft, Medium, Hard) was used
- **Pit stops**: When drivers stopped and how long they stayed
- **Race results**: Final positions, points, and DNFs
- **Driver/Team info**: Names, numbers, and team affiliations

This data covers the **2019-2023 seasons** 

## Pipeline Architecture

### `pipeline.py` - ETL Pipeline

The main data pipeline that handles three stages:

**Stage 1: Ingestion**
- Fetches race data from FastF1 API for all configured seasons
- Loads lap timing and race results for each Grand Prix
- Implements retry logic with exponential backoff for API failures
- Stores raw data in DuckDB tables: `raw_laps` and `raw_results`

**Stage 2: Cleaning**
- Converts lap times from timedelta objects to seconds (float)
- Flags pit laps and filters outliers (>200s lap times)
- Normalizes team names across seasons (e.g., "Red Bull" → "Red Bull Racing")
- Creates dimension tables for drivers and teams
- Produces: `clean_laps`, `clean_results`, `dim_driver`, `dim_team`

**Stage 3: Feature Engineering**
- **Stint metrics**: Calculates tyre degradation slopes using linear regression (seconds lost per lap)
- **Driver metrics**: Aggregates pace statistics, pit timing, and compound usage per driver per race
- **Team metrics**: Computes team-level pace deltas relative to fastest team, strategy patterns
- Produces: `stint_metrics`, `driver_race_metrics`, `team_race_metrics`

### `analysis.py` - Visualization & Analysis

All figures are saved to `data/figures/` as high-resolution PNG files.

## Technology Stack

### Prefect (Workflow Orchestration)
- **Tasks**: Each function (`load_session`, `clean_laps`, etc.) is a `@task` that can be monitored
- **Flows**: The main `f1_data_pipeline()` is a `@flow` that orchestrates all tasks
- **Logging**: Automatic logging to track progress and debug failures
- **Retry logic**: Built-in retry mechanisms for API failures
- **Remote tracking**: Connects to `sds-prefect.pods.uvarc.io` to track runs (local execution)

### DuckDB (Analytical Database)
- **Columnar storage**: Optimized for analytical queries on large datasets
- **Embedded database**: No separate server needed - stored as `data/f1.duckdb`
- **SQL interface**: Query data using standard SQL
- **7 tables**: Raw data → cleaned data → analytical features
- **Fast aggregations**: Compute team/driver statistics efficiently





