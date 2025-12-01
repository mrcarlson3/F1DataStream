# F1 Data Pipeline

A small data engineering project that uses the **FastF1** library to pull official Formula 1 timing data and build a compact analytical dataset in **DuckDB**.

The pipeline focuses on:

- Lap time and race pace evolution
- Tyre strategy (stints, degradation, compound usage)
- Driver and team race performance

Configured seasons (via `config.yaml`) currently cover **2019–2023**.

---

## Project Structure

- `ingest.py` – **main entry point** (run this)
- `transform.py` – data cleaning and feature engineering logic (called by `ingest.py`)
- `analysis.py` – optional visualizations and ad‑hoc analysis
- `config.yaml` – seasons, paths, and basic parameters

---

## Pipeline Overview

### 1. Ingestion (`ingest.py` – Stage 1)

Implemented with **Prefect** tasks and flows.

- Reads configuration from `config.yaml`
- Enables FastF1 cache and prepares directories
- For each configured season:
  - Loads race schedule via FastF1
  - For each race:
    - Loads the race session (`fastf1.get_session(year, round, "R")`)
    - Loads laps and results with retry & backoff for API/rate-limit errors
  - Skips test events and races without valid timing data
- Stores raw data in a local **DuckDB** database:

Tables created:
- `raw_laps`
- `raw_results`

---

### 2. Cleaning & Standardization (`transform.py` – Stage 2)

Pure Python + DuckDB SQL, called from Prefect tasks in `ingest.py`.

Key steps:

- **Dimension tables**
  - `dim_driver`: driver number, code, and name
  - `dim_team`: team names (normalized across seasons)

- **Result cleaning (`clean_results`)**
  - Filters to teams of interest defined in `config.yaml`
  - Normalizes team names (e.g., `"Red Bull"` → `"Red Bull Racing"`)
  - Casts positions and points to numeric types
  - Produces a tidy race-results table per driver per race

- **Lap cleaning (`clean_laps`)**
  - Converts lap and sector times from timedeltas to seconds
  - Flags:
    - `is_pit_lap`
    - `is_pit_in_lap`
    - `is_pit_out_lap`
  - Filters obvious outliers using configurable min/max lap time thresholds
  - Selects analytics-relevant columns only

Tables produced:
- `clean_laps`
- `clean_results`
- `dim_driver`
- `dim_team`

---

### 3. Feature Engineering (`transform.py` – Stage 3)

Also pure functions, invoked from `ingest.py`.

- **Stint metrics (`stint_metrics`)**
  - Groups laps by driver × stint
  - Filters out pit laps
  - Computes:
    - Stint length (laps)
    - Tyre compound
    - Median stint lap time
    - Degradation slope (seconds per lap via simple linear fit)

- **Driver race metrics (`driver_race_metrics`)**
  - Joins `clean_laps` with `clean_results`
  - Computes per driver × race:
    - Median race pace (non‑pit laps)
    - Total laps
    - Pit-stop count and first stop lap
    - Final position and points
    - Fraction of race distance on Soft / Medium / Hard

- **Team race metrics (`team_race_metrics`)**
  - Aggregates driver metrics to team level per race
  - Computes:
    - Average team race pace
    - Average pit stops and first stop lap
    - Compound usage fractions
    - Average finishing position
    - Total team points
    - Pace delta vs. fastest team in the race

Tables produced:
- `stint_metrics`
- `driver_race_metrics`
- `team_race_metrics`

---

## Technology Stack

### FastF1

- Access to official F1 timing data:
  - Lap times, sectors, compounds, pit windows
  - Race results and driver/team metadata
- Local caching to reduce API load and speed up re-runs

### Prefect

Used only in `ingest.py` to orchestrate the pipeline:

- `@task` for ingestion and high-level stage calls
- `@flow` (`f1_data_pipeline`) as the main entry point
- Logging, retries, and basic monitoring

The transform functions in `transform.py` are plain Python; no Prefect dependency there.

### DuckDB

- Embedded analytical database (single `.duckdb` file)
- Columnar storage and fast aggregations
- All stages read/write via DuckDB:

Raw → Cleaned → Features:
- `raw_laps`, `raw_results`
- `clean_laps`, `clean_results`, `dim_driver`, `dim_team`
- `stint_metrics`, `driver_race_metrics`, `team_race_metrics`

