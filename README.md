# F1 Data Pipeline

A data engineering project that analyzes Formula 1 race performance using the FastF1 API. The pipeline ingests, cleans, and analyzes lap timing data to understand team pace evolution, tyre strategy patterns, and race performance trends.

## What is F1 Data?

The data comes from the **FastF1** library, which provides access to official Formula 1 timing data including:

- **Lap times**: Every lap completed by every driver in every race
- **Tyre compounds**: Which tyre type (Soft, Medium, Hard) was used
- **Pit stops**: When drivers stopped and how long they stayed
- **Race results**: Final positions, points, and DNFs
- **Driver/Team info**: Names, numbers, and team affiliations

This data covers the **2019-2023 seasons**  and includes over 100,000 individual lap records.

---

## Project Structure

```
F1DataStream/
├── src/
│   ├── ingest.py              # Main ETL pipeline (Prefect flow)
│   ├── transform.py           # Data cleaning & feature engineering
│   ├── evolution_analysis.py  # Track-level evolution analysis
│   └── general_analysis.py    # Team/driver performance analysis
├── data/
│   ├── f1.duckdb             # DuckDB database
│   ├── logs/                 # Pipeline logs
│   └── figures/              # Generated visualizations
│       ├── pace_evolution.png
│       ├── strategy_patterns.png
│       ├── degradation_analysis.png
│       ├── anomaly_case_study.png
│       └── track_evolution_analysis.png
├── fastf1_cache/             # FastF1 API cache
├── config.yaml               # Configuration
└── requirements.txt          # Python dependencies
```

---

## Pipeline Architecture

### `ingest.py` - Main ETL Pipeline

The entry point that orchestrates the entire data pipeline using **Prefect**:

**Stage 1: Data Ingestion**
- Fetches race data from FastF1 API for all configured seasons
- Loads lap timing and race results for each Grand Prix
- Implements retry logic with exponential backoff for API failures
- Stores raw data in DuckDB tables: `raw_laps` and `raw_results`

**Stage 2: Data Cleaning** (calls `transform.py`)
- Converts lap times from timedelta objects to seconds (float)
- Flags pit laps and filters outliers (>200s lap times)
- Normalizes team names across seasons (e.g., "Red Bull" → "Red Bull Racing")
- Creates dimension tables for drivers and teams
- Produces: `clean_laps`, `clean_results`, `dim_driver`, `dim_team`

**Stage 3: Feature Engineering** (calls `transform.py`)
- **Stint metrics**: Calculates tyre degradation slopes using linear regression (seconds lost per lap)
- **Driver metrics**: Aggregates pace statistics, pit timing, and compound usage per driver per race
- **Team metrics**: Computes team-level pace deltas relative to fastest team, strategy patterns
- Produces: `stint_metrics`, `driver_race_metrics`, `team_race_metrics`

### `transform.py` - Data Processing Functions

Contains pure Python functions for data cleaning and feature engineering:

- `create_dimension_tables()` - Build driver and team reference tables
- `clean_results()` - Standardize race results with team name normalization
- `clean_laps()` - Convert times, flag pit laps, filter outliers
- `build_stint_metrics()` - Calculate tyre degradation per stint
- `build_driver_race_metrics()` - Driver-level pace and strategy aggregation
- `build_team_race_metrics()` - Team-level performance metrics

### `evolution_analysis.py` - Track Evolution Analysis

Generates track-specific evolution visualizations:

- **Track-by-track pace analysis**: How performance evolved at each circuit
- **Circuit characteristics**: Identifies fast vs. technical tracks
- **Seasonal patterns**: Year-over-year performance changes at the same venue

Output: `track_evolution_analysis.png`

### `general_analysis.py` - Performance Analysis

Generates 4 core analytical visualizations:

1. **Team Pace Evolution** (`pace_evolution.png`) - Season-long pace trends
2. **Strategy Patterns** (`strategy_patterns.png`) - Pit stop timing and compound usage
3. **Degradation Analysis** (`degradation_analysis.png`) - Tyre wear vs race results
4. **Anomaly Case Study** (`anomaly_case_study.png`) - Statistical outliers in lap times

---

## Technology Stack

### Prefect (Workflow Orchestration)
- **Tasks**: Each function (`load_session`, `clean_laps`, etc.) is a `@task` that can be monitored
- **Flows**: The main pipeline in `ingest.py` is a `@flow` that orchestrates all tasks
- **Logging**: Automatic logging to track progress and debug failures
- **Retry logic**: Built-in retry mechanisms for API failures
- **Remote tracking**: Connects to `sds-prefect.pods.uvarc.io` to track runs (local execution)

### DuckDB (Analytical Database)
- **Columnar storage**: Optimized for analytical queries on large datasets
- **Embedded database**: No separate server needed - stored as `data/f1.duckdb`
- **SQL interface**: Query data using standard SQL
- **7 tables**: Raw data → cleaned data → analytical features
- **Fast aggregations**: Compute team/driver statistics efficiently

---

## How to Run

```bash
# 1. Run the ETL pipeline (ingestion + cleaning + features)
python src/ingest.py

# 2. Generate performance visualizations
python src/general_analysis.py

# 3. Generate track evolution analysis
python src/evolution_analysis.py
```


## Configuration

Edit `config.yaml` to adjust:
- `seasons`: Which years to analyze (currently `[2019, 2023]`)
- `teams_of_interest`: Which teams to include in analysis
- `max_lap_time_seconds`: Outlier threshold for lap time filtering

## Database Schema

**Raw Tables:**
- `raw_laps` - All lap records from FastF1
- `raw_results` - Race results from FastF1

**Dimension Tables:**
- `dim_driver` - Driver reference (number, code, name)
- `dim_team` - Team reference (normalized names)

**Clean Tables:**
- `clean_laps` - Validated laps with time conversions
- `clean_results` - Standardized race results

**Analytical Features:**
- `stint_metrics` - Stint-level degradation slopes
- `driver_race_metrics` - Driver pace, pit timing, compound usage
- `team_race_metrics` - Team pace delta, strategy patterns


