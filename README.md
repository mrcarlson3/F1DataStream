# F1 Data Pipeline: Race Pace & Strategy Analysis (2019-2024)

A comprehensive data engineering and analysis project using **FastF1** to analyze Formula 1 race performance, tyre strategy, and degradation patterns across 6 seasons (2019-2024). Built with **Prefect** orchestration, **DuckDB** storage, and production-grade error handling.





### Database Schema

```sql
-- Raw tables
raw_laps          -- All lap records from FastF1
raw_results       -- Race results from FastF1

-- Dimension tables
dim_driver        -- Driver reference
dim_team          -- Team reference (normalized names)

-- Clean tables
clean_laps        -- Validated laps with time conversions
clean_results     -- Standardized race results

-- Analytical features
stint_metrics           -- Stint-level degradation slopes
driver_race_metrics     -- Driver pace, pit timing, compound usage
team_race_metrics       -- Team pace delta, strategy patterns
```


### Running the Pipeline

```bash
# Run full ETL pipeline (takes 30-60 minutes)
python src/pipeline.py
```

This will:
1. Fetch all race sessions from FastF1 API
2. Store ~100k+ lap records in `data/f1.duckdb`
3. Generate analytical features
4. Log everything to `data/logs/pipeline_*.log`

**Monitor progress:** Check `data/logs/` for real-time execution logs

### Running Analysis

```bash
# Generate all visualizations
python src/analysis.py
```

Outputs (saved to `data/figures/`):
- `pace_evolution.png` - Team pace trends across seasons
- `strategy_patterns.png` - Pit timing and compound usage
- `degradation_analysis.png` - Tyre wear vs results
- `anomaly_case_study.png` - Notable race outliers

---



## 🛠️ Error Handling & Logging

### Robust Error Management

- **Task-level retries**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Session failures**: Logged and skipped gracefully (no pipeline crash)
- **Data validation**: Lap time outliers filtered (60s < time < 200s)
- **API rate limits**: Handled by FastF1 caching layer



