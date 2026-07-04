import logging
import sys
from pathlib import Path

import pandas as pd

from Race_Features       import add_race_features
from Lap_Features        import add_lap_features
from Tire_Features       import add_tire_features
from Circuit_Features    import add_circuit_features
from Temperature_Features import add_temperature_features


# Configuration 

INPUT_PATH     = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\Master\enriched_dataset.csv")
OUTPUT_PATH    = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\Master\featured_dataset.csv")

# Only compute features on valid laps.
# Invalid laps (ValidLap=False) are kept in the output but their
# feature columns will be NaN — they are excluded from computation
# to avoid degradation estimates being skewed by red-flag laps.
VALID_ONLY = True


# Logging

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rsdss.pipeline")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.propagate = False
    return logger

log = setup_logger()


# Helpers

def _log_step(step: int, name: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """Log how many columns were added by each module."""
    added = [c for c in df_after.columns if c not in df_before.columns]
    log.info("  Step %d — %-25s  +%d columns : %s", step, name, len(added), added)


# Pipeline

def run_feature_pipeline(
    input_path:  Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    valid_only:  bool = VALID_ONLY,
) -> pd.DataFrame:
    """
    Load the enriched dataset, apply all feature modules, save the result.
    """

    log.info("  RSDSS — Feature Engineering Pipeline")
    log.info("  Input  : %s", input_path)
    log.info("  Output : %s", output_path)
 

    # Load
    if not input_path.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found : {input_path}\n"
            "Run enrich_data.py first."
        )

    full_df = pd.read_csv(input_path, low_memory=False)
    log.info("Loaded : %s rows x %s columns", f"{len(full_df):,}", full_df.shape[1])

    # Split valid / invalid if requested
    if valid_only and "ValidLap" in full_df.columns:
        valid_df   = full_df[full_df["ValidLap"] == True].copy().reset_index(drop=True)
        invalid_df = full_df[full_df["ValidLap"] != True].copy().reset_index(drop=True)
        log.info(
            "Split  : %s valid laps | %s invalid laps (features will be NaN)",
            f"{len(valid_df):,}", f"{len(invalid_df):,}",
        )
        working_df = valid_df
    else:
        invalid_df = pd.DataFrame()
        working_df = full_df.copy()

    # Apply feature modules in order
    log.info("  Applying feature modules...")
    steps = [
        (1, "race_features",        add_race_features),
        (2, "lap_features",         add_lap_features),
        (3, "tire_features",        add_tire_features),
        (4, "circuit_features",     add_circuit_features),
        (5, "temperature_features", add_temperature_features),
    ]

    for step, name, fn in steps:
        before = working_df.copy()
        working_df = fn(working_df)
        _log_step(step, name, before, working_df)

    # Re-attach invalid laps
    if not invalid_df.empty:
        featured_df = pd.concat([working_df, invalid_df], ignore_index=True)

        # Restore original sort order
        sort_cols = [c for c in ["Season", "RaceID", "Driver", "LapNumber"]
                     if c in featured_df.columns]
        featured_df = featured_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        featured_df = working_df

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_df.to_csv(output_path, index=False)

    size_mb       = output_path.stat().st_size / (1024 * 1024)
    cols_before   = full_df.shape[1]
    cols_after    = featured_df.shape[1]
    cols_added    = cols_after - cols_before
    new_col_names = [c for c in featured_df.columns if c not in full_df.columns]

    log.info("=" * 55)
    log.info("  Pipeline complete.")
    log.info("  Rows             : %s", f"{len(featured_df):,}")
    log.info("  Columns before   : %d", cols_before)
    log.info("  Columns added    : %d", cols_added)
    log.info("  Columns after    : %d", cols_after)
    log.info("  New columns      : %s", new_col_names)
    log.info("  File size        : %.2f MB", size_mb)
    log.info("  Saved to         : %s", output_path)
    log.info("=" * 55)

    return featured_df


# Entry point

if __name__ == "__main__":
    run_feature_pipeline()