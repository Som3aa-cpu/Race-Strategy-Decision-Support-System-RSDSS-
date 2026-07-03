import logging
import sys
from pathlib import Path

import pandas as pd


# Configuration

MASTER_PATH   = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\Master\Master_Dataset.csv")
CIRCUITS_PATH = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Refrences\Circuit_Characteristics.csv")
OUTPUT_PATH   = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\Master\enriched_dataset.csv")

# The column used to join the two datasets
JOIN_KEY = "Circuit"


# Logging

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rsdss.enrich")
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


# Step 1 — Load master dataset

def load_master(path: Path = MASTER_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master dataset not found : {path}")

    df = pd.read_csv(path, low_memory=False)
    log.info("Master dataset loaded  : %s rows × %s cols  (%s)",
             f"{len(df):,}", df.shape[1], path.name)
    return df


# Step 2 — Load circuits metadata

def load_circuits(path: Path = CIRCUITS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Circuits file not found : {path}\n"
            f"Create a CSV with at least a '{JOIN_KEY}' column and any "
            f"additional metadata columns (Country, CircuitLength_km, etc.)."
        )

    circuits = pd.read_csv(path)

    if JOIN_KEY not in circuits.columns:
        raise ValueError(
            f"circuits.csv must contain a '{JOIN_KEY}' column to join on."
        )

    # Drop duplicates in case the CSV has repeated circuit entries
    circuits = circuits.drop_duplicates(subset=[JOIN_KEY])
    log.info("Circuits metadata loaded : %d circuits  (%d columns)  (%s)",
             len(circuits), circuits.shape[1], path.name)
    log.debug("Columns : %s", list(circuits.columns))
    return circuits


# Step 3 — Merge

def merge_datasets(master: pd.DataFrame, circuits: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join the master dataset with circuit metadata on the Circuit column.
    Uses a left join so that no lap rows are lost even if a circuit is
    missing from circuits.csv — those rows will have NaN for the metadata
    columns, which step 4 will flag explicitly.
    """
    cols_before = set(master.columns)
    enriched    = master.merge(circuits, on=JOIN_KEY, how="left")
    cols_added  = [c for c in enriched.columns if c not in cols_before]

    log.info("Merge complete — %d new column(s) added : %s",
             len(cols_added), cols_added)
    return enriched


# Step 4 — Check for missing circuits

def check_missing_circuits(enriched: pd.DataFrame, circuits: pd.DataFrame) -> None:
    """
    Warn about any Circuit values in the master dataset that are not
    present in circuits.csv.
    These rows will have NaN values for all metadata columns after the merge,
    which could cause issues in downstream feature engineering.
    """
    known_circuits    = set(circuits[JOIN_KEY].unique())
    master_circuits   = set(enriched[JOIN_KEY].dropna().unique())
    missing_circuits  = master_circuits - known_circuits

    if not missing_circuits:
        log.info("Circuit check passed — all %d circuits found in circuits.csv.",
                 len(master_circuits))
        return

    # Count affected rows per missing circuit
    log.warning("─" * 50)
    log.warning("  %d circuit(s) missing from circuits.csv :", len(missing_circuits))
    for circuit in sorted(missing_circuits):
        affected_rows = (enriched[JOIN_KEY] == circuit).sum()
        log.warning("    %-30s — %s rows affected", circuit, f"{affected_rows:,}")
    log.warning("  These rows will have NaN for all metadata columns.")
    log.warning("  Add them to circuits.csv to resolve this.")
    log.warning("─" * 50)


# Step 5 — Save enriched dataset

def save_enriched(enriched: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(path, index=False)

    size_mb = path.stat().st_size / (1024 * 1024)
    log.info("Enriched dataset saved  : %s rows × %s cols  |  %.2f MB  →  %s",
             f"{len(enriched):,}", enriched.shape[1], size_mb, path)


# the running function 

def run_enrichment(
    master_path:   Path = MASTER_PATH,
    circuits_path: Path = CIRCUITS_PATH,
    output_path:   Path = OUTPUT_PATH,
) -> pd.DataFrame:
    log.info("=" * 50)
    log.info("  RSDSS — Data Enrichment")
    log.info("  Master   : %s", master_path)
    log.info("  Circuits : %s", circuits_path)
    log.info("  Output   : %s", output_path)
    log.info("=" * 50)

    master   = load_master(master_path)
    circuits = load_circuits(circuits_path)
    enriched = merge_datasets(master, circuits)

    check_missing_circuits(enriched, circuits)

    save_enriched(enriched, output_path)

    log.info("=" * 50)
    log.info("  Enrichment complete.")
    log.info("=" * 50)

    return enriched


# ENTRY POINT

if __name__ == "__main__":
    run_enrichment()