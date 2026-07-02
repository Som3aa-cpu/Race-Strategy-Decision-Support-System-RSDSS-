import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


# Configration


RAW_DATA_DIR       = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Raw\Data\Processed\by_race")
PROCESSED_DATA_DIR = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed")
REPORTS_DIR        = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\reports")

# Max and Minimum for laps in seconds
LAP_TIME_MIN = 45.0
LAP_TIME_MAX = 250.0

# Expected Columns
REQUIRED_COLUMNS = [
    "RaceID", "RaceName", "Circuit", "Driver", "Team",
    "LapNumber", "Position", "LapTime_Seconds", "Compound", "TireAge",
    "Stint", "PitOutTime", "PitInTime", "AirTemp", "TrackTemp", "Rainfall",
]

# Reasons for exclusion 
REASON_MISSING   = "Missing Lap Time"
REASON_TOO_SHORT = "Unrealistically Short Lap"
REASON_TOO_LONG  = "Red Flag / Extremely Long Lap"
REASON_DUPLICATE = "Duplicate Row"



# Logging

def setup_logger(name: str = "rsdss") -> logging.Logger:
    """Configure et retourne un logger avec handlers console et fichier."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured
    
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%H:%M:%S")

    # Console handeler 
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # File handeler
    Path("logs").mkdir(exist_ok=True)
    file_h = logging.FileHandler("logs/rsdss_cleaning.log", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_h)
    logger.propagate = False
    return logger


log = setup_logger()



# Cleaning sample


@dataclass
class CleaningReport:
    """Statistiques produites pour un seul fichier CSV."""

    filename:         str
    rows_read:        int = 0
    duplicates:       int = 0
    missing_laptime:  int = 0
    short_laps:       int = 0
    long_laps:        int = 0

    # Computed properties no need to store them
    @property
    def rows_flagged(self) -> int:
        return self.missing_laptime + self.short_laps + self.long_laps

    @property
    def rows_removed(self) -> int:
        return self.duplicates + self.rows_flagged

    @property
    def rows_kept(self) -> int:
        # Kept = total after deduplication - flagged rows
        return (self.rows_read - self.duplicates) - self.rows_flagged

    def print_summary(self) -> None:
        """Displays a readable summary in the logs."""
        sep = "-" * 50
        log.info(sep)
        log.info(f" Report — {self.filename}")
        log.info(sep)
        log.info(f"  lines read              : {self.rows_read}")
        log.info(f"  Duplicates removed      : {self.duplicates}")
        log.info(f"  Missing lap times       : {self.missing_laptime}")
        log.info(f"  Too short laps          : {self.short_laps}")
        log.info(f"  Too long laps           : {self.long_laps}")
        log.info(sep)
        log.info(f"  Total deleted/reported  : {self.rows_removed}")
        log.info(f"  Valid lines kept        : {self.rows_kept}")
        log.info(sep)

    def to_dict(self) -> dict:
        """Serialize the report to a JSON-compatible dictionary."""
        return {
            "filename":        self.filename,
            "rows_read":       self.rows_read,
            "duplicates":      self.duplicates,
            "missing_laptime": self.missing_laptime,
            "short_laps":      self.short_laps,
            "long_laps":       self.long_laps,
            "rows_removed":    self.rows_removed,
            "rows_kept":       self.rows_kept,
        }



# I/O — reading and writing CSV files


def discover_files(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return all CSV files found in raw_dir."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")

    files = sorted(raw_dir.glob("*.csv"))
    log.info(f"{len(files)} CSV file(s) found in '{raw_dir}'")
    return files


def read_csv(file_path: Path) -> pd.DataFrame:
    """Reads a CSV and verifies that all required columns are present."""
    log.info(f"Lecture : {file_path.name}")
    df = pd.read_csv(file_path, low_memory=False)

    if df.empty:
        raise ValueError(f"Files empty : {file_path.name}")

    # Vérification des colonnes manquantes
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing in {file_path.name} : {missing_cols}")

    log.debug(f"  → {len(df)} lines × {len(df.columns)} columns loaded")
    return df


def inject_season(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Derives the Season from the filename and injects it as a column.
 
    The last 2 characters of the stem are the 2-digit year
    (e.g. BHR22 → 2022, AUS25 → 2025).
    If parsing fails the column is left as None so the rest of the
    pipeline still runs.
    """
    stem = Path(filename).stem   # ex: "BHR22"
    try:
        year_suffix = int(stem[-2:])              # 22, 23, 24, 25...
        season = 2000 + year_suffix               # 2022, 2023...
    except (ValueError, IndexError):
        season = None
        log.warning(f"  Unable to derive the season from '{filename}'")
 
    # Insère Season juste après RaceID pour un ordre logique
    insert_at = df.columns.get_loc("RaceID") + 1 if "RaceID" in df.columns else 0
    df.insert(insert_at, "Season", season)
    log.debug(f"  Column Season = {season} injected from the file name.")
    return df

def write_csv(df: pd.DataFrame, source_file: Path,
              out_dir: Path = PROCESSED_DATA_DIR) -> Path:
    """Saves the cleaned DataFrame in out_dir with the same file name."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source_file.name
    df.to_csv(out_path, index=False)
    log.info(f"  → Saved : {out_path}  ({len(df)} lines)")
    return out_path



# RÈGLES DE NETTOYAGE — une fonction par règle


def add_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the ValidLap and ExclusionReason columns if they don't exist."""
    if "ValidLap" not in df.columns:
        df["ValidLap"] = True
    if "ExclusionReason" not in df.columns:
        df["ExclusionReason"] = ""
    return df


def standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the columns to their correct types (int, float, string)."""
    # Colonnes numériques entières — nullable pour tolérer les NaN
    for col in ["Season", "LapNumber", "Position", "Stint"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Colonnes numériques décimales
    for col in ["LapTime_Seconds", "TireAge", 
                "AirTemp", "TrackTemp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["PitOutTime", "PitInTime"]:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col], errors="coerce")

    # Colonnes texte
    for col in ["RaceID", "RaceName", "Circuit", "Driver", "Team", "Compound"]:
        if col in df.columns:
            df[col] = df[col].astype("string")

    log.debug("  Types standardisés.")
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Removes physically the exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    count = before - len(df)
    if count:
        log.debug(f"  {count} duplicate(s) removed.")
    return df, count


def flag_missing_lap_times(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Signale les lignes où LapTime_Seconds est NaN."""
    mask = df["LapTime_Seconds"].isna() & df["ValidLap"]
    count = int(mask.sum())
    if count:
        df.loc[mask, "ValidLap"] = False
        df.loc[mask, "ExclusionReason"] = REASON_MISSING
        log.debug(f"  {count} lap(s) with missing time flagged.")
    return df, count


def flag_short_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Flags laps shorter than LAP_TIME_MIN seconds."""
    mask = (df["LapTime_Seconds"].notna()
            & (df["LapTime_Seconds"] < LAP_TIME_MIN)
            & df["ValidLap"])
    count = int(mask.sum())
    if count:
        df.loc[mask, "ValidLap"] = False
        df.loc[mask, "ExclusionReason"] = REASON_TOO_SHORT
        log.debug(f"  {count} lap(s) too short (< {LAP_TIME_MIN}s) flagged.")
    return df, count


def flag_long_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Flags laps longer than LAP_TIME_MAX seconds (red flag, VSC...)."""
    mask = (df["LapTime_Seconds"].notna()
            & (df["LapTime_Seconds"] > LAP_TIME_MAX)
            & df["ValidLap"])
    count = int(mask.sum())
    if count:
        df.loc[mask, "ValidLap"] = False
        df.loc[mask, "ExclusionReason"] = REASON_TOO_LONG
        log.debug(f"  {count} lap(s) too long (> {LAP_TIME_MAX}s) flagged.")
    return df, count



# PIPELINE


def clean_dataframe(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Applies all cleaning rules to df and returns
    the cleaned DataFrame + the statistics report.
    """
    report = CleaningReport(filename=filename, rows_read=len(df))
    df = df.copy()

    log.info(f"[{filename}] start of cleaning — {report.rows_read} lines")

    # step 1 : Add flag columns
    df = add_flag_columns(df)

    # step 2 : Standardise types
    df = standardise_types(df)

    # step 3 : Remove exact duplicates
    df, report.duplicates = remove_duplicates(df)

    # step 4 : Flag missing lap times
    df, report.missing_laptime = flag_missing_lap_times(df)

    # step 5 : Flag short laps
    df, report.short_laps = flag_short_laps(df)

    # step 6 : Flag long laps
    df, report.long_laps = flag_long_laps(df)

    log.info(f"[{filename}] Completed — {report.rows_kept} valid, {report.rows_removed} removed flagged")
    return df, report



# MAIN — loop over all CSV files in the raw directory


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
) -> list[CleaningReport]:
    """Run the pipeline on all CSV files in raw_dir."""

    
    log.info("  RSDSS — Cleaning pipeline F1")
    log.info(f"  Source   : {raw_dir}")
    log.info(f"  Output   : {out_dir}")
    log.info(f"  Reports  : {report_dir}")
    

    #
    try:
        csv_files = discover_files(raw_dir)
    except FileNotFoundError as e:
        log.error(e)
        return []

    if not csv_files:
        log.warning("Aucun fichier CSV trouvé. Pipeline arrêté.")
        return []

    all_reports = []

    for csv_path in csv_files:
       
        try:
            raw_df = read_csv(csv_path)
        except (ValueError, Exception) as e:
            log.error(f"Fichier ignoré '{csv_path.name}' — {e}")
            continue


        cleaned_df, report = clean_dataframe(raw_df, csv_path.name)

       
        write_csv(cleaned_df, csv_path, out_dir)

        report.print_summary()
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{csv_path.stem}_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        log.debug(f"  Rapport JSON → {report_path.name}")

        all_reports.append(report)

    
    if all_reports:
        log.info("=" * 50)
        log.info(f"  Global Report — {len(all_reports)} file(s) processed")
        log.info(f"  Total lines read    : {sum(r.rows_read    for r in all_reports)}")
        log.info(f"  Total valid rows kept : {sum(r.rows_kept    for r in all_reports)}")
        log.info(f"  Total rows removed      : {sum(r.rows_removed for r in all_reports)}")
        log.info("=" * 50)

    return all_reports





if __name__ == "__main__":
    reports = run_pipeline()
    sys.exit(0 if reports else 1)
