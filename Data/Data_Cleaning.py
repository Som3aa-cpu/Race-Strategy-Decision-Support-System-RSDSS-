import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


# CONFIGURATION — tous les constantes en un seul endroit


RAW_DATA_DIR       = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Raw\Data\Processed\by_race")
PROCESSED_DATA_DIR = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed")
REPORTS_DIR        = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\reports")

# Seuils pour les temps au tour (en secondes)
LAP_TIME_MIN = 45.0
LAP_TIME_MAX = 250.0

# Colonnes obligatoires dans chaque CSV
REQUIRED_COLUMNS = [
    "RaceID", "RaceName", "Circuit", "Driver", "Team",
    "LapNumber", "Position", "LapTime_Seconds", "Compound", "TireAge",
    "Stint", "PitOutTime", "PitInTime", "AirTemp", "TrackTemp", "Rainfall",
]

# Raisons d'exclusion (labels fixes pour éviter les fautes de frappe)
REASON_MISSING   = "Missing Lap Time"
REASON_TOO_SHORT = "Unrealistically Short Lap"
REASON_TOO_LONG  = "Red Flag / Extremely Long Lap"
REASON_DUPLICATE = "Duplicate Row"



# LOGGING — console + fichier rotatif


def setup_logger(name: str = "rsdss") -> logging.Logger:
    """Configure et retourne un logger avec handlers console et fichier."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # déjà configuré, on évite les doublons

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%H:%M:%S")

    # Handler console — INFO et plus
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    # Handler fichier — DEBUG et plus
    Path("logs").mkdir(exist_ok=True)
    file_h = logging.FileHandler("logs/rsdss_cleaning.log", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_h)
    logger.propagate = False
    return logger


log = setup_logger()



# RAPPORT DE NETTOYAGE — dataclass simple


@dataclass
class CleaningReport:
    """Statistiques produites pour un seul fichier CSV."""

    filename:         str
    rows_read:        int = 0
    duplicates:       int = 0
    missing_laptime:  int = 0
    short_laps:       int = 0
    long_laps:        int = 0

    # Propriétés calculées — pas besoin de les stocker
    @property
    def rows_flagged(self) -> int:
        return self.missing_laptime + self.short_laps + self.long_laps

    @property
    def rows_removed(self) -> int:
        return self.duplicates + self.rows_flagged

    @property
    def rows_kept(self) -> int:
        # Gardé = total après déduplication - rangées signalées
        return (self.rows_read - self.duplicates) - self.rows_flagged

    def print_summary(self) -> None:
        """Affiche un résumé lisible dans les logs."""
        sep = "-" * 50
        log.info(sep)
        log.info(f" RAPPORT — {self.filename}")
        log.info(sep)
        log.info(f"  Lignes lues             : {self.rows_read}")
        log.info(f"  Doublons supprimés      : {self.duplicates}")
        log.info(f"  Temps manquants         : {self.missing_laptime}")
        log.info(f"  Tours trop courts       : {self.short_laps}")
        log.info(f"  Tours trop longs        : {self.long_laps}")
        log.info(sep)
        log.info(f"  Total supprimé/signalé  : {self.rows_removed}")
        log.info(f"  Lignes valides gardées  : {self.rows_kept}")
        log.info(sep)

    def to_dict(self) -> dict:
        """Sérialise le rapport en dictionnaire JSON-compatible."""
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



# I/O — lecture et écriture des fichiers CSV


def discover_files(raw_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Retourne tous les fichiers CSV trouvés dans raw_dir."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {raw_dir}")

    files = sorted(raw_dir.glob("*.csv"))
    log.info(f"{len(files)} fichier(s) CSV trouvé(s) dans '{raw_dir}'")
    return files


def read_csv(file_path: Path) -> pd.DataFrame:
    """Lit un CSV et vérifie que toutes les colonnes requises sont présentes."""
    log.info(f"Lecture : {file_path.name}")
    df = pd.read_csv(file_path, low_memory=False)

    if df.empty:
        raise ValueError(f"Fichier vide : {file_path.name}")

    # Vérification des colonnes manquantes
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans {file_path.name} : {missing_cols}")

    log.debug(f"  → {len(df)} lignes × {len(df.columns)} colonnes chargées")
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
        log.warning(f"  Impossible de dériver la saison depuis '{filename}'")
 
    # Insère Season juste après RaceID pour un ordre logique
    insert_at = df.columns.get_loc("RaceID") + 1 if "RaceID" in df.columns else 0
    df.insert(insert_at, "Season", season)
    log.debug(f"  Colonne Season = {season} injectée depuis le nom de fichier.")
    return df

def write_csv(df: pd.DataFrame, source_file: Path,
              out_dir: Path = PROCESSED_DATA_DIR) -> Path:
    """Sauvegarde le DataFrame nettoyé dans out_dir avec le même nom de fichier."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / source_file.name
    df.to_csv(out_path, index=False)
    log.info(f"  → Sauvegardé : {out_path}  ({len(df)} lignes)")
    return out_path



# RÈGLES DE NETTOYAGE — une fonction par règle


def add_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes ValidLap et ExclusionReason si elles n'existent pas."""
    if "ValidLap" not in df.columns:
        df["ValidLap"] = True
    if "ExclusionReason" not in df.columns:
        df["ExclusionReason"] = ""
    return df


def standardise_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes vers leurs types corrects (int, float, string)."""
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
    """Supprime physiquement les lignes exactement identiques."""
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    count = before - len(df)
    if count:
        log.debug(f"  {count} doublon(s) supprimé(s).")
    return df, count


def flag_missing_lap_times(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Signale les lignes où LapTime_Seconds est NaN."""
    mask = df["LapTime_Seconds"].isna() & df["ValidLap"]
    count = int(mask.sum())
    if count:
        df.loc[mask, "ValidLap"] = False
        df.loc[mask, "ExclusionReason"] = REASON_MISSING
        log.debug(f"  {count} tour(s) avec temps manquant signalé(s).")
    return df, count


def flag_short_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Signale les tours inférieurs à LAP_TIME_MIN secondes."""
    mask = (df["LapTime_Seconds"].notna()
            & (df["LapTime_Seconds"] < LAP_TIME_MIN)
            & df["ValidLap"])
    count = int(mask.sum())
    if count:
        df.loc[mask, "ValidLap"] = False
        df.loc[mask, "ExclusionReason"] = REASON_TOO_SHORT
        log.debug(f"  {count} tour(s) trop court(s) (< {LAP_TIME_MIN}s) signalé(s).")
    return df, count


def flag_long_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Signale les tours supérieurs à LAP_TIME_MAX secondes (drapeau rouge, VSC...)."""
    mask = (df["LapTime_Seconds"].notna()
            & (df["LapTime_Seconds"] > LAP_TIME_MAX)
            & df["ValidLap"])
    count = int(mask.sum())
    if count:
        df.loc[mask, "ValidLap"] = False
        df.loc[mask, "ExclusionReason"] = REASON_TOO_LONG
        log.debug(f"  {count} tour(s) trop long(s) (> {LAP_TIME_MAX}s) signalé(s).")
    return df, count



# PIPELINE — applique toutes les règles sur un seul DataFrame


def clean_dataframe(df: pd.DataFrame, filename: str) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Applique toutes les règles de nettoyage sur df et retourne
    le DataFrame nettoyé + le rapport de statistiques.
    """
    report = CleaningReport(filename=filename, rows_read=len(df))
    df = df.copy()

    log.info(f"[{filename}] Début du nettoyage — {report.rows_read} lignes")

    # Étape 1 : Ajouter les colonnes de flag
    df = add_flag_columns(df)

    # Étape 2 : Standardiser les types
    df = standardise_types(df)

    # Étape 3 : Supprimer les doublons exacts
    df, report.duplicates = remove_duplicates(df)

    # Étape 4 : Signaler les temps manquants
    df, report.missing_laptime = flag_missing_lap_times(df)

    # Étape 5 : Signaler les tours trop courts
    df, report.short_laps = flag_short_laps(df)

    # Étape 6 : Signaler les tours trop longs
    df, report.long_laps = flag_long_laps(df)

    log.info(f"[{filename}] Terminé — {report.rows_kept} valides, {report.rows_removed} supprimés/signalés")
    return df, report



# MAIN — boucle sur tous les fichiers CSV du dossier raw


def run_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
    report_dir: Path = REPORTS_DIR,
) -> list[CleaningReport]:
    """Lance le pipeline sur tous les fichiers CSV de raw_dir."""

    
    log.info("  RSDSS — Pipeline de nettoyage F1")
    log.info(f"  Source   : {raw_dir}")
    log.info(f"  Sortie   : {out_dir}")
    log.info(f"  Rapports : {report_dir}")
    

    # Découverte des fichiers
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
        # Lecture — on saute les fichiers corrompus sans arrêter le pipeline
        try:
            raw_df = read_csv(csv_path)
        except (ValueError, Exception) as e:
            log.error(f"Fichier ignoré '{csv_path.name}' — {e}")
            continue

        # Nettoyage
        cleaned_df, report = clean_dataframe(raw_df, csv_path.name)

        # Sauvegarde CSV nettoyé
        write_csv(cleaned_df, csv_path, out_dir)

        # Affichage + sauvegarde du rapport JSON
        report.print_summary()
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{csv_path.stem}_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        log.debug(f"  Rapport JSON → {report_path.name}")

        all_reports.append(report)

    # Résumé global à la fin
    if all_reports:
        log.info("=" * 50)
        log.info(f"  RÉSUMÉ GLOBAL — {len(all_reports)} fichier(s) traité(s)")
        log.info(f"  Total lignes lues    : {sum(r.rows_read    for r in all_reports)}")
        log.info(f"  Total valides gardés : {sum(r.rows_kept    for r in all_reports)}")
        log.info(f"  Total supprimés      : {sum(r.rows_removed for r in all_reports)}")
        log.info("=" * 50)

    return all_reports



# POINT D'ENTRÉE


if __name__ == "__main__":
    reports = run_pipeline()
    sys.exit(0 if reports else 1)
