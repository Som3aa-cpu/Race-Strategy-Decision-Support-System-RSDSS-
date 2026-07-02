import logging
import sys
from pathlib import Path
import pandas as pd


# CONFIGURATION
PROCESSED_DIR  = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed")
MASTER_OUTPUT  = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\Master\Master_Dataset.csv")


# LOGGING
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rsdss_merge")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                            datefmt="%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)
    logger.propagate = False
    return logger

log = setup_logger()


# MERGE
def merge_processed_datasets(
    processed_dir: Path = PROCESSED_DIR,
    output_path:   Path = MASTER_OUTPUT,
) -> pd.DataFrame:
    """
    Charge tous les CSV nettoyés, les concatène en un seul DataFrame,
    trie par Season → RaceID → Driver → LapNumber, et sauvegarde.

    Returns le DataFrame master.
    """

    # --- Découverte des fichiers ---
    csv_files = sorted(processed_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Aucun CSV trouvé dans : {processed_dir}")

    log.info("  RSDSS — Fusion des datasets nettoyés")
    log.info(f"  Source  : {processed_dir}")
    log.info(f"  Sortie  : {output_path}")
    log.info(f"  {len(csv_files)} fichier(s) trouvé(s)")

    # --- Chargement et concaténation ---
    frames = []
    skipped = 0

    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
            log.debug(f"  ✓ {f.name:<12} — {len(df):>5,} lignes")
        except Exception as e:
            log.warning(f"  ✗ {f.name} ignoré — {e}")
            skipped += 1

    if not frames:
        raise RuntimeError("Aucun fichier n'a pu être chargé.")

    master = pd.concat(frames, ignore_index=True)
    log.info(f"  Concaténation : {len(master):,} lignes × {len(master.columns)} colonnes")

    # --- Tri logique ---
    sort_cols = [c for c in ["Season", "RaceID", "Driver", "LapNumber"]
                 if c in master.columns]
    if sort_cols:
        master = master.sort_values(sort_cols).reset_index(drop=True)
        log.info(f"  Trié par : {sort_cols}")

    # --- Sauvegarde ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output_path, index=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    log.info("=" * 52)
    log.info(f"  Master dataset sauvegardé ✓")
    log.info(f"  Lignes        : {len(master):,}")
    log.info(f"  Colonnes      : {len(master.columns)}")
    log.info(f"  Courses       : {master['RaceID'].nunique() if 'RaceID' in master.columns else '?'}")
    log.info(f"  Saisons       : {sorted(master['Season'].dropna().unique().astype(int).tolist()) if 'Season' in master.columns else '?'}")
    log.info(f"  Taille        : {size_mb:.2f} MB")
    log.info(f"  Fichier       : {output_path}")
    log.info("=" * 52)

    if skipped:
        log.warning(f"  {skipped} fichier(s) ignoré(s) à cause d'erreurs.")

    return master


# POINT D'ENTRÉE

if __name__ == "__main__":
    merge_processed_datasets()