import os
import glob
import pandas as pd
from pathlib import Path

INDIVIDUAL_DIR = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Raw\Data\Processed\by_race")
MASTER_FILE = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Raw\Data\Processed\all_races_2022_2025.csv")
REPORT_TXT = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Raw\Data\Processed\validation_report.txt")
REPORT_CSV = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Raw\Data\Processed\validation_issues.csv")

EXPECTED_COLUMNS = [
    'RaceID', 'Year', 'RaceName', 'Circuit', 'Driver', 'Team', 'LapNumber',
    'Position', 'LapTime_Seconds', 'Compound', 'TireAge', 'Stint',
    'PitOutTime', 'PitInTime', 'AirTemp', 'TrackTemp', 'Rainfall'
]

VALID_COMPOUNDS = {
    'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET',
    'UNKNOWN', 'TEST_UNKNOWN'
}

# Plages réalistes (larges volontairement pour couvrir tous les circuits/saisons)
MIN_LAP_TIME = 45       # secondes - aucun circuit F1 n'a un tour aussi court
MAX_LAP_TIME = 250      # secondes - couvre les tours sous safety car / pluie
MIN_AIR_TEMP, MAX_AIR_TEMP = -10, 55      # °C
MIN_TRACK_TEMP, MAX_TRACK_TEMP = -10, 75  # °C
MIN_POSITION, MAX_POSITION = 1, 24
MIN_TIRE_AGE, MAX_TIRE_AGE = 0, 80        # tours - large marge de sécurité


def add_issue(issues: list, race_id: str, check: str, detail: str, severity: str = 'ERROR'):
    issues.append({
        'RaceID': race_id,
        'Check': check,
        'Severity': severity,
        'Detail': detail
    })


def validate_file(path: str, issues: list):
    race_id = os.path.splitext(os.path.basename(path))[0]

    # --- 6. La course a des données -------------------------------------
    try:
        df = pd.read_csv(path)
    except Exception as e:
        add_issue(issues, race_id, 'file_readable', f"Impossible de lire le fichier: {e}")
        return

    if df.empty:
        add_issue(issues, race_id, 'has_data', "Fichier vide (0 ligne)")
        return  # inutile de continuer les autres checks sur un fichier vide

    # --- 1. Colonnes attendues -------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        add_issue(issues, race_id, 'columns_exist', f"Colonnes manquantes: {missing_cols}")
        # On continue quand même les checks possibles sur les colonnes présentes

    # --- 2. Lignes dupliquées --------------------------------------------
    full_dupes = df.duplicated().sum()
    if full_dupes > 0:
        add_issue(issues, race_id, 'duplicate_rows', f"{full_dupes} ligne(s) entièrement dupliquée(s)")

    if {'Driver', 'LapNumber'}.issubset(df.columns):
        key_dupes = df.duplicated(subset=['Driver', 'LapNumber']).sum()
        if key_dupes > 0:
            add_issue(issues, race_id, 'duplicate_driver_lap',
                      f"{key_dupes} doublon(s) Driver+LapNumber (un pilote ne peut avoir "
                      f"deux fois le même numéro de tour)")

    # --- 3. Temps au tour valides -----------------------------------------
    if 'LapTime_Seconds' in df.columns:
        lt = df['LapTime_Seconds']
        n_missing = lt.isna().sum()
        if n_missing > 0:
            add_issue(issues, race_id, 'laptime_missing', f"{n_missing} temps au tour manquant(s)",
                      severity='WARNING')

        invalid_lt = df[lt.notna() & ((lt <= 0) | (lt < MIN_LAP_TIME) | (lt > MAX_LAP_TIME))]
        if not invalid_lt.empty:
            add_issue(issues, race_id, 'laptime_range',
                      f"{len(invalid_lt)} temps au tour hors plage réaliste "
                      f"[{MIN_LAP_TIME}-{MAX_LAP_TIME}s], ex: {invalid_lt['LapTime_Seconds'].head(3).tolist()}")

    # --- 4. Composés de pneus valides --------------------------------------
    if 'Compound' in df.columns:
        compounds_present = set(df['Compound'].dropna().unique())
        invalid_compounds = compounds_present - VALID_COMPOUNDS
        if invalid_compounds:
            add_issue(issues, race_id, 'compound_valid', f"Composés inconnus: {invalid_compounds}")

    # --- 5. TireAge jamais négatif ------------------------------------------
    if 'TireAge' in df.columns:
        negative_age = df[df['TireAge'] < MIN_TIRE_AGE]
        if not negative_age.empty:
            add_issue(issues, race_id, 'tireage_negative',
                      f"{len(negative_age)} valeur(s) de TireAge négative(s)")
        too_old = df[df['TireAge'] > MAX_TIRE_AGE]
        if not too_old.empty:
            add_issue(issues, race_id, 'tireage_implausible',
                      f"{len(too_old)} valeur(s) de TireAge > {MAX_TIRE_AGE} tours (suspect)",
                      severity='WARNING')

    # --- 7. Numéros de tour consécutifs par pilote --------------------------
    if {'Driver', 'LapNumber'}.issubset(df.columns):
        for driver, group in df.groupby('Driver'):
            laps = sorted(group['LapNumber'].dropna().unique())
            if not laps:
                continue
            expected_range = set(range(int(laps[0]), int(laps[-1]) + 1))
            missing_laps = sorted(expected_range - set(int(l) for l in laps))
            if missing_laps:
                add_issue(issues, race_id, 'consecutive_laps',
                          f"{driver}: tour(s) manquant(s) dans la séquence: {missing_laps}",
                          severity='WARNING')

    # --- 8. Valeurs impossibles ---------------------------------------------
    if 'AirTemp' in df.columns:
        bad_air = df[(df['AirTemp'] < MIN_AIR_TEMP) | (df['AirTemp'] > MAX_AIR_TEMP)]
        if not bad_air.empty:
            add_issue(issues, race_id, 'airtemp_range',
                      f"{len(bad_air)} valeur(s) AirTemp hors plage [{MIN_AIR_TEMP}, {MAX_AIR_TEMP}]°C")

    if 'TrackTemp' in df.columns:
        bad_track = df[(df['TrackTemp'] < MIN_TRACK_TEMP) | (df['TrackTemp'] > MAX_TRACK_TEMP)]
        if not bad_track.empty:
            add_issue(issues, race_id, 'tracktemp_range',
                      f"{len(bad_track)} valeur(s) TrackTemp hors plage [{MIN_TRACK_TEMP}, {MAX_TRACK_TEMP}]°C")

    if 'Rainfall' in df.columns:
        bad_rain = df[~df['Rainfall'].isin([True, False, 0, 1]) & df['Rainfall'].notna()]
        if not bad_rain.empty:
            add_issue(issues, race_id, 'rainfall_invalid',
                      f"{len(bad_rain)} valeur(s) Rainfall ni booléenne ni 0/1")

    if 'Position' in df.columns:
        bad_pos = df[(df['Position'] < MIN_POSITION) | (df['Position'] > MAX_POSITION)]
        if not bad_pos.empty:
            add_issue(issues, race_id, 'position_range',
                      f"{len(bad_pos)} valeur(s) Position hors plage [{MIN_POSITION}, {MAX_POSITION}]")

    if 'LapNumber' in df.columns:
        bad_lapnum = df[df['LapNumber'] <= 0]
        if not bad_lapnum.empty:
            add_issue(issues, race_id, 'lapnumber_invalid',
                      f"{len(bad_lapnum)} LapNumber <= 0")

    if 'Stint' in df.columns:
        bad_stint = df[df['Stint'] <= 0]
        if not bad_stint.empty:
            add_issue(issues, race_id, 'stint_invalid', f"{len(bad_stint)} Stint <= 0")


def run_validation():
    files = sorted(glob.glob(os.path.join(INDIVIDUAL_DIR,'*.csv')))
    if not files:
        print(f" Aucun fichier trouvé dans {INDIVIDUAL_DIR}")
        return

    issues = []
    for path in files:
        validate_file(path, issues)

    issues_df = pd.DataFrame(issues)
    n_files = len(files)
    n_clean = n_files - issues_df['RaceID'].nunique() if not issues_df.empty else n_files
    n_errors = (issues_df['Severity'] == 'ERROR').sum() if not issues_df.empty else 0
    n_warnings = (issues_df['Severity'] == 'WARNING').sum() if not issues_df.empty else 0

    # --- Rapport console / texte ------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT DE VALIDATION - F1 DATA PIPELINE")
    lines.append("=" * 70)
    lines.append(f"Fichiers analysés   : {n_files}")
    lines.append(f"Fichiers sans anomalie : {n_clean}")
    lines.append(f"Erreurs (ERROR)     : {n_errors}")
    lines.append(f"Avertissements (WARNING) : {n_warnings}")
    lines.append("")

    if issues_df.empty:
        lines.append("Aucune anomalie détectée. Toutes les courses sont valides.")
    else:
        for race_id, group in issues_df.groupby('RaceID'):
            lines.append(f"--- {race_id} ---")
            for _, row in group.iterrows():
                icon = "error" if row['Severity'] == 'ERROR' else "issue"
                lines.append(f"  {icon} [{row['Check']}] {row['Detail']}")
            lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs(os.path.dirname(REPORT_TXT), exist_ok=True)
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_text)

    if not issues_df.empty:
        issues_df.to_csv(REPORT_CSV, index=False)
        print(f"\nDétail des anomalies: {REPORT_CSV}")
    print(f"Rapport complet: {REPORT_TXT}")


if __name__ == '__main__':
    run_validation()