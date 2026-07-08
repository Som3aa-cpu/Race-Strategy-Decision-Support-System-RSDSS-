import os
import pandas as pd
from pathlib import Path

MASTER_FILE = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\Master\featured_dataset.csv")
REPORT_TXT = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\validation_report.txt")
REPORT_CSV = Path(r"D:\Automotive\Race Strategy Decision Support System (RSDSS)\Race-Strategy-Decision-Support-System-RSDSS-\Data\Processed\validation_issues.csv")

EXPECTED_COLUMNS = [
    'RaceID', 'Year', 'RaceName', 'Circuit', 'Driver', 'Team', 'LapNumber',
    'Position', 'LapTime_Seconds', 'Compound', 'TireAge', 'Stint',
    'PitOutTime', 'PitInTime', 'AirTemp', 'TrackTemp', 'Rainfall',
    'Length_km','Corners','DRS_Zones','Direction','HighSpeedCircuit',
    'SurfaceAbrasion','TotalRaceLaps','RaceProgress','RemainingLaps',
    'IsFinal10Laps','RaceHalf','PrevLapTime','PrevLapTime2','RollingMean3',
    'RollingMean5','RollingStd5','LapDelta','PaceTrend','AvgCompoundLife',
    'TireLifePercentage','IsFreshTire','IsOldTire','IsLongStint','DegradationRate',
    'LapLengthFactor','CornerDensity','DRS_per_km','TempDifference','IsHotTrack','IsHotAir','TempCategory'
]

VALID_COMPOUNDS = {
    'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET',
    'UNKNOWN', 'TEST_UNKNOWN'
}

# Realistic boundaries
MIN_LAP_TIME = 45       
MAX_LAP_TIME = 250      
MIN_AIR_TEMP, MAX_AIR_TEMP = -10, 55      # °C
MIN_TRACK_TEMP, MAX_TRACK_TEMP = -10, 75  # °C
MIN_POSITION, MAX_POSITION = 1, 24
MIN_TIRE_AGE, MAX_TIRE_AGE = 0, 80        


def add_issue(issues: list, race_id: str, check: str, detail: str, severity: str = 'ERROR'):
    issues.append({
        'RaceID': race_id,
        'Check': check,
        'Severity': severity,
        'Detail': detail
    })


def validate_file(path: Path, issues: list):
    race_id = path.stem  # filename without extension

    # 6. The race has data.
    try:
        df = pd.read_csv(path)
    except Exception as e:
        add_issue(issues, race_id, 'file_readable', f"can't read file: {e}")
        return

    if df.empty:
        add_issue(issues, race_id, 'has_data', "Empty file (0 rows)")
        return

    # --- 1. Expected columns -------------------------------------------
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        add_issue(issues, race_id, 'columns_exist', f"Missing columns: {missing_cols}")
        # Continue with checks on available columns

    # --- 2. Duplicate rows --------------------------------------------
    full_dupes = df.duplicated().sum()
    if full_dupes > 0:
        add_issue(issues, race_id, 'duplicate_rows', f"{full_dupes} row(s) fully duplicated")

    if {"RaceID", "Driver", "LapNumber"}.issubset(df.columns):
        key_dupes = df.duplicated(subset=["RaceID", "Driver", "LapNumber"]).sum()
        if key_dupes > 0:
            add_issue(issues, race_id, 'duplicate_driver_lap',
                      f"{key_dupes} duplicate(s) Driver+LapNumber (a driver cannot have "
                      f"two laps with the same number)")

# --- 3. Valid lap times -----------------------------------------
    if 'LapTime_Seconds' in df.columns:
        lt = df['LapTime_Seconds']

    n_missing = (lt.isna() & df["ValidLap"]).sum()
    if n_missing > 0:
        add_issue(
            issues,
            race_id,
            'laptime_missing',
            f"{n_missing} lap time(s) missing",
            severity='WARNING'
        )

    invalid_lt = df[
        lt.notna()
        & df["ValidLap"]
        & (
            (lt <= 0)
            | (lt < MIN_LAP_TIME)
            | (lt > MAX_LAP_TIME)
        )
    ]

    if not invalid_lt.empty:
        add_issue(
            issues,
            race_id,
            'laptime_range',
            f"{len(invalid_lt)} lap time(s) outside realistic range "
            f"[{MIN_LAP_TIME}-{MAX_LAP_TIME}s], e.g.: "
            f"{invalid_lt['LapTime_Seconds'].head(3).tolist()}"
        )

    # --- 4. Valid tire compounds --------------------------------------
    if 'Compound' in df.columns:
        compounds_present = set(df['Compound'].dropna().unique())
        invalid_compounds = compounds_present - VALID_COMPOUNDS
        if invalid_compounds:
            add_issue(issues, race_id, 'compound_valid', f"Unknown compounds: {invalid_compounds}")

    # --- 5. TireAge never negative ------------------------------------------
    if 'TireAge' in df.columns:
        negative_age = df[df['TireAge'] < MIN_TIRE_AGE]
        if not negative_age.empty:
            add_issue(issues, race_id, 'tireage_negative',
                      f"{len(negative_age)} negative TireAge value(s)")
        too_old = df[df['TireAge'] > MAX_TIRE_AGE]
        if not too_old.empty:
            add_issue(issues, race_id, 'tireage_implausible',
                      f"{len(too_old)} TireAge value(s) outside realistic range "
                      f"[{MIN_TIRE_AGE}-{MAX_TIRE_AGE}], e.g.: {too_old['TireAge'].head(3).tolist()}")

    # --- 7. Consecutive lap numbers per driver --------------------------
    if {"RaceID", "Driver", "LapNumber"}.issubset(df.columns):
        for driver, group in df.groupby('Driver'):
            laps = sorted(group['LapNumber'].dropna().unique())
            if not laps:
                continue
            expected_range = set(range(int(laps[0]), int(laps[-1]) + 1))
            missing_laps = sorted(expected_range - set(int(l) for l in laps))
            if missing_laps:
                add_issue(issues, race_id, 'consecutive_laps',
                          f"{driver}: lap(s) missing in the sequence: {missing_laps}",
                          severity='WARNING')

    # --- 8. Impossible values ---------------------------------------------
    if 'AirTemp' in df.columns:
        bad_air = df[(df['AirTemp'] < MIN_AIR_TEMP) | (df['AirTemp'] > MAX_AIR_TEMP)]
        if not bad_air.empty:
            add_issue(issues, race_id, 'airtemp_range',
                      f"{len(bad_air)} AirTemp value(s) outside range [{MIN_AIR_TEMP}, {MAX_AIR_TEMP}]°C")

    if 'TrackTemp' in df.columns:
        bad_track = df[(df['TrackTemp'] < MIN_TRACK_TEMP) | (df['TrackTemp'] > MAX_TRACK_TEMP)]
        if not bad_track.empty:
            add_issue(issues, race_id, 'tracktemp_range',
                      f"{len(bad_track)} TrackTemp value(s) outside range [{MIN_TRACK_TEMP}, {MAX_TRACK_TEMP}]°C")

    if 'Rainfall' in df.columns:
        bad_rain = df[~df['Rainfall'].isin([True, False, 0, 1]) & df['Rainfall'].notna()]
        if not bad_rain.empty:
            add_issue(issues, race_id, 'rainfall_invalid',
                      f"{len(bad_rain)} Rainfall value(s) not boolean nor 0/1")

    if 'Position' in df.columns:
        bad_pos = df[(df['Position'] < MIN_POSITION) | (df['Position'] > MAX_POSITION)]
        if not bad_pos.empty:
            add_issue(issues, race_id, 'position_range',
                      f"{len(bad_pos)} Position value(s) outside range [{MIN_POSITION}, {MAX_POSITION}]")

    if 'LapNumber' in df.columns:
        bad_lapnum = df[df['LapNumber'] <= 0]
        if not bad_lapnum.empty:
            add_issue(issues, race_id, 'lapnumber_invalid',
                      f"{len(bad_lapnum)} LapNumber <= 0")

    if 'Stint' in df.columns:
        bad_stint = df[df['Stint'] <= 0]
        if not bad_stint.empty:
            add_issue(issues, race_id, 'stint_invalid', f"{len(bad_stint)} Stint <= 0")


def run_validation(csv_path: Path = MASTER_FILE):
    issues = []
    
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return
    
    validate_file(csv_path, issues)
    
    issues_df = pd.DataFrame(issues)
    has_issues = not issues_df.empty
    
    n_errors = (issues_df['Severity'] == 'ERROR').sum() if has_issues else 0
    n_warnings = (issues_df['Severity'] == 'WARNING').sum() if has_issues else 0

    # --- Console / text report ------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("Report of validation - F1 DATA PIPELINE")
    lines.append("=" * 70)
    lines.append(f"File analyzed    : {csv_path.name}")
    lines.append(f"File path        : {csv_path}")
    lines.append(f"Status           : {'VALID' if not has_issues else 'ISSUES FOUND'}")
    lines.append(f"Errors (ERROR)   : {n_errors}")
    lines.append(f"Warnings (WARNING): {n_warnings}")
    lines.append("")

    if not has_issues:
        lines.append("No anomalies detected. The file is valid.")
    else:
        for race_id, group in issues_df.groupby('RaceID'):
            lines.append(f"--- {race_id} ---")
            for _, row in group.iterrows():
                icon = "error" if row['Severity'] == 'ERROR' else "issue"
                lines.append(f"  {icon} [{row['Check']}] {row['Detail']}")
            lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    os.makedirs(REPORT_TXT.parent, exist_ok=True)
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_text)

    if has_issues:
        os.makedirs(REPORT_CSV.parent, exist_ok=True)
        issues_df.to_csv(REPORT_CSV, index=False)
        print(f"\nDetail of anomalies: {REPORT_CSV}")
    print(f"Complete report: {REPORT_TXT}")


if __name__ == '__main__':
    run_validation()