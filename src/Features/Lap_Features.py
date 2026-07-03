import logging

import pandas as pd


# Logger inherited from the pipeline; falls back to basicConfig in isolation.
log = logging.getLogger("rsdss.features.lap")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

# Grouping keys — every feature is computed within this scope
_GROUP_KEYS = ["RaceID", "Driver"]



# Feature 1 — PrevLapTime


def add_prev_lap_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PrevLapTime — the driver's lap time on the immediately preceding lap.
    Computed as a shift(1) of LapTime_Seconds within each [RaceID, Driver]
    group. The first lap of each stint will be NaN (no prior lap exists).
    This is the most direct signal of recent pace and serves as the base
    for LapDelta and PaceTrend.
    """
    df["PrevLapTime"] = (
        df.groupby(_GROUP_KEYS)["LapTime_Seconds"]
        .shift(1)
    )
    log.info("Added : PrevLapTime")
    return df



# Feature 2 — PrevLapTime2


def add_prev_lap_time_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PrevLapTime2 — the driver's lap time two laps ago.
    Computed as a shift(2) of LapTime_Seconds within each [RaceID, Driver]
    group. The first two laps of each race will be NaN.
    Together with PrevLapTime this gives the model a short-term pace
    trajectory without requiring the full rolling window.
    """
    df["PrevLapTime2"] = (
        df.groupby(_GROUP_KEYS)["LapTime_Seconds"]
        .shift(2)
    )
    log.info("Added : PrevLapTime2")
    return df



# Feature 3 — RollingMean3


def add_rolling_mean_3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RollingMean3 — rolling mean of the previous 3 lap times.
    Uses a window of 3 with min_periods=1, shifted by 1 so the current
    lap is never included (strict look-back only).
    A 3-lap window captures short-term pace changes — useful for detecting
    undercuts and immediate tyre degradation effects.
    """
    df["RollingMean3"] = (
        df.groupby(_GROUP_KEYS)["LapTime_Seconds"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    log.info("Added : RollingMean3  (window=3, look-back only)")
    return df



# Feature 4 — RollingMean5 


def add_rolling_mean_5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RollingMean5 — rolling mean of the previous 5 lap times.
    Uses a window of 5 with min_periods=1, shifted by 1 to exclude
    the current lap.
    The 5-lap window is the primary pace baseline used in strategy models.
    It smooths out single-lap anomalies (traffic, minor lock-ups) while
    still reflecting genuine medium-term pace changes.
    """
    df["RollingMean5"] = (
        df.groupby(_GROUP_KEYS)["LapTime_Seconds"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )
    log.info("Added : RollingMean5  (window=5, look-back only)")
    return df



# FEATURE 5 — RollingStd5


def add_rolling_std_5(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RollingStd5 — rolling standard deviation of the previous 5 lap times.
    Uses a window of 5 with min_periods=2 (std requires at least 2 values),
    shifted by 1 to exclude the current lap.
    Measures pace consistency: a high RollingStd5 indicates erratic lap
    times which may signal tyre degradation, traffic, or instability.
    A low value means the driver is in a clean, consistent rhythm.
    """
    df["RollingStd5"] = (
        df.groupby(_GROUP_KEYS)["LapTime_Seconds"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=2).std())
    )
    log.info("Added : RollingStd5  (window=5, consistency metric)")
    return df



# Feature 6 — LapDelta


def add_lap_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LapDelta — difference between the current and previous lap time.
    Formula : LapDelta = LapTime_Seconds - PrevLapTime
    Interpretation:
        LapDelta > 0 → driver is slower than the previous lap
        LapDelta < 0 → driver is faster than the previous lap
        LapDelta ≈ 0 → stable pace
    This is the single most important per-lap signal for detecting
    pace changes, degradation onset, and tyre cliff moments.
    """
    if "PrevLapTime" not in df.columns:
        log.warning("PrevLapTime not found — computing automatically.")
        df = add_prev_lap_time(df)

    df["LapDelta"] = df["LapTime_Seconds"] - df["PrevLapTime"]
    log.info("Added : LapDelta  (current - previous lap time)")
    return df



# Feature 7 — PaceTrend


def add_pace_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add PaceTrend — difference between the 3-lap and 5-lap rolling means.
    Formula : PaceTrend = RollingMean3 - RollingMean5
    Interpretation:
        PaceTrend > 0 → recent 3-lap pace is slower than 5-lap baseline
                         → pace is getting worse (tyre degradation likely)
        PaceTrend < 0 → recent 3-lap pace is faster than 5-lap baseline
                         → pace is improving (fresh tyres, track evolution)
        PaceTrend ≈ 0 → stable pace, no meaningful trend
    This feature gives the model a directional signal about whether the
    driver's pace is improving or deteriorating at any given point.
    """
    if "RollingMean3" not in df.columns:
        log.warning("RollingMean3 not found — computing automatically.")
        df = add_rolling_mean_3(df)
    if "RollingMean5" not in df.columns:
        log.warning("RollingMean5 not found — computing automatically.")
        df = add_rolling_mean_5(df)

    df["PaceTrend"] = df["RollingMean3"] - df["RollingMean5"]
    log.info("Added : PaceTrend  (RollingMean3 - RollingMean5)")
    return df



# the function that will be called by the feature engineering pipeline


def add_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all lap-pace features in dependency order.
    Sorts the DataFrame by [RaceID, Driver, LapNumber] before computing
    any feature to guarantee correct shift and rolling behaviour.
    The original index is reset after sorting.
    Execution order:
        sort → PrevLapTime → PrevLapTime2 → RollingMean3 → RollingMean5
             → RollingStd5 → LapDelta → PaceTrend
    """
    log.info("Running lap_features module...")

    # Sort is mandatory — shifts and rolling windows depend on row order
    df = df.sort_values(["RaceID", "Driver", "LapNumber"]).reset_index(drop=True)

    df = add_prev_lap_time(df)
    df = add_prev_lap_time_2(df)
    df = add_rolling_mean_3(df)
    df = add_rolling_mean_5(df)
    df = add_rolling_std_5(df)
    df = add_lap_delta(df)
    df = add_pace_trend(df)

    log.info("lap_features done — 7 columns added.")
    return df