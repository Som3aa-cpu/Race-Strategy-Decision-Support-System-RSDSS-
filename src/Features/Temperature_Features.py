import logging

import numpy as np
import pandas as pd


log = logging.getLogger("rsdss.features.tire")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

# Minimum laps in a stint required to compute a meaningful degradation slope.
# Below this threshold DegradationRate is set to NaN.
MIN_LAPS_FOR_SLOPE = 3

# Thresholds for the boolean flags
FRESH_TIRE_THRESHOLD = 2    # TireAge <= 2  → fresh
OLD_TIRE_THRESHOLD   = 20   # TireAge >= 20 → old



# Intermadiate — AvgCompoundLife


def compute_avg_compound_life(df: pd.DataFrame) -> dict[str, float]:
    """Compute the average number of laps each compound typically lasts.
    Method:
        1. For every [RaceID, Driver, Stint, Compound] group take the
           maximum TireAge — this is how long that particular stint lasted.
        2. Average across all stints of the same compound.
    The result is a dictionary used internally to compute TireLifePercentage
    and IsLongStint.  It is also stored as AvgCompoundLife on the DataFrame
    so downstream models can use the baseline directly.
    """
    stint_lengths = (
        df.groupby(["RaceID", "Driver", "Stint", "Compound"])["TireAge"]
        .max()
        .reset_index(name="StintLength")
    )
    avg_life = (
        stint_lengths.groupby("Compound")["StintLength"]
        .mean()
        .round(2)
        .to_dict()
    )
    log.info("Computed AvgCompoundLife : %s", avg_life)
    return avg_life


def add_avg_compound_life(
    df: pd.DataFrame,
    avg_life: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Add AvgCompoundLife column — the average lifespan of the tyre compound
    currently fitted, derived from historical stints in the dataset.
    This is an intermediate feature: useful on its own as a model input
    and required by TireLifePercentage and IsLongStint.
    """
    if avg_life is None:
        avg_life = compute_avg_compound_life(df)

    df["AvgCompoundLife"] = df["Compound"].map(avg_life)
    log.info("Added : AvgCompoundLife")
    return df, avg_life



# Feature 1 — TireLifePercentage 


def add_tire_life_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TireLifePercentage — how far through its expected life the tyre is.
    Formula : TireLifePercentage = TireAge / AvgCompoundLife
    Examples:
        Hard avg life = 30 laps, current TireAge = 15  →  0.50  (50% used)
        Soft avg life = 18 laps, current TireAge = 18  →  1.00  (fully used)
        Soft avg life = 18 laps, current TireAge = 22  →  1.22  (beyond avg)
    Values above 1.0 are valid and meaningful — they tell the model the
    team is running the tyre beyond its typical lifespan, which often
    precedes a sudden pace cliff.
    """
    if "AvgCompoundLife" not in df.columns:
        log.warning("AvgCompoundLife not found — computing automatically.")
        df, _ = add_avg_compound_life(df)

    df["TireLifePercentage"] = (df["TireAge"] / df["AvgCompoundLife"]).round(4)
    log.info("Added : TireLifePercentage  (TireAge / AvgCompoundLife)")
    return df



# FEATURE 2 — IsFreshTire


def add_is_fresh_tire(df: pd.DataFrame) -> pd.DataFrame:
    """Add IsFreshTire — boolean flag for tyres in their first two laps.

    Formula : IsFreshTire = (TireAge <= 2)

    Fresh tyres behave differently from worn ones: grip levels are still
    building, lap times are often not representative of true pace, and
    drivers adapt their driving style for the first couple of laps.

    Parameters
    ----------
    df : DataFrame with TireAge.

    Returns
    -------
    DataFrame with IsFreshTire column added (dtype: bool).
    """
    df["IsFreshTire"] = df["TireAge"] <= FRESH_TIRE_THRESHOLD
    log.info("Added : IsFreshTire  (TireAge <= %d)", FRESH_TIRE_THRESHOLD)
    return df



# FEATURE 3 — IsOldTire


def add_is_old_tire(df: pd.DataFrame) -> pd.DataFrame:
    """Add IsOldTire — boolean flag for tyres that have done 20 or more laps.

    Formula : IsOldTire = (TireAge >= 20)

    After 20 laps most compounds — particularly Softs and Mediums — are
    showing significant degradation. This flag helps the model learn that
    strategy decisions made on old tyres have a different risk profile.

    Parameters
    ----------
    df : DataFrame with TireAge.

    Returns
    -------
    DataFrame with IsOldTire column added (dtype: bool).
    """
    df["IsOldTire"] = df["TireAge"] >= OLD_TIRE_THRESHOLD
    log.info("Added : IsOldTire  (TireAge >= %d)", OLD_TIRE_THRESHOLD)
    return df



# FEATURE 4 — IsLongStint


def add_is_long_stint(df: pd.DataFrame) -> pd.DataFrame:
    """Add IsLongStint — boolean flag for stints running longer than average.

    Formula : IsLongStint = (TireAge > AvgCompoundLife)

    A long stint signals an aggressive strategy: the team is stretching
    the tyre beyond its typical lifespan, accepting pace loss in exchange
    for track position or to cover a rival's stop.

    Parameters
    ----------
    df : DataFrame with TireAge and AvgCompoundLife.

    Returns
    -------
    DataFrame with IsLongStint column added (dtype: bool).
    """
    if "AvgCompoundLife" not in df.columns:
        log.warning("AvgCompoundLife not found — computing automatically.")
        df, _ = add_avg_compound_life(df)

    df["IsLongStint"] = df["TireAge"] > df["AvgCompoundLife"]
    log.info("Added : IsLongStint  (TireAge > AvgCompoundLife)")
    return df



# FEATURE 5 — DegradationRate  ⭐⭐⭐⭐⭐


def _slope(series_x: pd.Series, series_y: pd.Series) -> float:
    """Fit a linear regression and return the slope (s/lap).

    Returns NaN if fewer than MIN_LAPS_FOR_SLOPE valid data points exist
    or if the fit cannot be computed.
    """
    mask = series_x.notna() & series_y.notna()
    x = series_x[mask].values
    y = series_y[mask].values

    if len(x) < MIN_LAPS_FOR_SLOPE:
        return np.nan

    # np.polyfit degree=1 → [slope, intercept]
    try:
        slope, _ = np.polyfit(x, y, 1)
        return round(float(slope), 6)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add DegradationRate — the rate at which lap time increases per lap
    of tyre age within each individual stint.

    Method:
        For every [RaceID, Driver, Stint] group, fit a linear regression
        of LapTime_Seconds ~ TireAge and store the slope.

        Example: slope = 0.08 means the driver loses 0.08 s per lap
        of tyre age within that stint — i.e. the tyre is degrading at
        80 ms per lap.

    Interpretation:
        DegradationRate > 0  → lap time increasing as tyre ages (normal degradation)
        DegradationRate ≈ 0  → no degradation (hard compound, cool conditions)
        DegradationRate < 0  → lap time improving despite tyre age
                                (track rubbering-in, fuel burn, driver push)

    Requirements:
        - Minimum MIN_LAPS_FOR_SLOPE laps per stint (default 3).
        - Stints below this threshold receive NaN.

    This is expected to be the strongest ML feature in the project because
    it directly quantifies tyre wear — the primary driver of pit stop timing.

    Parameters
    ----------
    df : DataFrame sorted by [RaceID, Driver, LapNumber] with
         LapTime_Seconds, TireAge, and Stint columns.

    Returns
    -------
    DataFrame with DegradationRate column added (units: seconds per lap).
    """
    # Compute slope per stint and broadcast back to every lap in that stint
    degradation = (
        df.groupby(["RaceID", "Driver", "Stint"])
        .apply(
            lambda g: pd.Series(
                _slope(g["TireAge"], g["LapTime_Seconds"]),
                index=g.index,
            )
        )
    )

    # apply with groupby returns a multi-index Series — flatten it
    if isinstance(degradation.index, pd.MultiIndex):
        degradation = degradation.droplevel([0, 1, 2])

    df["DegradationRate"] = degradation
    log.info(
        "Added : DegradationRate  (linear slope LapTime ~ TireAge per stint, "
        "min %d laps required)",
        MIN_LAPS_FOR_SLOPE,
    )
    return df



# PUBLIC ENTRY POINT — called by the feature engineering pipeline


def add_tire_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all tyre features in dependency order.

    AvgCompoundLife is computed once from the full DataFrame and reused
    by both TireLifePercentage and IsLongStint to avoid redundant passes.

    Execution order:
        AvgCompoundLife → TireLifePercentage → IsFreshTire
        → IsOldTire → IsLongStint → DegradationRate

    This is the only function the pipeline needs to call.

    Parameters
    ----------
    df : Cleaned DataFrame (master dataset or any valid subset).
         Must contain: Compound, TireAge, Stint, LapTime_Seconds, RaceID, Driver.

    Returns
    -------
    DataFrame with 6 new columns added.
    """
    log.info("Running tire_features module...")

    # Compute avg compound life once — shared by multiple features
    avg_life = compute_avg_compound_life(df)

    df, _ = add_avg_compound_life(df, avg_life)
    df    = add_tire_life_percentage(df)
    df    = add_is_fresh_tire(df)
    df    = add_is_old_tire(df)
    df    = add_is_long_stint(df)
    df    = add_degradation_rate(df)

    log.info("tire_features done — 6 columns added.")
    return df