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

# Rolling window size for DegradationRate.
DEGRADATION_WINDOW   = 5   # use the last 5 laps (including current)
MIN_LAPS_FOR_SLOPE   = 3   # NaN until at least 3 laps are available

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



# Feature 2 — IsFreshTire


def add_is_fresh_tire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsFreshTire — boolean flag for tyres in their first two laps.
    Formula : IsFreshTire = (TireAge <= 2)
    Fresh tyres behave differently from worn ones: grip levels are still
    building, lap times are often not representative of true pace, and
    drivers adapt their driving style for the first couple of laps.
    """
    df["IsFreshTire"] = df["TireAge"] <= FRESH_TIRE_THRESHOLD
    log.info("Added : IsFreshTire  (TireAge <= %d)", FRESH_TIRE_THRESHOLD)
    return df



# Feature 3 — IsOldTire


def add_is_old_tire(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsOldTire — boolean flag for tyres that have done 20 or more laps.
    Formula : IsOldTire = (TireAge >= 20)
    After 20 laps most compounds — particularly Softs and Mediums — are
    showing significant degradation. This flag helps the model learn that
    strategy decisions made on old tyres have a different risk profile.
    """
    df["IsOldTire"] = df["TireAge"] >= OLD_TIRE_THRESHOLD
    log.info("Added : IsOldTire  (TireAge >= %d)", OLD_TIRE_THRESHOLD)
    return df



# Feature 4 — IsLongStint


def add_is_long_stint(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsLongStint — boolean flag for stints running longer than average.
    Formula : IsLongStint = (TireAge > AvgCompoundLife)
    A long stint signals an aggressive strategy: the team is stretching
    the tyre beyond its typical lifespan, accepting pace loss in exchange
    for track position or to cover a rival's stop.
    """
    if "AvgCompoundLife" not in df.columns:
        log.warning("AvgCompoundLife not found — computing automatically.")
        df, _ = add_avg_compound_life(df)

    df["IsLongStint"] = df["TireAge"] > df["AvgCompoundLife"]
    log.info("Added : IsLongStint  (TireAge > AvgCompoundLife)")
    return df



# Feature 5 — DegradationRate  


def _rolling_slope(group: pd.DataFrame,
                   window: int = DEGRADATION_WINDOW,
                   min_laps: int = MIN_LAPS_FOR_SLOPE) -> pd.Series:
    """
    Compute a rolling linear regression slope of LapTime ~ TireAge.
    For every row i, fits a regression on the window [i-window+1 : i+1].
    The current lap IS included — a race engineer knows the current lap
    time when making decisions.  Returns NaN until min_laps are available.
    """
    laps = group[["TireAge", "LapTime_Seconds"]].to_numpy(dtype=float)
    slopes = []
 
    for i in range(len(laps)):
        start       = max(0, i - window + 1)
        window_data = laps[start : i + 1]
 
        # Remove rows where either value is NaN
        mask = ~(np.isnan(window_data[:, 0]) | np.isnan(window_data[:, 1]))
        x, y = window_data[mask, 0], window_data[mask, 1]
 
        if len(x) < min_laps:
            slopes.append(np.nan)
        else:
            try:
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(round(float(slope)))
            except (np.linalg.LinAlgError, ValueError):
                slopes.append(np.nan)
 
    return pd.Series(slopes, index=group.index)
 
 
def add_degradation_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add DegradationRate — a dynamic, rolling estimate of how quickly
    lap time is increasing with tyre age at each point in the stint.
 
    Method (rolling window):
        For every lap, fit a linear regression of LapTime_Seconds ~ TireAge
        using the current lap and up to the previous 4 laps (window = 5).
        If fewer than 5 laps are available but at least 3 exist, use all
        available laps (expanding window).  Below 3 laps → NaN.
 
    Why rolling and not whole-stint:
        The whole-stint approach uses future laps to compute a slope on
        early laps — that is look-ahead leakage.  The rolling window only
        uses information available at race time, matching how an F1
        strategist actually monitors tyre behaviour lap by lap.
 
    Example output for a degrading Soft tyre:
        TireAge   DegradationRate
            1     NaN              ← not enough data
            2     NaN              ← not enough data
            3     0.06             ← first estimate, 3 laps
            4     0.07
            5     0.09
            6     0.12             ← rate rising: tyre starting to fall off
            7     0.15             ← cliff approaching
 
    Interpretation:
        DegradationRate > 0  → pace getting slower as tyre wears (normal)
        DegradationRate ≈ 0  → stable pace, no meaningful degradation
        DegradationRate < 0  → pace improving  (track evolution, fuel burn)
    """
    # Use a list-based approach to guarantee index alignment across
    # pandas versions — iterate groups, compute slopes, collect results.
    results = pd.Series(index=df.index, dtype=float)
    for _, group in df.groupby(["RaceID", "Driver", "Stint"]):
        group = group.sort_values("LapNumber")
        results.loc[group.index] = _rolling_slope(group).values
 
    df["DegradationRate"] = results
    log.info(
        "Added : DegradationRate  "
        "(rolling window=%d, min_laps=%d, no look-ahead)",
        DEGRADATION_WINDOW, MIN_LAPS_FOR_SLOPE,
    )
    return df



# the function that will be called by the feature engineering pipeline


def add_tire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all tyre features in dependency order.
    AvgCompoundLife is computed once from the full DataFrame and reused
    by both TireLifePercentage and IsLongStint to avoid redundant passes.
    Execution order:
        AvgCompoundLife → TireLifePercentage → IsFreshTire
        → IsOldTire → IsLongStint → DegradationRate
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