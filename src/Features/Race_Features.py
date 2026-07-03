import logging

import pandas as pd


# Logger provided by the pipeline — falls back to a module-level logger
# if this module is imported in isolation during development.
log = logging.getLogger("rsdss.features.race")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )



# Feature 1 — TotalRaceLaps


def add_total_race_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TotalRaceLaps — the actual number of laps completed in each race.
    Uses the maximum LapNumber observed across all drivers per RaceID.
    This approach is robust to shortened races (red flags, rain stoppages)
    because it reflects what actually happened rather than a scheduled distance.
    """
    df["TotalRaceLaps"] = df.groupby("RaceID")["LapNumber"].transform("max")
    log.info("Added : TotalRaceLaps")
    return df



# Feature 2 — RaceProgress 


def add_race_progress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RaceProgress — normalised race position on a 0.0 → 1.0 scale.
    Formula : RaceProgress = LapNumber / TotalRaceLaps
    This is the most important feature in this module. It allows the model
    to compare laps across races of different lengths on a common scale,
    which is critical for learning strategy timing patterns..
    """
    if "TotalRaceLaps" not in df.columns:
        log.warning("TotalRaceLaps not found — computing automatically.")
        df = add_total_race_laps(df)

    df["RaceProgress"] = (df["LapNumber"] / df["TotalRaceLaps"]).round(4)
    log.info("Added : RaceProgress  (0.0 → 1.0)")
    return df



# Feature 3 — RemainingLaps
    

def add_remaining_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RemainingLaps — laps left until the chequered flag.
    Formula : RemainingLaps = TotalRaceLaps - LapNumber
    Key for pit stop decision logic: a stop only makes strategic sense
    if enough laps remain to recover the time lost in the pit lane.
    """
    if "TotalRaceLaps" not in df.columns:
        log.warning("TotalRaceLaps not found — computing automatically.")
        df = add_total_race_laps(df)

    df["RemainingLaps"] = df["TotalRaceLaps"] - df["LapNumber"]
    log.info("Added : RemainingLaps")
    return df



# Feature 4 — IsFinal10Laps


def add_is_final_10_laps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsFinal10Laps — boolean flag for the last 10 laps of the race.
    Formula : IsFinal10Laps = (RemainingLaps <= 10)
    The final 10 laps represent a distinct strategic phase: teams rarely
    stop, tyre management priorities shift, and safety car probability
    heavily influences push-vs-conserve decisions.
    """
    if "RemainingLaps" not in df.columns:
        log.warning("RemainingLaps not found — computing automatically.")
        df = add_remaining_laps(df)

    df["IsFinal10Laps"] = df["RemainingLaps"] <= 10
    log.info("Added : IsFinal10Laps")
    return df



# Feature 5 — RaceHalf


def add_race_half(df: pd.DataFrame) -> pd.DataFrame:
    """Add RaceHalf — categorical label for the first or second half of the race.

    Formula : RaceHalf = "FirstHalf"  if RaceProgress < 0.5
                       = "SecondHalf" if RaceProgress >= 0.5

    The two halves have distinct strategic signatures:
    - FirstHalf  : undercut opportunities, cold tyres, heavy traffic
    - SecondHalf : tyre management, overcut attempts, end-of-race push
    """
    if "RaceProgress" not in df.columns:
        log.warning("RaceProgress not found — computing automatically.")
        df = add_race_progress(df)

    df["RaceHalf"] = df["RaceProgress"].apply(
        lambda p: "FirstHalf" if p < 0.5 else "SecondHalf"
    )
    log.info("Added : RaceHalf  (FirstHalf / SecondHalf)")
    return df



# The function that will be called by the feature engineering pipeline


def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all race-progress features in dependency order.
    Execution order (each step depends on the previous):
        TotalRaceLaps → RaceProgress → RemainingLaps → IsFinal10Laps → RaceHalf
    This is the only function the pipeline needs to call.
    """
    log.info("Running race_features module...")

    df = add_total_race_laps(df)
    df = add_race_progress(df)
    df = add_remaining_laps(df)
    df = add_is_final_10_laps(df)
    df = add_race_half(df)

    log.info("race_features done — 5 columns added.")
    return df