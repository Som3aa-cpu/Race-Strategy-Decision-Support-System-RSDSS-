import logging

import pandas as pd


log = logging.getLogger("rsdss.features.temperature")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

# Thresholds
HOT_TRACK_THRESHOLD = 40   # °C
HOT_AIR_THRESHOLD   = 30   # °C

# TempCategory boundaries based on TrackTemp
COLD_UPPER   = 25   # TrackTemp < 25   → Cold
HOT_LOWER    = 40   # TrackTemp >= 40  → Hot
              #      25 <= TrackTemp < 40 → Medium


# Feature 1 — TempDifference

def add_temp_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TempDifference — the gap between track surface and air temperature.
    Formula : TempDifference = TrackTemp - AirTemp
    The track is almost always hotter than the air due to solar radiation
    and tarmac heat absorption. A large difference signals high track
    rubber degradation and reduced grip in the first few laps after a stop
    as tyres struggle to reach optimal operating temperature quickly.
    Typical range: +5°C (cool/overcast) to +25°C (hot sunny race).
    """
    df["TempDifference"] = df["TrackTemp"] - df["AirTemp"]
    log.info("Added : TempDifference  (TrackTemp - AirTemp)")
    return df


# Feature 2 — IsHotTrack

def add_is_hot_track(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsHotTrack — boolean flag for track temperatures above 40°C.
    Formula : IsHotTrack = (TrackTemp > 40)
    Above 40°C the tarmac surface significantly accelerates tyre wear,
    particularly for Soft compounds.  Teams at hot-track races typically
    shift one compound step harder and shorten their stint targets.
    """
    df["IsHotTrack"] = df["TrackTemp"] > HOT_TRACK_THRESHOLD
    log.info("Added : IsHotTrack  (TrackTemp > %d°C)", HOT_TRACK_THRESHOLD)
    return df


# Feature 3 — IsHotAir

def add_is_hot_air(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add IsHotAir — boolean flag for ambient air temperatures above 30°C.
    Formula : IsHotAir = (AirTemp > 30)
    Hot ambient air affects engine cooling, brake temperatures, and driver
    fatigue — all of which influence pace consistency over long stints.
    Combined with IsHotTrack it creates a complete picture of thermal stress.
    """
    df["IsHotAir"] = df["AirTemp"] > HOT_AIR_THRESHOLD
    log.info("Added : IsHotAir  (AirTemp > %d°C)", HOT_AIR_THRESHOLD)
    return df


# Feature 4 — TempCategory

def add_temp_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TempCategory — a three-level label based on track temperature.
    Boundaries (based on TrackTemp):
        Cold   : TrackTemp < 25°C
        Medium : 25°C <= TrackTemp < 40°C
        Hot    : TrackTemp >= 40°C
    These boundaries roughly correspond to real F1 tyre operating windows:
    - Cold races (Silverstone, Zandvoort in poor weather) favour Softs
      to generate tyre temperature; degradation is minimal.
    - Hot races (Bahrain, Abu Dhabi, Qatar) punish Softs and reward
      conservative compound selection.
    - Medium races represent the majority of the calendar.
    """
    df["TempCategory"] = pd.cut(
        df["TrackTemp"],
        bins=[-float("inf"), COLD_UPPER, HOT_LOWER, float("inf")],
        labels=["Cold", "Medium", "Hot"],
        right=False,
    ).astype("object")       # convert from Categorical to plain string

    dist = df["TempCategory"].value_counts().to_dict()
    log.info(
        "Added : TempCategory  (Cold / Medium / Hot)  |  distribution: %s", dist
    )
    return df


# the function that will be called by the feature engineering pipeline

def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Apply all temperature features in dependency order.
    Execution order:
        TempDifference → IsHotTrack → IsHotAir → TempCategory
    """
    log.info("Running temperature_features module...")

    df = add_temp_difference(df)
    df = add_is_hot_track(df)
    df = add_is_hot_air(df)
    df = add_temp_category(df)

    log.info("temperature_features done — 4 columns added.")
    return df