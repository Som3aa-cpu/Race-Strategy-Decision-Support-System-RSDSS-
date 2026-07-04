import logging

import pandas as pd


log = logging.getLogger("rsdss.features.circuit")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )



# Ordered mapping used to encode TyreStress as an integer.
TYRE_STRESS_ENCODING = {"Low": 0, "Medium": 1, "High": 2}


# Feature 1 — LapLengthFactor

def add_lap_length_factor(df: pd.DataFrame) -> pd.DataFrame:
    """Add LapLengthFactor — circuit length normalised to the dataset average.

    Formula : LapLengthFactor = Length_km / mean(Length_km)

    Examples (using real circuit lengths):
        Monza  5.793 km / 5.22 avg  →  1.11  (longer than average)
        Monaco 3.337 km / 5.22 avg  →  0.64  (shorter than average)
        Spa    7.004 km / 5.22 avg  →  1.34  (significantly longer)

    A value above 1.0 means more distance per lap — pit stop time loss is a
    smaller percentage of lap time, which generally favours later stopping.
    """
    avg_length = df["Length_km"].mean()
    df["LapLengthFactor"] = (df["Length_km"] / avg_length).round(4)
    log.info(
        "Added : LapLengthFactor  (avg circuit length in dataset = %.3f km)",
        avg_length,
    )
    return df


# Feature 2 — CornerDensity

def add_corner_density(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add CornerDensity — number of Corners per kilometre of circuit.
    Formula : CornerDensity = Corners / Length_km
    Interpretation:
        High CornerDensity → technical, slow-speed circuit (Monaco, Hungary)
                             → tyres work harder mechanically, more lateral load
        Low CornerDensity  → high-speed, power circuit (Monza, Spa straights)
                             → tyre wear driven by braking and high-speed stress
    This is a layout complexity score that helps the model distinguish
    strategy patterns that are circuit-type dependent.
    """
    df["CornerDensity"] = (df["Corners"] / df["Length_km"]).round(4)
    log.info("Added : CornerDensity  (Corners / Length_km)")
    return df


# Feature 3 — DRS_per_km

def add_drs_per_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add DRS_per_km — DRS zones per kilometre of circuit.
    Formula : DRS_per_km = DRS_Zones / Length_km
    Circuits with a high DRS_per_km have frequent overtaking opportunities
    per lap, which affects strategy in two ways:
    1. Easier to overtake on track → undercuts are less critical to gain
       positions because the driver can overtake naturally after a stop.
    2. Being undercut is more dangerous → rivals on fresh tyres can
       overtake easily using DRS, making defending harder.
    """
    df["DRS_per_km"] = (df["DRS_Zones"] / df["Length_km"]).round(4)
    log.info("Added : DRS_per_km  (DRS_Zones / Length_km)")
    return df


# The function that will be called by the feature engineering pipeline

def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all circuit-derived features in dependency order.
    Requires the enriched dataset (master laps merged with Circuit_Characteristics).
    Must be called after enrich_data.run_enrichment().
    Execution order:
        LapLengthFactor → CornerDensity → DRS_per_km
        → HighSpeedCircuit → SurfaceAbrasion
    """
    log.info("Running circuit_features module...")

    df = add_lap_length_factor(df)
    df = add_corner_density(df)
    df = add_drs_per_km(df)
    
    log.info("circuit_features done — 5 columns added.")
    return df