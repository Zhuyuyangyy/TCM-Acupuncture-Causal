"""Acupuncture Target Trial Emulation."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AcupunctureProtocol:
    """Protocol specification for an acupuncture target trial emulation.

    Attributes
    ----------
    condition : str
        Clinical condition under study, e.g. "chronic_pain".
    eligibility : dict
        Eligibility criteria expressed as column-level rules.  Each key is a
        column name in the patient data and each value is a dict with optional
        ``"min"`` / ``"max"`` bounds.  Example::

            {"age": {"min": 18, "max": 80}, "baseline_vas": {"min": 3}}

    treatment : str
        Description or identifier for the treatment arm.
    control : str
        Description or identifier for the control arm.
    primary_outcome : str
        Name of the primary outcome column.
    follow_up_weeks : int
        Duration of follow-up in weeks (default 12).
    time_zero_col : str | None
        Column used to define time-zero (treatment initiation).  If ``None``
        the time-zero step is a no-op.
    """

    condition: str
    eligibility: dict
    treatment: str
    control: str
    primary_outcome: str
    follow_up_weeks: int = 12
    time_zero_col: Optional[str] = None


class AcupunctureTargetTrial:
    """针灸目标试验模拟器

    Implements the key design elements of target trial emulation:
    *defining time-zero* and *applying eligibility criteria* so that
    observational data is structured as-if it came from a randomised trial.
    """

    def __init__(self, protocol: AcupunctureProtocol):
        self.protocol = protocol

    # ------------------------------------------------------------------
    # Step 1: Define time-zero
    # ------------------------------------------------------------------
    def define_time_zero(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign or align each record to its time-zero event.

        If the protocol specifies a ``time_zero_col`` the values from that
        column are preserved.  Otherwise this step is a no-op and the data is
        returned unchanged (the caller is responsible for ensuring that each
        row already represents a valid time-zero snapshot).

        Parameters
        ----------
        data : pd.DataFrame
            Patient-level data.

        Returns
        -------
        pd.DataFrame
            Copy of *data* with a ``time_zero`` column added.
        """
        data = data.copy()
        tz_col = self.protocol.time_zero_col
        if tz_col and tz_col in data.columns:
            data["time_zero"] = data[tz_col]
        else:
            # Treat each row as its own time-zero (default behaviour for
            # pre-processed data where every record already represents a
            # treatment initiation point).
            data["time_zero"] = 0
        return data

    # ------------------------------------------------------------------
    # Step 2: Apply eligibility criteria
    # ------------------------------------------------------------------
    def apply_eligibility(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter the cohort to patients who meet the protocol eligibility.

        Each entry in ``protocol.eligibility`` maps a column name to a dict
        with optional ``"min"`` and ``"max"`` keys.  Rows whose value falls
        outside the specified range are dropped.

        Parameters
        ----------
        data : pd.DataFrame
            Patient-level data (typically output of :meth:`define_time_zero`).

        Returns
        -------
        pd.DataFrame
            Filtered copy of *data* containing only eligible patients.
        """
        mask = pd.Series(True, index=data.index)
        for col, criteria in self.protocol.eligibility.items():
            if col not in data.columns:
                continue
            if "min" in criteria:
                mask &= data[col] >= criteria["min"]
            if "max" in criteria:
                mask &= data[col] <= criteria["max"]
        return data.loc[mask].reset_index(drop=True)
