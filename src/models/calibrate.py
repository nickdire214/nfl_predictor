"""Model calibration module.

Implements in-season rolling calibration via shrinkage: an additive
offset, derived from the mean prediction error observed so far this
season, is subtracted from raw model predictions. The offset is
shrunk toward zero when little data has been observed (n small
relative to n0) and approaches the full mean error as the season
progresses (n large relative to n0).

Validated via walk-forward grid search (see scripts/test_rolling_calibration_v2.py):
n0=150 roughly halves summed |bias| across 4 seasons at ~zero MAE cost.
"""

import numpy as np


class RollingCalibrator:
    """Tracks in-season prediction errors and computes a shrunk offset.

    offset = mean(errors) * n / (n + n0)

    where n is the number of completed-game errors observed so far this
    season, and errors are defined as (predicted - actual). Subtracting
    the offset from a raw prediction nudges it toward the in-season
    observed bias, with the correction strength growing smoothly from
    0 (n=0) toward the full mean error as n grows relative to n0.
    """

    def __init__(self, n0=150):
        self.n0 = n0
        self.errors = []

    def update(self, predicted, actual):
        """Append completed-game error(s) (predicted - actual) for the current season.

        Accepts scalars or array-likes.
        """
        predicted = np.atleast_1d(predicted)
        actual = np.atleast_1d(actual)
        self.errors.extend((predicted - actual).tolist())

    @property
    def offset(self):
        """Current shrunk offset. 0.0 if no errors have been observed yet."""
        n = len(self.errors)
        if n == 0:
            return 0.0
        mean_error = np.mean(self.errors)
        return mean_error * n / (n + self.n0)

    def calibrate(self, predictions):
        """Return predictions minus the current offset."""
        return np.asarray(predictions) - self.offset

    def reset(self):
        """Clear accumulated errors at a season boundary."""
        self.errors = []
