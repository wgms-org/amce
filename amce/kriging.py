import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

import amce.propagation


def krige_ba_anom(xobs: np.ndarray, yobs: np.ndarray, ba_anom_obs: np.ndarray, xpred: np.ndarray, ypred: np.ndarray, var_anom: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate annual mass balance anomaly using kriging.

    :param xobs: X coordinates of observed glaciers.
    :param yobs: Y coordinates of observed glaciers.
    :param ba_anom_obs: Annual mass balance anomalies of observed glaciers.
    :param xpred: X coordinates of glaciers to predict.
    :param ypred: Y coordinates of glaciers to predict.
    :param var_anom: Variance of anomalies for the region.

    :return: Annual mass balance anomalies of glacier to predict, Error (1-sigma) of predicted anomalies.
    """

    # We have a spatial correlation function based on standardized anomalies, so we need to de-standardize it here
    # to derive a variogram equivalent
    var = var_anom
    # If sample size too small, replace by global average variance of annual mass balance anomaly (Huss et al.)
    # In mm w.e. yr-1, the STD of annual anomalies globally is 470, so we take its square
    # var = 470**2

    def variogram_func(_, distance: np.ndarray):
        """The variogram is the variance minus the covariance, and the covariance is the variance times correlation."""
        return var * (
            1 - amce.propagation.distance_to_mass_balance_anomaly_correlation(distance)
        )

    # If there is only a single observation, kriging fails
    if len(xobs) == 1:

        # For the mean, we return the valid observation
        ba_anom_pred = np.repeat(ba_anom_obs, len(xpred))

        # For the error: we compute the distance to that unique value, and use the correlation function
        xdiff = xpred - xobs[0]
        ydiff = ypred - yobs[0]
        dist_matrix = np.sqrt(xdiff**2 + ydiff**2)
        sig_anom_pred = np.sqrt(variogram_func(0, dist_matrix))

        return ba_anom_pred, sig_anom_pred

    else:
        # Ordinary kriging = kriging with a mean function
        OK = OrdinaryKriging(
            np.atleast_1d(xobs),
            np.atleast_1d(yobs),
            np.atleast_1d(ba_anom_obs),
            variogram_model="custom",
            variogram_parameters=[1],  # Placeholder argument that is not used (otherwise PyKrige doesn't like empty list)
            variogram_function=variogram_func,
            verbose=False,
            enable_plotting=False,
        )

        # Predict on grid, with uncertainty
        ba_anom_pred, sigsq_anom_pred = OK.execute("points", np.atleast_1d(xpred), np.atleast_1d(ypred))
        sig_anom_pred = np.sqrt(sigsq_anom_pred)

        return ba_anom_pred, sig_anom_pred


def wrapper_latlon_krige_ba_anom(df_obs: pd.DataFrame,
                                 df_pred: pd.DataFrame,
                                 var_anom: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for kriging with dataframe input, converting lat/lon coordinates in a local metric projection.

    :param df_obs: Dataframe containing "lat", "lon", "ba_anom" for inputs series.
    :param df_pred: Dataframe containing "lat" and "lon" for points where to predict.
    :param var_anom: Variance of anomalies to de-standardize variogram per region.

    :return: Annual mass balance anomalies of glacier to predict, Error (1-sigma) of predicted anomalies.
    """
    # Reproject latitude and longitude to easting/northing
    easting, northing = amce.propagation.geographic_to_utm(
        latitude=df_obs.lat, longitude=df_obs.lon
    )
    easting_pred, northing_pred = amce.propagation.geographic_to_utm(
        latitude=df_pred.lat, longitude=df_pred.lon
    )
    return krige_ba_anom(
        xobs=easting,
        yobs=northing,
        ba_anom_obs=df_obs.ba_anom,
        xpred=easting_pred,
        ypred=northing_pred,
        var_anom=var_anom
    )
