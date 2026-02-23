# imported from step 2
from __future__ import annotations
import numpy as np
import pandas as pd
import pyproj
from pykrige.ok import OrdinaryKriging

#################################
# ESTIMATED CORRELATION FUNCTIONS
#################################

def ba_anom_spatialcorr(d: np.ndarray):
    """
    Spatial correlation of annual anomaly in mass balance (this study = Dussaillant et al., 2024).

    :param d: Distance between two glaciers (meters).

    :return: Spatial correlation function (input = distance in meters, output = correlation between 0 and 1).
    """

    # Three ranges and partial sill for three exponential models (first range = nugget)

    r1 = 8.79259877e+01
    r2 = 2.06635202e+05
    r3 = 5.000000e+06
    ps1 = 0.04484324
    ps2 = 0.36955494
    ps3 = 0.58560182

    exp1 = ps1 * (1 - np.exp(-3 * d / r1))
    exp2 = ps2 * (1 - np.exp(-3 * d / r2))
    exp3 = ps3 * (1 - np.exp(-3 * d / r3))

    # Spatial correlation
    return 1 - (exp1 + exp2 + exp3)

############################################################################
# COORDINATE TRANSFORMATIONS (FOR SPEED COMPUTING DISTS DURING ERROR PROPAG)
############################################################################

def latlon_to_utm(lat: float, lon: float) -> str:
    """
    Get UTM zone for a given latitude and longitude coordinates.

    :param lat: Latitude coordinate.
    :param lon: Longitude coordinate.

    :returns: UTM zone.
    """

    if not (
        isinstance(lat, (float, np.floating, int, np.integer))
        and isinstance(lon, (float, np.floating, int, np.integer))
    ):
        raise TypeError("Latitude and longitude must be floats or integers.")

    if not -180 <= lon < 180:
        raise ValueError("Longitude value is out of range [-180, 180[.")
    if not -90 <= lat < 90:
        raise ValueError("Latitude value is out of range [-90, 90[.")

    # Get UTM zone from name string of crs info
    utm_zone = pyproj.database.query_utm_crs_info(
        "WGS 84", area_of_interest=pyproj.aoi.AreaOfInterest(lon, lat, lon, lat)
    )[0].name.split(" ")[-1]

    return str(utm_zone)


def utm_to_epsg(utm: str) -> int:
    """
    Get EPSG code of UTM zone.

    :param utm: UTM zone.

    :return: EPSG of UTM zone.
    """

    if not isinstance(utm, str):
        raise TypeError("UTM zone must be a str.")

    # Whether UTM is passed as single or double digits, homogenize to single-digit
    utm = str(int(utm[:-1])) + utm[-1].upper()

    # Get corresponding EPSG
    epsg = pyproj.CRS(f"WGS 84 / UTM Zone {utm}").to_epsg()

    return int(epsg)


def reproject_points(
    points: list[list[float]] | list[float] | tuple[list[float], list[float]] | np.ndarray, in_crs: pyproj.CRS, out_crs: pyproj.CRS
) -> tuple[list[float], list[float]]:
    """
    Reproject a set of point from input_crs to output_crs.

    :param points: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param in_crs: Input CRS
    :param out_crs: Output CRS

    :returns: Reprojected points, of same shape as points.
    """
    assert np.shape(points)[0] == 2, "points must be of shape (2, N)"

    x, y = points
    transformer = pyproj.Transformer.from_crs(in_crs, out_crs)
    xout, yout = transformer.transform(x, y)
    return (xout, yout)


def reproject_from_latlon(
    points: list[list[float]] | tuple[list[float], list[float]] | np.ndarray, out_crs: pyproj.CRS, round_: int = 2
) -> tuple[list[float], list[float]]:
    """
    Reproject a set of point from lat/lon to out_crs.

    :param points: Input points to be reprojected. Must be of shape (2, N), i.e (x coords, y coords)
    :param out_crs: Output CRS
    :param round_: Output rounding. Default of 2 ensures cm accuracy

    :returns: Reprojected points, of same shape as points.
    """
    crs_4326 = pyproj.CRS.from_epsg(4326)
    proj_points = reproject_points(points, crs_4326, out_crs)
    proj_points = np.round(proj_points, round_)
    return proj_points


##################################################
# SPATIAL INTERPOLATION WITH CORRELATION = KRIGING
##################################################

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

    def variogram_func(placeholder, d: np.ndarray):
        """The variogram is the variance minus the covariance, and the covariance is the variance times correlation."""
        return var * (1 - ba_anom_spatialcorr(d))

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

    # Get median latitude and longitude among all values
    med_lat = np.median(df_obs.lat)
    med_lon = np.median(df_obs.lon)

    # Find the metric (UTM) system centered on these coordinates
    utm_zone = latlon_to_utm(med_lat, med_lon)
    epsg = utm_to_epsg(utm_zone)

    # Reproject latitude and longitude to easting/northing
    easting, northing = reproject_from_latlon((df_obs.lat, df_obs.lon),
                                              out_crs=pyproj.CRS.from_epsg(epsg))
    easting_pred, northing_pred = reproject_from_latlon((df_pred.lat, df_pred.lon),
                                              out_crs=pyproj.CRS.from_epsg(epsg))
    # Extract anomalies
    ba_anom = df_obs.ba_anom

    return krige_ba_anom(xobs=easting, yobs=northing, ba_anom_obs=ba_anom, xpred=easting_pred, ypred=northing_pred, var_anom=var_anom)

