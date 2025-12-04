from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.spatial.distance import cdist, squareform, pdist
import pandas as pd
import pyproj
from pykrige.ok import OrdinaryKriging

#################################
# ESTIMATED CORRELATION FUNCTIONS
# ################################

@np.vectorize
def sig_rho_dv_spatialcorr(d: np.ndarray, dt: float):
    """
    Spatial correlation of error of density of volume change (Huss et al., in prep).

    :param d: Distance between two glaciers (meters).
    :param dt: Period length (years).

    :return: Spatial correlation function (input = distance in meters, output = correlation between 0 and 1).
    """

    # Parameters estimated in Huss et al. (in prep)
    if d == 0:
        return 1

    # Two ranges and four parameters to determine partial sills varying with period length
    r1 = 200000
    r2 = 5000000

    f = 0.1709584
    g = 0.725352
    h = 0.32859605
    i = 0.89780742

    # Spatial correlation
    return (np.exp(-f*dt**g) - np.exp(-h*dt**i)) * np.exp(-3*d/r1) + np.exp(-h*dt**i) * np.exp(-3*d/r2)


def _global_average_sig_dh_spatialcorr_hugonnet():
    """
    Estimate the parameters for a globally-averaged spatial correlation of dh errors, based on the average
    distribution of closest time to ASTER observation.
    """

    # Correlation lengths, calibrated with ICESat globally
    corr_ranges = [150, 2000, 5000, 20000, 50000, 500000]

    # Corresponding coefficients for a function SQRT(variance) = A * deltaT + B (ED Fig. 5b)
    # with deltaT = time lag to closest ASTER observation
    # Same as: https://github.com/iamdonovan/pyddem/blob/2fbd54049cf90f77076ad62ba5e1b7cf179cf52e/pyddem/volint_tools.py#L260
    coefs = [np.array([1.26694247e-03, 3.03486839e+00]),
             np.array([1.35708936e-03, 4.05065698e+00]),
             np.array([1.42572733e-03, 4.20851582e+00]),
             np.array([1.82537137e-03, 4.28515920e+00]),
             np.array([1.87250755e-03, 4.31311254e+00]),
             np.array([2.06249620e-03, 4.33582812e+00])]

    # We want to average the above function for a generic glacier = mix of closest time lags
    # We use the global distribution of time lags of ED Fig. 4d (time lag to closest)
    # Medium bins of delta t (in days to closest obs)
    delta_t_lags = np.array([30, 90, 150, 210, 270, 330, 450, 630, 810, 990])
    # Percentage of total data (eye-balling from the histogram of ED Fig. 4, a few % won't change much anyway)
    perc_lags = [20, 25, 15, 7, 5, 5, 7, 5, 3, 3]
    perc_lags = np.array(perc_lags) / 100

    # Get the time lag weighted-averaged variance for each correlation range
    vars = [np.sum((c[0] * delta_t_lags + c[1])**2 * perc_lags) / np.sum(perc_lags) for c in coefs]

    # The first three ranges are useful for propagating errors from pixels, but don't matter for regional size
    vars = vars[3:]

    # Finally, we standardize the sills to be equal to 1 and get a correlation
    vars /= sum(vars)

def sig_dh_spatialcorr(d: np.ndarray):
    """
    Spatial correlation of error in mean elevation change (Hugonnet et al., 2021).

    :param d: Distance between two glaciers (meters).

    :return: Spatial correlation function (input = distance in meters, output = correlation between 0 and 1).
    """

    # Results of function "_global_average_sig_dh_spatialcorr_hugonnet" right above, hard-coding outputs here to avoid
    # repeating the calculation everytime this function is called, for speed

    # About a third of the partial sill is correlated to 20km, another third to 50km and the last third to 500km
    ps1 = 0.326432
    ps2 = 0.33190818
    ps3 = 0.34165983
    r1 = 20000
    r2 = 50000
    r3 = 500000

    exp1 = ps1 * (1 - np.exp(-3 * d / r1))
    exp2 = ps2 * (1 - np.exp(-3 * d / r2))
    exp3 = ps3 * (1 - np.exp(-3 * d / r3))

    # Spatial correlation
    return 1 - (exp1 + exp2 + exp3)


def ba_anom_spatialcorr(d: np.ndarray):
    """
    Spatial correlation of annual anomaly in mass balance (this study = Dussaillant et al., 2024).

    :param d: Distance between two glaciers (meters).

    :return: Spatial correlation function (input = distance in meters, output = correlation between 0 and 1).
    """

    # Three ranges and partial sill for three exponential models
    r1 = 5.051675e+03
    r2 = 9.985127e+04
    r3 = 5.000000e+06
    ps1 = 0.224308
    ps2 = 0.140278
    ps3 = 0.635414

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

#############################
# ERROR PROPAGATION FUNCTIONS
#############################

def double_sum_covar(coords: np.ndarray, errors: np.ndarray, spatialcorr_func: Callable[[np.ndarray], np.ndarray]):
    """
    Error propagation with spatial correlation, i.e. a double sum of covariance that accepts any correlation
    function and expects euclidean coordinates.

    Exact for a small sample size, approximated for a large sample size.

    :param coords: Center glacier coordinates, array with size (N,2).
    :param errors: Errors at the glacier coordinates, array with size (N,).
    :param spatialcorr_func: Spatial correlation function (input = distance in unit of the coordinates,
        output = correlation between 0 and 1).

    :return: Total variance from double sum.
   """

    # Define random state to subset for speed
    rng = np.random.default_rng(42)

    # Length of coordinates and subsample size
    n = len(coords)
    subsample_size = 10000

    # At maximum, the number of subsamples has to be equal to number of points
    subsample_size = min(subsample_size, n)

    # Get random subset of points for one of the sums
    rand_points = rng.choice(n, size=subsample_size, replace=False)

    # Subsample coordinates in 1D
    sub_coords = coords[rand_points, :]
    sub_errors = errors[rand_points]

    # Compute pairwise distance between all points
    pds_matrix = cdist(coords, sub_coords, "euclidean")

    # Convert the compact pairwise distance form into a square matrix (N, N)
    # pds = pdist(coords)
    # pds_matrix = squareform(pds)

    # Vectorize double sum calculation for speed
    # First, we compute a matrix of all pairwise errors accounting for spatial correlation
    mat_var = errors.reshape((-1, 1)) @ sub_errors.reshape((1, -1)) * spatialcorr_func(pds_matrix.flatten()).reshape(pds_matrix.shape)
    # Then we sum everything and scale by the sample size
    var = np.sum(mat_var) * n ** 2 / (n * subsample_size)

    return np.sqrt(var)


def wrapper_latlon_double_sum_covar(df: pd.DataFrame, spatialcorr_func: Callable[[np.ndarray], np.ndarray]):
    """
    Wrapper to compute the double sum of covariance from a dataframe with lat/lon coordinates.

    :param df: Dataframe with "lat", "lon" and "errors" columns for each point.
    :param spatialcorr_func: Spatial correlation function (input = distance in unit of the coordinates,
        output = correlation between 0 and 1).

    :return: Total variance from double sum.
    """

    # Get median latitude and longitude among all values
    med_lat = np.median(df.lat.values)
    med_lon = np.median(df.lon.values)


    # Find the metric (UTM) system centered on these coordinates
    utm_zone = latlon_to_utm(med_lat, med_lon)
    epsg = utm_to_epsg(utm_zone)

    # Reproject latitude and longitude to easting/northing
    easting, northing = reproject_from_latlon((df.lat.values, df.lon.values),
                                              out_crs=pyproj.CRS.from_epsg(epsg))
    coords = np.array([easting, northing]).T

    # Extract errors
    errors = df.errors.values

    return double_sum_covar(coords=coords, errors=errors, spatialcorr_func=spatialcorr_func)


##################################################
# SPATIAL INTERPOLATION WITH CORRELATION = KRIGING
##################################################

def krige_ba_anom(xobs: np.ndarray, yobs: np.ndarray, ba_anom_obs: np.ndarray, xpred: np.ndarray, ypred: np.ndarray):
    """
    Interpolate annual mass balance anomaly using kriging.

    :param xobs: X coordinates of observed glaciers.
    :param yobs: Y coordinates of observed glaciers.
    :param ba_anom_obs: Annual mass balance anomalies of observed glaciers.
    :param xpred: X coordinates of glaciers to predict.
    :param ypred: Y coordinates of glaciers to predict.

    :return: Annual mass balance anomalies of glacier to predict, Error (1-sigma) of predicted anomalies.
    """

    # TODO: Either pass standardized anomalies, or multiply model by variance here

    # Ordinary kriging = kriging with a mean function
    OK = OrdinaryKriging(
        xobs,
        yobs,
        ba_anom_obs,
        variogram_model="custom",
        variogram_parameters=[],
        variogram_function=ba_anom_spatialcorr,
        verbose=False,
        enable_plotting=False,
    )

    # Predict on grid, with uncertainty
    ba_anom_pred, sig_anom_pred = OK.execute("points", xpred, ypred)

    return ba_anom_pred, sig_anom_pred
