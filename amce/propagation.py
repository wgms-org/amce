from typing import Tuple, Union

import numpy as np
import pyproj.aoi
import pyproj.database
import scipy.spatial.distance


def geographic_to_utm_zone(latitude: float, longitude: float) -> int:
    """Get UTM zone of geographic coordinates as EPSG code."""
    if abs(latitude) > 84 or abs(longitude) > 180:
        raise ValueError(f'Outside range [-84, 84], [-180, 180]: {latitude}, {longitude}')
    aoi = pyproj.aoi.AreaOfInterest(longitude, latitude, longitude, latitude)
    datum = pyproj.database.query_utm_crs_info(datum_name='WGS84', area_of_interest=aoi)[0]
    return int(datum.code)


def geographic_to_utm(
    latitude: np.ndarray,
    longitude: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project geographic coordinates to UTM."""
    epsg = geographic_to_utm_zone(np.median(latitude), np.median(longitude))
    transformer = pyproj.Transformer.from_crs(crs_from=4326, crs_to=epsg)
    return transformer.transform(latitude, longitude)


def geodetic_to_grid_cell(
    latitude: np.ndarray,
    longitude: np.ndarray,
    origin: Tuple[float, float] = (90.0, -180.0),
    size: Tuple[float, float] = (0.5, 0.5)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert geodetic coordinates to grid cell indices.

    Parameters
    ----------
    latitude
        Latitude in degrees (n,).
    longitude
        Longitude in degrees (n,).
    box
        Bounding box as (min latitude, min longitude, max latitude, max longitude).
    size
        Grid cell size as (latitude size, longitude size) in degrees.

    Returns
    -------
    i
        Grid cell row index (n,).
    j
        Grid cell column index (n,).
    """
    return (
        np.floor((origin[0] - latitude) / size[0]).astype(int),
        np.floor((longitude - origin[1]) / size[1]).astype(int)
    )


def geographic_to_distance(
    latitude: np.ndarray,
    longitude: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise distances between geographic coordinates projected to UTM.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    """
    x, y = geographic_to_utm(latitude, longitude)
    return scipy.spatial.distance.pdist(np.column_stack((x, y)), metric='euclidean')


def distance_to_elevation_change_correlation(distance: np.ndarray) -> np.ndarray:
    """
    Convert distance to a spatial correlation of elevation change errors.

    Based on Hugonnet et al. (2021).
    A third of the partial sill is correlated to 20 km, another third to 50 km and the
    last third to 500 km.

    Parameters
    ----------
    distance
        Pairwise glacier distance (meters).

    Returns
    -------
    Spatial correlation coefficient (between 0 and 1).
    """
    ps1 = 0.326432
    ps2 = 0.33190818
    ps3 = 0.34165983
    r1 = 20000
    r2 = 50000
    r3 = 500000
    # 1 - (ps1 * (1 - e(-3/r1 * d)) + ps2 * (1 - e(-3/r2 * d)) + ps3 * (1 - e(-3/r3 * d)))
    return (
        1 - (ps1 + ps2 + ps3) +
        ps1 * np.exp(-3 / r1 * distance) +
        ps2 * np.exp(-3 / r2 * distance) +
        ps3 * np.exp(-3 / r3 * distance)
    )


def distance_to_density_correlation(distance: np.ndarray, duration: float = 1.0) -> np.ndarray:
    """
    Convert distance to a spatial correlation of density errors.

    Based on Huss and Hock (2015).

    Parameters
    ----------
    distance
        Pairwise glacier distance (meters).
    duration
        Time duration (years).

    Returns
    -------
    Spatial correlation coefficient (between 0 and 1).
    """
    r1 = 200000
    r2 = 5000000
    f = 0.1709584
    g = 0.725352
    h = 0.32859605
    i = 0.89780742
    correlation = (
        (
            np.exp(-f * duration ** g) - np.exp(-h * duration ** i)
        ) * np.exp(-3 / r1 * distance) +
        np.exp(-h * duration ** i) * np.exp(-3 / r2 * distance)
    )
    return np.where(distance == 0, 1.0, correlation)


def distance_to_mass_balance_anomaly_correlation(distance: np.ndarray) -> np.ndarray:
    """
    Convert distance to a spatial correlation of mass balance anomaly errors.

    Based on Dussaillant et al. (2024).

    Parameters
    ----------
    distance
        Pairwise glacier distance (meters).

    Returns
    -------
    Spatial correlation coefficient (between 0 and 1).
    """
    r1 = 5051.675
    r2 = 99851.27
    r3 = 5000000
    ps1 = 0.224308
    ps2 = 0.140278
    ps3 = 0.635414
    # 1 - (ps1 * (1 - e(-3/r1 * d)) + ps2 * (1 - e(-3/r2 * d)) + ps3 * (1 - e(-3/r3 * d)))
    return (
        1 - (ps1 + ps2 + ps3) +
        ps1 * np.exp(-3 / r1 * distance) +
        ps2 * np.exp(-3 / r2 * distance) +
        ps3 * np.exp(-3 / r3 * distance)
    )


def calculate_ij_of_condensed_pairwise(condensed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate i, j indices of condensed pairwise array.

    Parameters
    ----------
    condensed
        Condensed pairwise array (as from scipy.spatial.distance.pdist).

    Returns
    -------
    i
        Row indices of the pairwise matrix.
    j
        Column indices of the pairwise matrix.
    """
    if len(condensed) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n = scipy.spatial.distance.num_obs_y(condensed)
    return np.triu_indices(n, k=1)


def regional_sigma(
    sigma: np.ndarray,
    correlation: np.ndarray,
    ij: Tuple[np.ndarray, np.ndarray] = None
) -> np.ndarray:
    """
    Propagate glacier sigma to regional scale.

    Parameters
    ----------
    sigma
        Glacier sigma by year (n years, m glaciers) or (m glaciers, ).
    correlation
        Pairwise glacier correlation (condensed form).
    ij
        Precomputed row and column indices of the correlation matrix.
    """
    i, j = ij or calculate_ij_of_condensed_pairwise(correlation)
    # Expand sigma to match pairwise correlation
    sigma = np.atleast_2d(sigma)
    # Compute pairwise terms incrementally to save memory
    pairs = correlation * sigma[:, i]
    pairs *= sigma[:, j]
    self = sigma ** 2
    return np.sqrt(np.sum(self, axis=1) + 2 * np.sum(pairs, axis=1))


def choose_sample(a: Union[int, np.ndarray], size: int) -> np.ndarray:
    """
    Get random sample indices without replacement.

    Parameters
    ----------
    a
        Population or population size.
    size
        Sample size.
    """
    generator = np.random.default_rng(42)
    return generator.choice(a=a, size=size, replace=False, shuffle=False)


def geographic_to_distance_sample(
    latitude: np.ndarray,
    longitude: np.ndarray,
    sample: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise distances between geographic coordinates projected to UTM.

    All original points are compared to a sample of points.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """
    x, y = geographic_to_utm(latitude, longitude)
    coords = np.column_stack((x, y))
    return scipy.spatial.distance.cdist(coords, coords[sample], metric='euclidean')


def regional_sigma_sample(
    sigma: np.ndarray,
    correlation: np.ndarray,
    sample: np.ndarray
) -> np.ndarray:
    """
    Propagate glacier sigma to regional scale.

    Parameters
    ----------
    sigma
        Glacier sigma (m glaciers, ).
    correlation
        Glacier correlation (m glaciers, n glaciers).
    sample
        Sample indices (n glaciers, ).
    """
    m, n = correlation.shape
    pairs = sigma.reshape((-1, 1)) @ sigma[sample].reshape((1, -1)) * correlation
    # Sum pairwise error terms and scale by the total number of glaciers
    return np.atleast_1d(np.sqrt(np.sum(pairs) * m**2 / (m * n)))


def regional_sigma_wrapper(
    latitude: np.ndarray,
    longitude: np.ndarray,
    sigma_dh: np.ndarray,
    sigma_rho: np.ndarray,
    sigma_anom: np.ndarray,
    by_year: bool = True,
    sample_threshold: int = 30000,
    sample_size: int = 10000,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper to compute regional sigmas.

    Parameters
    ----------
    latitude
        Glacier latitudes (m glaciers, ).
    longitude
        Glacier longitudes (m glaciers, ).
    sigma_dh
        Glacier elevation change sigma by year (n years, m glaciers) or (m glaciers, ).
    sigma_rho
        Glacier density sigma by year (n years, m glaciers) or (m glaciers, ).
    sigma_anom
        Glacier mass balance anomaly sigma by year (n years, m glaciers) or (m glaciers, ).
    by_year
        Whether to compute regional sigma for each year separately (lower memory usage).
    sample_threshold
        Number of glaciers above which to use sampling for distance computation.
    sample_size
        Number of glaciers to use, in which case m glaciers x sample_size glaciers
        are used for error propagation.
    verbose
        Whether to print progress information.
    """
    n = len(latitude)
    is_sampled = n > sample_threshold
    if is_sampled:
        if verbose:
            print(f'[INFO] Computing distance from {n} to {sample_size} sample glaciers')
        sample = choose_sample(a=n, size=sample_size)
        distances = geographic_to_distance_sample(
            latitude=latitude,
            longitude=longitude,
            sample=sample
        )
        by_year = True
        regional_func = regional_sigma_sample
        kwargs = {'sample': sample}
    else:
        # TODO: Use uint32 (or uint16 as km) to reduce memory
        distances = geographic_to_distance(latitude=latitude, longitude=longitude)
        # TODO: Use uint32 (or uint16 if regions are small enough) to reduce memory
        ij = calculate_ij_of_condensed_pairwise(distances)
        regional_func = regional_sigma
        kwargs = {'ij': ij}
    variables = [
        ('dh', sigma_dh, distance_to_elevation_change_correlation),
        ('rho', sigma_rho, distance_to_density_correlation),
        ('anom', sigma_anom, distance_to_mass_balance_anomaly_correlation)
    ]
    results = []
    for name, sigma, correlation_func in variables:
        if verbose:
            print(f'[INFO] Spatial correlation for {name}')
        # TODO: Use float32 to reduce memory
        correlation = correlation_func(distances)
        all_equal = (sigma == sigma[0]).all()
        if not all_equal and name in ('dh', 'rho') and verbose:
            print(f'[WARNING] sigma_{name} not equal for all years')
        if all_equal:
            result = regional_func(
                sigma=sigma[0],
                correlation=correlation,
                **kwargs
            ).tolist()
            result = result * sigma.shape[0]
        elif by_year:
            result = []
            for i, row in enumerate(sigma, start=1):
                if verbose:
                    print(f'year (of {sigma.shape[0]}): {i}', end='\r')
                result += regional_func(
                    sigma=row,
                    correlation=correlation,
                    **kwargs
                ).tolist()
            if verbose:
                print('')
        else:
            result = regional_func(
                sigma=sigma,
                correlation=correlation,
                **kwargs
            )
        results.append(np.asarray(result))
    return tuple(results)
