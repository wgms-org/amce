from pathlib import Path
import re
from typing import Dict
import zipfile

import cartopy.crs
import cartopy.feature
import jinja2
import matplotlib
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot
import pandas as pd
import plotnine as p9
import numpy as np
import xarray

import amce.helpers
import amce.propagation


PUBLISH_DIR: Path = Path('publish')

REGION_NAMES: Dict[str, str] = {
  'ALA': 'Alaska',
  'WNA': 'Western Canada & USA',
  'ACN': 'Arctic Canada North',
  'ACS': 'Arctic Canada South',
  'GRL': 'Greenland',
  'ISL': 'Iceland',
  'SJM': 'Svalbard & Jan Mayen',
  'SCA': 'Scandinavia',
  'RUA': 'Russian Arctic',
  'ASN': 'Asia North',
  'CEU': 'Central Europe',
  'CAU': 'Caucasus & Middle East',
  'ASC': 'Asia Central',
  'ASW': 'Asia South West',
  'ASE': 'Asia South East',
  'TRP': 'Low Latitudes',
  'SAN': 'Southern Andes',
  'NZL': 'New Zealand',
  'ANT': 'Antarctic & Subantarctic',
}


# ---- DOI release ----

def format_global_grid(
    global_grid_netcdf_dir: Path,
    mass_loss_dir: Path,
    version: str
) -> None:
    paths = sorted(global_grid_netcdf_dir.glob('*.nc'))
    results = []
    for path in paths:
        print(path)
        results.append(xarray.open_dataset(path, engine='netcdf4'))
    # Merge into single dataset
    ds = xarray.concat(results, dim='time')
    # Fix units due to error in original files (mm w.e. instead of m w.e.)
    variables = [
        'glacier_mass_change_gt', 'uncertainty_gt',
        'glacier_mass_change_mwe', 'uncertainty_mwe'
    ]
    for variable in variables:
        ds[variable] *= 1e-3
    # Rename variables
    ds = ds.rename(
        {
            'glacier_mass_change_gt': 'gt',
            'uncertainty_gt': 'gt_sigma',
            'glacier_mass_change_mwe': 'mwe',
            'uncertainty_mwe': 'mwe_sigma',
            'glacier_area_km2': 'area_km2'
        }
    )
    # Make copy for later comparison
    ds_original = ds.copy()
    # Convert data variables to float32
    for var in ds.data_vars:
        if ds[var].dtype == 'float64':
            ds[var] = ds[var].astype('float32')
    # Convert dimensions to float32
    for dim in ds.dims:
        if ds[dim].dtype == 'float64':
            ds[dim] = ds[dim].astype('float32')
    # Check that values did not change significantly
    for var in ds.data_vars:
        assert abs(ds[var].fillna(0) - ds_original[var].fillna(0)).max() < 1e-3
    # Extract observational sample
    stats_path = list(mass_loss_dir.glob('Cum_DM_Gt_per_region_PoR_*.csv'))[0]
    stats = pd.read_csv(stats_path)
    sample = round(stats.set_index('region_code').loc['GLOBAL']['percentage_area_obs'])
    # Update attributes
    begin_year = ds['time'].dt.year.min().item()
    end_year = ds['time'].dt.year.max().item()
    ds.attrs.update({
        'created_by': 'World Glacier Monitoring Service (WGMS) - wgms@geo.uzh.ch',
        'data_version': f'https://doi.org/10.5904/wgms-amce-{version}',
        'references': f'Fluctuations of Glaciers (FoG) database https://doi.org/10.5904/wgms-fog-{version}',
        'citation': f"WGMS ({version[:4]}): Annual mass-change estimates for the world's glaciers. Individual glacier time series and gridded data products. Digital media. https://doi.org/10.5904/wgms-amce-{version}",
        'publication': "Dussaillant, I., Hugonnet, R., Huss, M., Berthier, E., Bannwart, J., Paul, F., and Zemp, M. (2025): Annual mass change of the world's glaciers from 1976 to 2024 by temporal downscaling of satellite data with in-situ observations. Earth System Science Data, https://doi.org/10.5194/essd-17-1977-2025",
        'dataset_description': f'Horizontal resolution: 0.5° (latitude - longitude), GCS_WGS_1984 | Temporal resolution: Annual, hydrological year | Temporal coverage: Hydrological years from {begin_year} to {end_year} | Observational sample: {sample}% of world glaciers with valid observations | Spatial interpolation method: Kriging'
    })
    del ds.attrs['institution']
    del ds.attrs['project']
    # Write to file
    new_path = PUBLISH_DIR / 'global_grid.nc4'
    if new_path.exists():
        new_path.unlink()
    ds.to_netcdf(
        path=new_path,
        format='NETCDF4',
        mode='w',
        encoding={
            var: {'zlib': True, 'complevel': 9} for var in ds.data_vars if var != 'spatial_ref'
        }
    )


def format_glacier(
    regional_balance_essd_dir: Path
) -> None:
    paths = regional_balance_essd_dir.glob('*.csv')
    file_renames = {
        'gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs': 'mwe_sigma_anom',
        'gla_mean-cal-mass-change_DH-ERROR_obs_unobs': 'mwe_sigma_dh',
        'gla_mean-cal-mass-change_RHO-ERROR_obs_unobs': 'mwe_sigma_rho',
        'gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs': 'mwe_sigma',
        'gla_MEAN-CAL-mass-change-series_obs_unobs': 'mwe'
    }
    output_dir = PUBLISH_DIR / 'glacier'
    output_dir.mkdir(parents=True, exist_ok=True)
    column_renames = {
        'RGIId': 'outline_id',
        'GLIMS_ID': 'outline_id',
        'WGMS_ID': 'glacier_id',
        'CenLat': 'latitude',
        'CenLon': 'longitude',
        'Area': 'area_km2',
    }
    nulls = ['no_WGMS_ID', 'no_obs', 'N/A']
    for path in paths:
        print(path)
        df = pd.read_csv(path)
        # Drop null values
        df.replace(nulls, pd.NA, inplace=True)
        # Convert mm to m w.e. and reduce to 3 decimal places
        years = [col for col in df.columns if re.match(r'^\d{4}$', col)]
        df[years] = df[years].divide(1e3).round(3)
        # Convert WGMS_ID to integer
        df['WGMS_ID'] = df['WGMS_ID'].astype('Float64').astype('Int64')
        # Rename columns
        df.rename(columns=column_renames, inplace=True)
        # Write to file (glacier_metadata)
        # TODO: Update with full metadata once available from output
        region, suffix = path.stem.split('_', maxsplit=1)
        main_columns = ['outline_id', 'glacier_id', 'latitude', 'longitude', 'area_km2']
        metadata_path = output_dir / f'{region}_metadata.csv'
        if not metadata_path.exists():
            df[main_columns].to_csv(metadata_path, index=False)
        # Write to file
        new_path = output_dir / f'{region}_{file_renames[suffix]}.csv'
        df[['outline_id', *years]].to_csv(new_path, index=False)


def format_regional(
    mass_loss_dir: Path,
) -> None:
    output_dir = PUBLISH_DIR / 'region'
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in mass_loss_dir.joinpath('regional_mass_loss_series').glob('*.csv'):
        print(path)
        df = pd.read_csv(path)
        # Drop empty rows
        df.dropna(how='any', inplace=True)
        # Rename columns
        column_renames = {
            'YEAR': 'year',
            'Aw_mwe': 'mwe',
            'sig_tot_mwe': 'mwe_sigma',
            'area_tot_km2': 'area_km2',
            'DM_Gt': 'gt',
            'sig_tot_DM': 'gt_sigma'
        }
        df.rename(columns=column_renames, inplace=True)
        # Reduce to 3 decimal places
        columns = ['mwe', 'mwe_sigma', 'gt', 'gt_sigma']
        df[columns] = df[columns].round(3)
        # Write to file
        region = path.stem.split('_')[-2]
        new_path = output_dir / f'{region}.csv'
        df[['year', 'area_km2', 'mwe', 'mwe_sigma', 'gt', 'gt_sigma']].to_csv(new_path, index=False)


def format_global(
    mass_loss_dir: Path,
) -> None:
    path = list(mass_loss_dir.glob('Global_DM_series_year_*.csv'))[0]
    df = pd.read_csv(path)
    # Rename columns
    column_renames = {
        'YEAR': 'year',
        'B [m w.e.]': 'mwe',
        'sigma B [m w.e.]': 'mwe_sigma',
        'B_cum [m w.e.]': 'mwe_cumsum',
        'DM [Gt]': 'gt',
        'sigma_DM [Gt]': 'gt_sigma',
        'DM_cum [Gt]': 'gt_cumsum',
        'sigma_DM_cum [Gt]': 'gt_cumsum_sigma',
        'SLE [mm]': 'mmsle',
        'sigma_SLE [mm]': 'mmsle_sigma',
        'SLE_cum [Gt]': 'mmsle_cumsum',
        'sigma_SLE_cum [mm]': 'mmsle_cumsum_sigma'
    }
    df.rename(columns=column_renames, inplace=True)
    # Reduce to 3 decimal places
    columns = list(column_renames.values())[1:]
    df[columns] = df[columns].round(3)
    # Add area column
    paths = mass_loss_dir.joinpath('regional_mass_loss_series').glob('*.csv')
    areas = []
    for path in paths:
        areas.append(pd.read_csv(path).set_index('YEAR')['area_tot_km2'])
    df['area_km2'] = pd.concat(areas, axis=1).sum(axis=1).loc[df['year']].round(2).values
    # Write to file
    new_path = PUBLISH_DIR / 'global.csv'
    df[['year', 'area_km2', *columns]].to_csv(new_path, index=False)


def format_readme(
    version: str
) -> None:
    template = Path('templates/readme.md.jinja').read_text()
    text = jinja2.Template(template).render(version=version)
    new_path = PUBLISH_DIR / 'README.md'
    new_path.write_text(text)


def format_website(
    version: str
) -> None:
    global_df = pd.read_csv(PUBLISH_DIR / 'global.csv')
    region_dfs = [pd.read_csv(path) for path in (PUBLISH_DIR / 'region').glob('*.csv')]
    zip_path = PUBLISH_DIR / f'wgms-amce-{version}.zip'
    kwargs = {
        'begin_year': global_df['year'].min(),
        'end_year': global_df['year'].max(),
        'begin_year_min': min(df['year'].min() for df in region_dfs),
        'begin_year_max': max(df['year'].min() for df in region_dfs),
        'megabytes': round(zip_path.stat().st_size / 1e6)
    }
    template = Path('templates/website.html.jinja').read_text()
    text = jinja2.Template(template).render(version=version, **kwargs)
    new_path = PUBLISH_DIR / 'website.html'
    new_path.write_text(text)


def build_doi_release(
    mass_loss_dir: Path,
    regional_balance_essd_dir: Path,
    global_grid_netcdf_dir: Path,
    version: str
) -> None:
    # Prepare files
    format_readme(version=version)
    format_global(mass_loss_dir=mass_loss_dir)
    format_global_grid(
        global_grid_netcdf_dir=global_grid_netcdf_dir,
        mass_loss_dir=mass_loss_dir,
        version=version
    )
    format_regional(mass_loss_dir=mass_loss_dir)
    format_glacier(regional_balance_essd_dir=regional_balance_essd_dir)
    # Build zip archive (wgms-amce-{version}.zip)
    # README.md | global.csv | global_grid.nc4 | region/*.csv | glacier/*.csv
    zip_path = PUBLISH_DIR / f'wgms-amce-{version}.zip'
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zip:
        zip.write(PUBLISH_DIR / 'README.md', arcname='README.md')
        zip.write(PUBLISH_DIR / 'global.csv', arcname='global.csv')
        zip.write(PUBLISH_DIR / 'global_grid.nc4', arcname='global_grid.nc4')
        for path in (PUBLISH_DIR / 'region').glob('*.csv'):
            zip.write(path, arcname=f'region/{path.name}')
        for path in (PUBLISH_DIR / 'glacier').glob('*.csv'):
            zip.write(path, arcname=f'glacier/{path.name}')
    format_website(version=version)


# --- Website figures ----

# Configure matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'

# Configure plotnine
p9.theme_set(p9.theme_538(base_family='Arial'))
p9.theme_update(
  panel_background=p9.element_rect(fill='white'),
  plot_background=p9.element_rect(fill='white', color='none'),
  plot_title=p9.element_text(size=14, family='Arial', ha='left'),
  plot_subtitle=p9.element_text(
    size=10, fontstyle='italic', family='Arial', ha='left',
    linespacing=1.25
  ),
  axis_title=p9.element_text(size=12, family='Arial'),
  axis_text=p9.element_text(size=12, family='Arial', linespacing=1.25),
  axis_text_x=p9.element_text(size=12, family='Arial'),
  axis_text_y=p9.element_text(size=12, family='Arial'),
  axis_line=p9.element_blank(),
  panel_grid_major=p9.element_line(color='#dddddd', size=0.5),
  panel_grid_minor=p9.element_blank(),
  legend_key=p9.element_rect(fill='white', color='none'),
  strip_background=p9.element_rect(fill='#dddddd', color='none'),
  # NOTE: ha is ignored (https://github.com/has2k1/plotnine/issues/867)
  strip_text=p9.element_text(size=12, family='Arial', ha='left'),
  strip_text_x=p9.element_text(size=12, family='Arial', ha='left'),
  strip_text_y=p9.element_text(size=12, family='Arial', ha='left'),
  svg_usefonts=True
)


def plot_global_annual_mass_change_bars() -> None:
    # https://wgms.ch/sea-level-rise/
    # https://wgms.ch/data/faq/_Fig2_global_glacier_mass_changes_bars.svg
    temp = pd.read_csv(PUBLISH_DIR / 'global.csv')
    df = pd.DataFrame({
        'year': temp['year'],
        'y': temp['gt'],
        'ymin': temp['gt'] - 2 * temp['gt_sigma'],
        'ymax': temp['gt'] + 2 * temp['gt_sigma'],
        'sign': temp['gt'].apply(np.sign).eq(1)
    })
    plot = (
        p9.ggplot(df, p9.aes(x='year', y='y', ymin='ymin', ymax='ymax', color='sign', fill='sign')) +
        p9.geom_bar(stat='identity', color='none', width=1) +
        p9.scale_fill_manual(values=['tab:red', 'tab:blue'], guide=None) +
        p9.scale_color_manual(values=['tab:red', 'tab:blue'], guide=None) +
        p9.scale_y_continuous(
            breaks=(-724, -362, 0, 362),
            labels=(
                '−724 Gt\n2 mm'.replace(' ', u'\u2009'),
                '−362 Gt\n1 mm'.replace(' ', u'\u2009'),
                '0',
                '362 Gt\n−1 mm'.replace(' ', u'\u2009')
            ),
            limits=(-724, 362)
        ) +
        p9.scale_x_continuous(
            breaks=range(1980, df['year'].max() + 1, 10)
        ) +
        p9.labs(
            x='Year',
            y='Mass change (Gt) · Sea level rise (mm)',
            title='Global annual glacier mass change',
            subtitle='About −362 Gt raises global sea level by 1 mm.',
        )
    )
    plot.save(
        PUBLISH_DIR / '_Fig2_global_glacier_mass_changes_bars.svg',
        dpi=250, width=6, height=4, units='in'
    )


def plot_global_cumulative_annual_mass_change() -> None:
    # https://wgms.ch/sea-level-rise/
    # https://wgms.ch/data/faq/_Fig2_global_glacier_cumulative_mass_changes.svg
    temp = pd.read_csv(PUBLISH_DIR / 'global.csv')
    df = pd.DataFrame({
        'year': temp['year'],
        'y': temp['gt_cumsum'],
        'ymin': temp['gt_cumsum'] - 2 * temp['gt_cumsum_sigma'],
        'ymax': temp['gt_cumsum'] + 2 * temp['gt_cumsum_sigma']
    })
    plot = (
        p9.ggplot(df, p9.aes(x='year', y='y', ymin='ymin', ymax='ymax')) +
        p9.geom_hline(yintercept=0, color='#bbb', linetype='solid', size=0.5) +
        p9.geom_ribbon(alpha=0.3, color='none', fill='tab:blue') +
        p9.geom_line(color='tab:blue') +
        p9.scale_y_continuous(
            breaks=(-10860, -7240, -3620, 0, 3620),
            labels=(
                '−10860 Gt\n30 mm'.replace(' ', u'\u2009'),
                '−7240 Gt\n20 mm'.replace(' ', u'\u2009'),
                '−3620 Gt\n10 mm'.replace(' ', u'\u2009'),
                '0',
                '3620 Gt\n−10 mm'.replace(' ', u'\u2009')
            ),
            limits=(-10860, 3620)
        ) +
        p9.scale_x_continuous(
            breaks=range(1980, df['year'].max() + 1, 10)
        ) +
        p9.labs(
            x='Year',
            y='Mass change (Gt) · Sea level rise (mm)',
            title='Global cumulative glacier mass change',
            subtitle='Relative to 1975. About −362 Gt raises global sea level by 1 mm.'
        )
    )
    plot.save(
        PUBLISH_DIR / '_Fig2_global_glacier_cumulative_mass_changes.svg',
        dpi=250, width=6, height=4, units='in'
    )


def plot_regional_annual_mass_change(
    n_years: int = 5,
    reference_period: tuple[int, int] = (2001, 2016),
) -> None:
    # https://wgms.ch/sea-level-rise
    # http://wgms.ch/data/faq/FoG-20240208_adhoc_2017-23.svg
    # Read regions
    df = pd.concat((
        pd.read_csv(path).assign(region=path.stem)
        for path in PUBLISH_DIR.glob('region/*.csv')
    ), ignore_index=True)
    # Append global
    world = pd.read_csv(PUBLISH_DIR / 'global.csv').assign(region='Global').reindex(
        columns=df.columns
    )
    df = pd.concat((df, world), ignore_index=True)
    # Merge SA1 and SA2 as SAN
    sa1 = df[df['region'] == 'SA1'].set_index('year')
    sa2 = df[df['region'] == 'SA2'].set_index('year')
    san = pd.DataFrame({
        'region': 'SAN',
        'year': sa1.index,
        'area_km2': sa1['area_km2'] + sa2['area_km2'],
        'mwe': (sa1['mwe'] * sa1['area_km2'] + sa2['mwe'] * sa2['area_km2']) / (sa1['area_km2'] + sa2['area_km2']),
        'mwe_sigma': np.sqrt(
            ((sa1['mwe_sigma'] * sa1['area_km2']) ** 2 + (sa2['mwe_sigma'] * sa2['area_km2']) ** 2) /
            (sa1['area_km2'] + sa2['area_km2']) ** 2
        ),
        'gt': sa1['gt'] + sa2['gt'],
        'gt_sigma': np.sqrt(sa1['gt_sigma'] ** 2 + sa2['gt_sigma'] ** 2)
    })
    df = pd.concat((df[~df['region'].isin(['SA1', 'SA2'])], san), ignore_index=True)
    # Replace region_id with region_name and place Global at the end
    df['region_name'] = pd.Categorical(
        df['region'].apply(lambda x: REGION_NAMES.get(x, x)).values,
        categories=list(REGION_NAMES.values()) + ['Global']
    )
    # Calculate reference period specific mass change
    mask = df['year'].between(*reference_period, inclusive='both')
    reference_mwe = df[mask].groupby('region_name', observed=True)['mwe'].mean()
    # Compute mwe relative to reference period
    df['mwe_reference'] = reference_mwe.loc[df['region_name']].values
    df['mwe_relative'] = df['mwe'] - df['mwe_reference']
    df['mwe_relative_sign'] = df['mwe_relative'].apply(np.sign).eq(1)
    df['mwe_sign'] = df['mwe'].apply(np.sign).eq(1)
    df['gt_label'] = df['gt'].round(0).astype('Int64').apply(
        lambda x: pd.NA if pd.isna(x) else f'{int(x)}'.replace('-', '−')
    )
    # Add ' Gt' to each regions' last dm_label
    end_year = df['year'].max()
    mask = df['year'].eq(end_year)
    df.loc[mask, 'gt_label'] = df.loc[mask, 'gt_label']
    # Filter by year
    mask = df['year'].ge(end_year - n_years + 1)
    df = df[mask]
    # Plot
    plot = (
        p9.ggplot(df, p9.aes(x='year', y='mwe')) +
        p9.geom_rect(
            mapping=p9.aes(xmin='year - 0.4', xmax='year + 0.4', ymin='mwe_reference', ymax='mwe', fill='mwe_relative_sign'),
            color='none',
            alpha=0.3
        ) +
        p9.geom_rect(
            mapping=p9.aes(xmin='year - 0.4', xmax='year + 0.4', ymin=0, ymax='mwe', color='mwe_sign'),
            fill='none'
        ) +
        p9.scale_fill_manual(values=['tab:red', 'tab:blue'], guide=None) +
        p9.scale_color_manual(values=['tab:red', 'tab:blue'], guide=None) +
        p9.geom_hline(yintercept=0, color='#333333', linetype='solid', size=0.75) +
        p9.geom_hline(
            data=reference_mwe.reset_index(),
            mapping=p9.aes(yintercept='mwe'),
            color='orange',
            linetype='solid',
            size=0.75
        ) +
        p9.geom_linerange(
            mapping=p9.aes(ymin='mwe - 2 * mwe_sigma', ymax='mwe + 2 * mwe_sigma', color='mwe_sign')
        ) +
        p9.geom_text(
            mapping=p9.aes(label='gt_label'),
            y=1.25,
            color='#333333',
            va='center_baseline',
            size='11',
        ) +
        p9.annotate(
            geom='text',
            x=end_year + 0.65,
            y=1.25,
            label='Gt',
            color='#333333',
            size='9',
            va='center_baseline'
        ) +
        p9.facet_wrap('region_name', ncol=4) +
        p9.scale_x_continuous(
            breaks=range(end_year - n_years + 1, end_year + 1, 1)
        ) +
        p9.scale_y_continuous(
            breaks=[-3, -2, -1, 0, 1],
            labels=lambda x: [f'{x}'.replace('-', '−') for x in x]
        ) +
        p9.labs(
            title='Regional annual glacier mass change ({}–{})'.format(end_year - n_years + 1, end_year),
            subtitle=(
                'Shaded bars are the deviation from the {}–{} mean (orange line).'.format(*reference_period) +
                '\nNumbers at the top of each panel are the annual mass change in gigatonnes (Gt).'
            ),
            x='Year',
            y='Specific mass change (m w.e.)'
        ) +
        p9.theme(
            panel_border=p9.element_rect(fill='none', color='#dddddd', size=0.5),
            axis_ticks_major=p9.element_line(color='#dddddd', size=0.5),
            axis_ticks_length=5,
            axis_ticks_pad_major=3,
            axis_text_x=p9.element_text(size=11),
        )
    )
    plot.save(
        PUBLISH_DIR / 'FoG-20240208_adhoc_2017-23.svg',
        dpi=250, width=10, height=12, units='in', limitsize=False
    )


def plot_global_annual_mass_change_stripes(height: float = 5) -> None:
    df = pd.read_csv(PUBLISH_DIR / 'global.csv').sort_values('year')
    s = df['gt']
    # Configure the figure
    width = 10
    figure, axis = matplotlib.pyplot.subplots(1, 1, figsize=(width, height * width / s.size))
    axis.set_xlim(0, s.size)
    axis.set_ylim(0, height)
    axis.set_axis_off()
    axis.set_position([0, 0, 1, 1])
    figure.patch.set_visible(False)
    axis.patch.set_visible(False)
    # Define a red (negative) to blue (positive) colorscale centered around 0
    colormap = matplotlib.pyplot.get_cmap('RdBu')
    vmin = s.min()
    normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=abs(vmin))
    # Create a colored rectangle for each year
    for i, value in enumerate(s):
        axis.add_patch(matplotlib.patches.Rectangle(
            xy=(i, 0), width=1, height=height, facecolor=colormap(normalizer(value))
        ))
    # Save figure
    matplotlib.pyplot.savefig(
        PUBLISH_DIR / 'stripes.svg',
        bbox_inches='tight', transparent=True, pad_inches=0
    )
    matplotlib.pyplot.close()


def plot_map() -> None:
    ds = xarray.open_dataset(PUBLISH_DIR / 'global_grid.nc4')
    # Create a map with a plate carrée projection
    figure = matplotlib.pyplot.figure(figsize=(12, 6))
    axis = matplotlib.pyplot.axes(projection=cartopy.crs.PlateCarree())
    # Add gray continents against a white background
    axis.set_facecolor('white')
    axis.add_feature(
        cartopy.feature.NaturalEarthFeature('physical', 'land', '50m'),
        facecolor='lightgray',
        zorder=0
    )
    # Define range and ticks of color scale
    ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    # Plot mass loss in red and gain in blue
    im = matplotlib.pyplot.pcolormesh(
        ds['lon'],
        ds['lat'],
        ds['mwe'][-1],
        cmap='RdBu',
        norm=matplotlib.colors.Normalize(vmin=ticks[0], vmax=ticks[-1]),
        zorder=1
    )
    # # Zoom to Greenland, Europe, and Caucasus
    # upper left: 83.91315240, -63.60397800
    # lower right: 25.88312402, 59.19444938
    # axis.set_extent([-63.60397800, 59.19444938, 25.88312402, 83.91315240])
    # Add a colorbar with custom ticks
    matplotlib.pyplot.colorbar(
        im,
        fraction=0.025,
        pad=0.05,
        label='Mass change (m w.e.)',
        ticks=ticks
    )
    # Show plot
    figure.tight_layout(pad=1)
    matplotlib.pyplot.savefig(PUBLISH_DIR / 'map.png', dpi=200)
    matplotlib.pyplot.close('all')


def build_website_figures() -> None:
    plot_global_annual_mass_change_bars()
    plot_global_cumulative_annual_mass_change()
    plot_regional_annual_mass_change()


def extract_mass_balance_anomaly_glacier_ids(lookup_anomaly_dir: Path) -> set[int]:
    ids = []
    for path in lookup_anomaly_dir.glob('*_all_SEL_gla_anomalies.csv'):
        df = pd.read_csv(path)
        ids += [int(column) for column in df.columns[1:]]
    return set(ids)


EXCLUDED_MASS_BALANCE_GLACIERS: list[int] = [
    # --- ASN: Asia North
    # Hamagury Yuki
    897,
    # --- ASC: Asia Central
    # Urumqi No. 1 east branch
    1511,
    # Urumqi No. 1 west branch
    1512,
    # --- TRP: Tropics
    # Yanamarey
    226,
    # --- SAN-01: Southern Andes Patagonia
    # "All except Martial Este"
    # Martial
    917,
    # De Los Tres
    1675,
    # --- SAN-02: Southern Andes Central
    # "All except Echaurren Norte"
    # Conconta Norte
    3902,
    # Brown Superior
    3903,
    # Los Amarillos
    3904,
    # Amarillo
    3905,
    # Mocho Choshuenco (southeast)
    3972,
    # --- ANT: Antarctica and Subantarctic
    # "Dry valley glaciers"
    878,
    3973
]


def build_seasonal_regional_mass_change(
    mass_balance_file: Path,
    urumqi_missing_years_file: Path,
    lookup_anomaly_dir: Path,
    glacier_series_file: Path
) -> None:
    # ---- Prepare seasonal mass balances
    df = pd.read_csv(mass_balance_file)
    urumqi_missing_years = (
        pd.read_csv(urumqi_missing_years_file)
        .rename(columns={'853': 'ANNUAL_BALANCE'})
        .assign(WGMS_ID=853, GLACIER_REGION_CODE='ASC')
    )
    df = pd.concat((df, urumqi_missing_years), ignore_index=True)
    # Keep all seasonal amplitudes from non-excluded glaciers
    mask = (
        df['ANNUAL_BALANCE'].notnull() &
        df['WINTER_BALANCE'].notnull() &
        ~df['WGMS_ID'].isin(EXCLUDED_MASS_BALANCE_GLACIERS)
    )
    df = df[mask]
    # --- Calculate annual regional mass balance amplitudes (m w.e.)
    df['amplitude'] = (df['WINTER_BALANCE'] - df['SUMMER_BALANCE']).abs() / 2
    amplitudes = df.groupby(['GLACIER_REGION_CODE', 'YEAR'])['amplitude'].mean() * 1e-3
    # ---- Compute calibrated seasonal annual mass change (Gt)
    SAN_SUBREGIONS = ['SA1', 'SA2']
    region_dfs = []
    for path in PUBLISH_DIR.glob('region/*.csv'):
        region_df = pd.read_csv(path).set_index('year')
        region = path.stem
        amplitude_region = region
        if region in SAN_SUBREGIONS:
            amplitude_region = 'SAN'
        region_amplitude = amplitudes.loc[amplitude_region]
        winter = region_df['mwe'] / 2 + region_amplitude
        summer = region_df['mwe'] / 2 - region_amplitude
        winter_gt = winter * region_df['area_km2'] * 1e-3
        summer_gt = summer * region_df['area_km2'] * 1e-3
        region_dfs.append(pd.DataFrame({
            'region_id': amplitude_region,
            'year': winter.index,
            'winter_gt': winter_gt,
            'summer_gt': summer_gt,
            'gt': region_df['gt']
        }))
    # Sum subregions
    parent_path = PUBLISH_DIR / 'seasonal'
    parent_path.mkdir(parents=True, exist_ok=True)
    (
        pd.concat(region_dfs, ignore_index=True)
        .dropna(subset='gt')
        .convert_dtypes()
        .groupby(['region_id', 'year'], dropna=False)
        .sum(min_count=1)
        .round(3)
        .to_csv(parent_path / 'region.csv', index=True)
    )
    # Compile glacier id | latitude | longitude | annual (boolean) | seasonal (boolean)
    glacier_series = pd.read_csv(
        glacier_series_file, usecols=['WGMS_ID', 'LATITUDE', 'LONGITUDE']
    ).set_index('WGMS_ID')
    seasonal_glacier_ids = set(df['WGMS_ID'])
    annual_glacier_ids = extract_mass_balance_anomaly_glacier_ids(
        lookup_anomaly_dir=lookup_anomaly_dir
    )
    glacier_ids = sorted(annual_glacier_ids | seasonal_glacier_ids)
    glaciers = pd.DataFrame({'glacier_id': glacier_ids}).set_index('glacier_id')
    glaciers[['latitude', 'longitude']] = glacier_series[
        ['LATITUDE', 'LONGITUDE']
    ].loc[glacier_ids].round(6)
    glaciers['annual'] = glaciers.index.isin(annual_glacier_ids)
    glaciers['seasonal'] = glaciers.index.isin(seasonal_glacier_ids)
    glaciers.reset_index().to_csv(parent_path / 'glacier.csv', index=False)


HYDROLOGICAL_END_DATE: dict[str, tuple[str, int]] = {
    'ALA': ('10-01', 0),
    'WNA': ('10-01', 0),
    'ACN': ('10-01', 0),
    'ACS': ('10-01', 0),
    'GRL': ('10-01', 0),
    'ISL': ('10-01', 0),
    'SJM': ('10-01', 0),
    'SCA': ('10-01', 0),
    'RUA': ('10-01', 0),
    'ASN': ('10-01', 0),
    'CEU': ('10-01', 0),
    'CAU': ('10-01', 0),
    'ASC': ('10-01', 0),
    'ASW': ('10-01', 0),
    'ASE': ('10-01', 0),
    'TRP': ('01-01', 1),
    'NZL': ('04-01', 0),
    'ANT': ('04-01', 0),
    'SA1': ('04-01', 0),
    'SA2': ('04-01', 0)
}


def build_glambie_submission(
    mass_loss_dir: Path,
    lookup_anomaly_dir: Path,
    rgi_code: dict[str, str],
    mass_balance_area_file: Path,
    regional_area_change_rate_file: Path,
    mass_balance_file: Path,
    begin_year: int,
    year_ini: int,
    year_fin: int,
    elevation_change_file: Path,
    investigators_to_drop: list[str],
    min_length_geo: float,
    long_norm_anomaly_dir: Path,
    glacier_series_file: Path,
    rgi_area_file: Path,
    glambie_begin_year: int = 1976
):
    """
    Build GlaMBIE regional mass change submission.

    See https://glambie.org/data-submission-guide/.
    """
    # Read regional mass change data
    output_dir = PUBLISH_DIR / 'glambie'
    output_dir.mkdir(parents=True, exist_ok=True)
    percentage_area_obs = pd.read_csv(
        mass_loss_dir / 'Cum_DM_Gt_per_region_PoR_1976_2025.csv'
    ).set_index('region_code')['percentage_area_obs'].to_dict()
    region_dfs = []
    san_dfs = []
    for path in mass_loss_dir.joinpath('regional_mass_loss_series').glob('*.csv'):
        print(path)
        df = pd.read_csv(path)
        # Drop empty rows
        df.dropna(how='any', inplace=True)
        # Drop earlier years
        df = df[df['YEAR'] >= glambie_begin_year]
        region = path.stem.split('_')[-2]
        mm_dd, offset = HYDROLOGICAL_END_DATE[region]
        region_df = pd.DataFrame({
            # region_id: GTN-G region integer ID
            'region_id': int(path.stem.split('_')[2]),
            # start_date: dd/mm/yyyy
            'start_date': pd.to_datetime(df['YEAR'].add(offset - 1).astype(str) + '-' + mm_dd).dt.strftime('%d/%m/%Y'),
            # end_date: dd/mm/yyyy
            'end_date': pd.to_datetime(df['YEAR'].add(offset).astype(str) + '-' + mm_dd).dt.strftime('%d/%m/%Y'),
            'glacier_change_observed': df['Aw_mwe'],
            'glacier_change_uncertainty': df['sig_tot_mwe'],
            'unit': 'mwe',
            # glacier_area_reference_start: km2
            'glacier_area_reference_start': df['area_tot_km2'],
            # glacier_area_reference_end: km2
            'glacier_area_reference_end': df['area_tot_km2'],
            # observational_coverage_percentage: %
            'observational_coverage_percentage': percentage_area_obs[region],
            'remarks': None
        })
        if region in ['SA1', 'SA2']:
            san_dfs.append(region_df)
        else:
            region_dfs.append(region_df)
    df = pd.concat(region_dfs, ignore_index=True)
    # Merge SA1 and SA2 as SAN
    san_df = pd.concat(san_dfs, ignore_index=True)
    san_df = san_df.groupby('start_date', as_index=False).agg({
        'region_id': 'first',
        'end_date': 'first',
        # area-weighted average
        'glacier_change_observed': (
            lambda x:
                (x * san_df.loc[x.index, 'glacier_area_reference_start']).sum() /
                san_df.loc[x.index, 'glacier_area_reference_start'].sum()
        ),
        # area-weighted average propagated uncertainty
        'glacier_change_uncertainty': (
            lambda x:
                np.sqrt((
                    san_df.loc[x.index, 'glacier_change_uncertainty'] *
                    san_df.loc[x.index, 'glacier_area_reference_start']
                ) ** 2).sum() /
            san_df.loc[x.index, 'glacier_area_reference_start'].sum()
        ),
        'unit': 'first',
        'glacier_area_reference_start': 'sum',
        'glacier_area_reference_end': 'sum',
        # area-weighted average
        'observational_coverage_percentage': (
            lambda x:
                (x * san_df.loc[x.index, 'glacier_area_reference_start']).sum() /
                san_df.loc[x.index, 'glacier_area_reference_start'].sum()
        ),
        'remarks': 'first'
    })
    df = pd.concat((df, san_df), ignore_index=True).sort_values(['region_id', 'start_date'])
    for column in ['glacier_change_observed', 'glacier_change_uncertainty']:
        df[column] = df[column].round(3)
    for colum in ['glacier_area_reference_start', 'glacier_area_reference_end', 'observational_coverage_percentage']:
        df[colum] = df[colum].round(0).astype('Int64')
    df.to_csv(output_dir / 'submission-combined.csv', index=False)

    # ---- Table of mass balance glacier ids by region and their total area ----

    # Load glacier ids by region and year
    dfs = []
    for path in lookup_anomaly_dir.glob('*_all_reg_gla_anomalies.csv'):
        region = path.stem.split('_')[0]
        series = pd.read_csv(path).set_index('YEAR')
        for glacier_id in series.columns:
            mask = series[glacier_id].notnull()
            df = pd.DataFrame({
                'region_code': region,
                'year': series.index[mask],
                'glacier_id': int(glacier_id),
            })
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Add normalized glaciers
    # TODO: Update if start year is changed in the future
    temp = pd.DataFrame(
        columns=['region_code', 'glacier_id'],
        data=[
            ('GRL', 16),
            ('GRL', 39),
            ('ISL', 302),
            ('ISL', 317),
            ('ISL', 2296),
            ('TRP', 1344),
            ('NZL', 1344),
            ('ANT', 1344)
        ]
    )
    df = pd.concat((df, temp), ignore_index=True).convert_dtypes()

    # Add region id
    df['region_id'] = df['region_code'].map(rgi_code).astype('Int64')

    # Merge SA1 and SA2 as SAN
    region_glacier_ids_sa12 = df.copy()
    is_san = df['region_code'].isin(['SA1', 'SA2'])
    df.loc[is_san, 'region_code'] = 'SAN'

    # Mark which glaciers are local to the region
    df['local'] = (
        pd.read_csv(mass_balance_file)[['GLACIER_REGION_CODE', 'WGMS_ID']]
        .drop_duplicates()
        .set_index('WGMS_ID')['GLACIER_REGION_CODE']
    ).loc[df['glacier_id']].eq(df['region_code'].values).values

    # Save for later
    region_glacier_ids = df.copy()

    # Tabulate region id, region code, glacier ids
    table = df.groupby('region_code', as_index=False).agg(
        region_id=pd.NamedAgg(column='region_id', aggfunc='first'),
        glacier_ids=pd.NamedAgg(column='glacier_id', aggfunc=lambda x: '|'.join(x.drop_duplicates().sort_values().astype(str)))
    ).sort_values('region_id')

    # Add glacier area
    areas = pd.read_csv(mass_balance_area_file).set_index('glacier_id')[['area_m2']]
    temp = df[df['local']][['region_id', 'glacier_id']].drop_duplicates().copy()
    temp['area_m2'] = areas.loc[temp['glacier_id'], 'area_m2'].values
    table['area_local_km2'] = (
        temp.groupby('region_id')['area_m2']
        .sum()
        .div(1e6)
        .round(0)
        .astype('Int64')
        .reindex(table['region_id'])
        .fillna(0)
        .values
    )
    table[['region_id', 'region_code', 'glacier_ids', 'area_local_km2']].to_csv(output_dir / 'glaciers.csv', index=False)

    # ---- Table of extra or excluded mass balance glacier ids by region ----
    # See manual table with descriptions in glaciers-special.csv

    # Filter mass balance data
    df = pd.read_csv(mass_balance_file)
    mask = df['YEAR'].ge(begin_year) & df['ANNUAL_BALANCE'].notnull()
    df = df[mask]
    min_ref_year = df['GLACIER_REGION_CODE'].map({'ASN': 4}).fillna(8)
    mask = df.groupby('WGMS_ID')['YEAR'].transform(lambda x: x.between(year_ini, year_fin, inclusive='both').sum()) >= min_ref_year
    df = df[mask]

    # Compile glacier ids by region
    df = (
        df[['GLACIER_REGION_CODE', 'WGMS_ID']]
        .drop_duplicates()
        .rename(columns={'GLACIER_REGION_CODE': 'region_code', 'WGMS_ID': 'glacier_id'})
    )

    # Compare to glacier_ids above
    results = []
    a = region_glacier_ids.set_index('region_code')
    b = df.set_index('region_code')
    for region in a.sort_values('region_id').index.unique():
        ai = set(a.loc[[region], 'glacier_id'])
        bi = set()
        if region in b.index:
            bi = set(b.loc[[region], 'glacier_id'])
        results.append({
            'region_id': a.loc[region, 'region_id'].iloc[0],
            'region_code': region,
            'added': '|'.join(map(str, sorted(ai - bi))),
            'removed': '|'.join(map(str, sorted(bi - ai)))
        })
    pd.DataFrame(results).to_csv(output_dir / 'glaciers-special-auto.csv', index=False)

    # ---- Look up tables for reference and author lists ----

    # geodetic
    df = pd.read_csv(elevation_change_file)
    mask = (
        df['ELEVATION_CHANGE'].notnull() &
        df['SURVEY_DATE'].notnull() &
        df['REFERENCE_DATE'].notnull() &
        df['GLACIER_REGION_CODE'].notnull() &
        (
            (df['GLACIER_REGION_CODE'].eq('CAU') & df['GLIMS_ID'].notnull()) |
            (df['GLACIER_REGION_CODE'].ne('CAU') & df['RGI60_ID'].notnull())
        ) &
        ~df['INVESTIGATOR'].isin(investigators_to_drop)
    )
    df = df[mask]
    begin_date = amce.helpers.wgms_date_to_decimal_year(df['REFERENCE_DATE'])
    end_date = amce.helpers.wgms_date_to_decimal_year(df['SURVEY_DATE'])
    # Gather regional glaciological periods
    region_periods = {}
    for path in mass_loss_dir.joinpath('regional_mass_loss_series').glob('*.csv'):
        region = path.stem.split('_')[-2]
        temp = pd.read_csv(path).set_index('YEAR')
        min_year = temp['Aw_mwe'].first_valid_index()
        max_year = temp['Aw_mwe'].last_valid_index()
        min_year = min(max(min_year, 1915), 2000)
        region_periods[region] = (min_year, max_year)
    # Include if within glaciological period
    region_id = df['GLACIER_REGION_CODE'].where(
        df['GLACIER_REGION_CODE'].ne('SAN'), df['GLACIER_SUBREGION_CODE'].map({'SAN-01': 'SA1', 'SAN-02': 'SA2'})
    )
    min_year = region_id.map(lambda x: region_periods[x][0])
    max_year = region_id.map(lambda x: region_periods[x][1])
    mask = (
        (end_date - begin_date).gt(min_length_geo) &
        (begin_date >= min_year - 2) &
        (end_date <= max_year + 1)
    )
    (
        df[mask][['SURVEY_ID']]
        .rename(columns={'SURVEY_ID': 'change_id'})
        .sort_values('change_id')
        .to_csv(output_dir / 'geodetic.csv', index=False)
    )

    # Add glaciological from before min_year that overlap geodetic
    mask &= begin_date.lt(glambie_begin_year)
    df = df[mask]
    begin_date = begin_date[mask]
    end_date = end_date[mask]
    region_id = region_id[mask]
    years = [list(range(int(begin), int(end))) for begin, end in zip(begin_date, end_date)]
    region_earlier_calibrated_years = pd.DataFrame({
        'region_id': region_id,
        'year': years
    }).explode('year').drop_duplicates()

    # glaciological
    df = pd.read_csv(mass_balance_file)
    region_id = df['GLACIER_REGION_CODE'].where(
        df['GLACIER_REGION_CODE'].ne('SAN'),
        df['GLACIER_SUBREGION_CODE'].map({'SAN-01': 'SA1', 'SAN-02': 'SA2'})
    )
    region_year = pd.MultiIndex.from_arrays([region_id, df['YEAR']])
    glacier_year = pd.MultiIndex.from_arrays([df['WGMS_ID'], df['YEAR']])
    mask = (
        (
            df['YEAR'].ge(glambie_begin_year) |
            region_year.isin(pd.MultiIndex.from_frame(region_earlier_calibrated_years)) |
            glacier_year.isin(pd.MultiIndex.from_frame(
                region_glacier_ids_sa12.drop(columns='year').drop_duplicates().merge(
                    region_earlier_calibrated_years,
                    left_on='region_code',
                    right_on='region_id',
                    how='inner'
                )[['glacier_id', 'year']].drop_duplicates()
            ))
        ) &
        df['ANNUAL_BALANCE'].notnull() &
        df['WGMS_ID'].isin(region_glacier_ids['glacier_id']) &
        region_id.notnull()
    )
    (
        df[mask][['WGMS_ID', 'YEAR']]
        .rename(columns={'WGMS_ID': 'glacier_id', 'YEAR': 'year'})
        .sort_values(['glacier_id', 'year'])
        .to_csv(output_dir / 'glaciological.csv', index=False)
    )

    # ---- Calculate regional mass balance anomalies ----

    # Build coordinate lookup table
    df = pd.read_csv(glacier_series_file)
    mask = (
        (df['GLACIER_REGION_CODE'].eq('CAU') & df['GLIMS_ID'].notnull()) |
        (df['GLACIER_REGION_CODE'].ne('CAU') & df['RGI60_ID'].notnull())
    ) & df['WGMS_ID'].notnull()
    df = df[mask]
    coords = df.set_index('WGMS_ID')[['LATITUDE', 'LONGITUDE']].round(6)

    # Build area lookup table
    mask = df['GLACIER_REGION_CODE'].eq('CAU')
    df.loc[mask, 'area_id'] = df.loc[mask, 'GLIMS_ID']
    df.loc[~mask, 'area_id'] = df.loc[~mask, 'RGI60_ID']
    area_ids = df.set_index('WGMS_ID')['area_id']
    df = pd.read_csv(rgi_area_file)
    mask = area_ids.isin(df['RGIId'])
    area_ids = area_ids[mask]
    areas = pd.Series(df.set_index('RGIId')['AREA'].loc[area_ids].values, index=area_ids.index)

    # Load mean anomalies for each glacier
    paths = sorted(long_norm_anomaly_dir.glob('*.csv'), key=lambda x: x.stem.lower())
    results = []
    for i in range(0, len(paths), 2):
        region = paths[i].stem[:3]
        print(region)
        # Anomaly mean
        mean = pd.read_csv(paths[i])
        mask = mean['YEAR'].ge(glambie_begin_year)
        mean = mean[mask]
        # Anomaly sigma
        sigma = pd.read_csv(paths[i + 1])
        mask = sigma['YEAR'].ge(glambie_begin_year)
        sigma = sigma[mask]
        # Glacier ID
        glacier_ids = mean.columns[1:].astype(int)
        # Area (km2)
        if region == 'CAU':
            # Drop glaciers without area
            mask = glacier_ids.isin(areas.index)
            glacier_ids = glacier_ids[mask]
            mean = mean.loc[:, ['YEAR'] + glacier_ids.astype(str).tolist()]
            sigma = sigma.loc[:, ['YEAR'] + glacier_ids.astype(str).tolist()]
        area = areas.loc[glacier_ids]
        # Coordinates (latitude, longitude)
        latlng = coords.loc[glacier_ids]
        # Area-weighted mean by year
        region_mean = mean.iloc[:, 1:].multiply(area.values / area.sum(), axis=1).sum(axis=1)
        # Propagate sigmas
        region_sigma = amce.propagation.regional_sigma_wrapper(
            latitude=latlng['LATITUDE'].values,
            longitude=latlng['LONGITUDE'].values,
            sigma_anom=sigma.iloc[:, 1:].multiply(area.values / area.sum(), axis=1).values,
            by_year=True,
            verbose=True
        )[-1]
        # Compile results
        mm_dd, offset = HYDROLOGICAL_END_DATE[region]
        result = pd.DataFrame({
            'region_id': int(rgi_code[region]),
            'region_code': region,
            'start_date': pd.to_datetime(mean['YEAR'].add(offset - 1).astype(str) + '-' + mm_dd).dt.strftime('%d/%m/%Y'),
            'end_date': pd.to_datetime(mean['YEAR'].add(offset).astype(str) + '-' + mm_dd).dt.strftime('%d/%m/%Y'),
            'glacier_change_observed': region_mean / 1e3,
            'glacier_change_uncertainty': region_sigma / 1e3,
            'unit': 'mwe',
            'glacier_area_reference_start': None,
            'glacier_area_reference_end': None,
            'observational_coverage_percentage': 0,
            'remarks': None
        })
        results.append(result)
    df = pd.concat(results, ignore_index=True)
    # Pull areas from elsewhere
    temp_dfs = []
    for path in mass_loss_dir.joinpath('regional_mass_loss_series').glob('*.csv'):
        region = path.stem.split('_')[-2]
        mm_dd, offset = HYDROLOGICAL_END_DATE[region]
        temp = pd.read_csv(path)
        temp = temp[temp['YEAR'].ge(glambie_begin_year)]
        temp = pd.DataFrame({
            'region_code': region,
            'start_date': pd.to_datetime(temp['YEAR'].add(offset - 1).astype(str) + '-' + mm_dd).dt.strftime('%d/%m/%Y'),
            'glacier_area_reference_start': temp['area_tot_km2'],
            'glacier_area_reference_end': temp['area_tot_km2']
        })
        temp_dfs.append(temp)
    area_df = pd.concat(temp_dfs, ignore_index=True).set_index(['region_code', 'start_date'])
    index = pd.MultiIndex.from_frame(df[['region_code', 'start_date']])
    df[['glacier_area_reference_start', 'glacier_area_reference_end']] = area_df.loc[index][['glacier_area_reference_start', 'glacier_area_reference_end']].values
    # Combine SA1 and SA2 as SAN
    mask = df['region_code'].isin(['SA1', 'SA2'])
    san_df = df[mask].groupby('start_date', as_index=False).agg({
        'region_id': 'first',
        'region_code': lambda x: 'SAN',
        'end_date': 'first',
        'glacier_change_observed': (
            lambda x:
                (x * df.loc[x.index, 'glacier_area_reference_start']).sum() /
                df.loc[x.index, 'glacier_area_reference_start'].sum()
        ),
        'glacier_change_uncertainty': (
            lambda x:
                np.sqrt((
                    df.loc[x.index, 'glacier_change_uncertainty'] *
                    df.loc[x.index, 'glacier_area_reference_start']
                ) ** 2).sum() /
                df.loc[x.index, 'glacier_area_reference_start'].sum()
        ),
        'unit': 'first',
        'glacier_area_reference_start': 'sum',
        'glacier_area_reference_end': 'sum',
        'observational_coverage_percentage': (
            lambda x:
                (x * df.loc[x.index, 'glacier_area_reference_start']).sum() /
                df.loc[x.index, 'glacier_area_reference_start'].sum()
        ),
        'remarks': 'first'
    })
    df = pd.concat((df[~mask], san_df), ignore_index=True).sort_values(['region_id', 'start_date'])

    # Calculate observational coverage percentage
    regional_areas = (
        pd.read_csv(regional_area_change_rate_file)
        .set_index('region_id')['reference_area_km2']
    )
    mask = region_glacier_ids['local'] & region_glacier_ids['year'].notnull()
    glacier_ids = region_glacier_ids[mask].copy()
    # Add glacier areas
    areas = pd.read_csv(mass_balance_area_file).set_index('glacier_id')['area_m2']
    glacier_ids['area_m2'] = areas.loc[glacier_ids['glacier_id']].values
    observed_areas = glacier_ids.groupby(['region_id', 'year'])['area_m2'].sum().div(1e6)
    df['observational_coverage_percentage'] = observed_areas.reindex(pd.MultiIndex.from_frame(
        df.assign(year=df['end_date'].str.slice(6, 10).astype(int))[['region_id', 'year']]
    )).fillna(0).div(regional_areas.loc[df['region_code']].values).mul(100).round(2).values

    # Round values
    for column in ['glacier_change_observed', 'glacier_change_uncertainty']:
        df[column] = df[column].round(3)
    for colum in ['glacier_area_reference_start', 'glacier_area_reference_end']:
        df[colum] = df[colum].round(0).astype('Int64')

    # Write to file
    df.drop(columns='region_code').to_csv(output_dir / 'submission-glaciological.csv', index=False)
