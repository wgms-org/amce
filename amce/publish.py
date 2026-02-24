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
