#!/usr/bin/env python3
"""
=============================================================================
STEP 4: Figures for BC Deposition at Khumbu Glacier
        Pre-monsoon 2021 (March, April, May) - Q1 JOURNAL EDITED VERSION
        * UPDATED: Output directory set to FINAL_NEW_EDITED
        * UPDATED: Figure 2 (Monthly Maps) split into 6 individual high-res images
        * ENHANCED: Fixed color scales, updated legends, and polished typography.
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & CLEANUP
# =============================================================================

BASE_DIR    = Path.home() / "Path.cwd()"
RESULTS_DIR = BASE_DIR / "RESULTS"
FIG_DIR     = RESULTS_DIR / "figures"

# UPDATE 1: Changed output directory to preserve previous results
EDITED_FIG_DIR = FIG_DIR / "Path.cwd()"
EDITED_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Khumbu Glacier receptor location
KHUMBU_LON = 86.825
KHUMBU_LAT = 27.955
SHP_DIR = BASE_DIR / "shpfiles"

CMAP_DEPO = plt.get_cmap('YlOrRd')
MONTH_LABELS = {'202103': 'March 2021', '202104': 'April 2021', '202105': 'May 2021'}
MONTH_COLORS = {'March': '#E07B39', 'April': '#D94F3D', 'May': '#7B3294'}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
})

# =============================================================================
# HELPERS
# =============================================================================

def make_map_ax(fig, pos, extent=None, title='', fontsize=14, show_left_ylabels=True, show_bottom_xlabels=True):
    if isinstance(pos, tuple):
        ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    else:
        ax = fig.add_subplot(pos, projection=ccrs.PlateCarree())
        
    if extent: 
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#d6e8f5', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#666666', linestyle='--', zorder=2)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_ylabels
    gl.bottom_labels = show_bottom_xlabels
    
    tick_interval = 10 if (extent and (extent[1] - extent[0] > 30)) else 5
    gl.xlocator = mticker.MultipleLocator(tick_interval)
    gl.ylocator = mticker.MultipleLocator(tick_interval)
    
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    
    gl.xlabel_style, gl.ylabel_style = {'size': 12}, {'size': 12}
    
    if title: 
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=10)
    return ax

def add_khumbu_marker(ax):
    ax.plot(KHUMBU_LON, KHUMBU_LAT, marker='*', color='cyan', markersize=14,
            transform=ccrs.PlateCarree(), zorder=10, markeredgecolor='black', markeredgewidth=1.0)

def add_region_boundaries(ax):
    import cartopy.io.shapereader as shpreader
    regions = {'Nepal': ('#2ecc71', 1.5, '--'), 'IGP': ('#3498db', 1.5, '-')}
    for region, (color, lw, ls) in regions.items():
        shp_path = SHP_DIR / f"{region}.shp"
        if shp_path.exists():
            reader = shpreader.Reader(str(shp_path))
            for geom in reader.geometries():
                ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor=color, linewidth=lw, linestyle=ls, zorder=3)

# =============================================================================
# FIGURE 1: Time Series
# =============================================================================

def plot_timeseries():
    print("  Plotting Figure 1: Daily BC deposition time series...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    scale = 1e9  

    for idx, domain in enumerate(['coarse', 'nested']):
        ax = axes[idx]
        df = pd.read_csv(RESULTS_DIR / f"timeseries/BC_deposition_timeseries_{domain}.csv", parse_dates=['date'])
        df = df[df['status'] == 'ok'].sort_values('date')

        for _, row in df.iterrows():
            ax.bar(row['date'], row['total_kg_m2'] * scale, color=MONTH_COLORS.get(row['date'].strftime('%B'), '#999'), alpha=0.8, width=0.8)

        for idx2 in df['total_kg_m2'].nlargest(3).index:
            ax.annotate(df.loc[idx2, 'date'].strftime('%d %b'), xy=(df.loc[idx2, 'date'], df.loc[idx2, 'total_kg_m2'] * scale),
                        xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=1.2))

        rolling = df.set_index('date').sort_index()['total_kg_m2'].rolling(7, center=True, min_periods=3).mean() * scale
        ax.plot(rolling.index, rolling.values, color='#1a1a2e', linewidth=2.5, label='7-day rolling mean', zorder=5)

        ax.set_ylabel('BC Deposition\n(µg m⁻²)', fontsize=14)
        ax.set_title('Coarse Domain (0.25°)' if domain == 'coarse' else 'Nested Domain (0.10°)', fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, axis='y', alpha=0.4, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=[mpatches.Patch(color=c, label=m) for m, c in MONTH_COLORS.items()] + [plt.Line2D([0], [0], color='#1a1a2e', lw=2.5, label='7-day mean')], fontsize=11, loc='upper right', ncol=4)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[1].set_xlabel('Date (2021)', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDITED_FIG_DIR / 'Fig1_BC_deposition_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# FIGURE 2: Monthly Maps (NOW SPLIT INTO INDIVIDUAL IMAGES)
# =============================================================================

def plot_monthly_maps(domain='coarse'):
    print(f"  Plotting Figure 2: Monthly total maps INDIVIDUAL ({domain})...")
    months = ['202103', '202104', '202105']
    extent = [60, 105, 15, 45] if domain == 'coarse' else [75, 95, 20, 35]
    
    # Keeping the exact same color scale so all images are comparable
    vmin = 1e-5
    vmax = 1e-2
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for month in months:
        fpath = RESULTS_DIR / f"monthly/{domain}/BC_deposition_monthly_{month}_{domain}.nc"
        if not fpath.exists():
            continue
            
        ds = xr.open_dataset(fpath)
        days_in_month = pd.Period(month, freq='M').days_in_month
        data = ds['BC_total_deposition'].values * days_in_month * 1e9

        # Create a single figure for this specific month
        fig = plt.figure(figsize=(9, 7)) 
        ax = make_map_ax(fig, 111, extent=extent, title=f"{MONTH_LABELS[month]} ({domain.capitalize()} Domain)", 
                         fontsize=14, show_left_ylabels=True, show_bottom_xlabels=True)
        
        lon2d, lat2d = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
        ax.pcolormesh(lon2d, lat2d, np.where(data <= 0, np.nan, data), norm=norm, cmap=CMAP_DEPO, zorder=1)
        
        add_region_boundaries(ax)
        add_khumbu_marker(ax)

        # Add the legend to every image
        legend_elements = [
            mpatches.Patch(facecolor='none', edgecolor='#3498db', lw=1.5, label='IGP'),
            mpatches.Patch(facecolor='none', edgecolor='#2ecc71', lw=1.5, linestyle='--', label='Nepal'),
            plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='cyan', markeredgecolor='black', markersize=10, label='Khumbu Glacier')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)

        # Add the colorbar specifically to this axis
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DEPO), ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label('BC Source Contribution (µg m⁻²)', fontsize=12, fontweight='bold')
        
        # Save individually
        month_name_file = MONTH_LABELS[month].replace(" ", "_")
        plt.savefig(EDITED_FIG_DIR / f'Fig2_monthly_total_{month_name_file}_{domain}.png', dpi=300, bbox_inches='tight')
        plt.close()
        ds.close()

# =============================================================================
# FIGURE 3: Peak Days (2x2 GRID - NO GAPS)
# =============================================================================

def plot_peak_day_maps(domain='coarse', n_peaks=4):
    print(f"  Plotting Figure 3: Peak day maps 2x2 ({domain})...")
    df = pd.read_csv(RESULTS_DIR / f"timeseries/BC_deposition_timeseries_{domain}.csv", parse_dates=['date'])
    top_days = df[df['status'] == 'ok'].nlargest(n_peaks, 'total_kg_m2')
    extent = [60, 105, 15, 45] if domain == 'coarse' else [75, 95, 20, 35]
    
    fig = plt.figure(figsize=(11, 8))

    peak_data = {}
    for _, row in top_days.iterrows():
        fpath = RESULTS_DIR / f"daily/{domain}/BC_deposition_{row['date'].strftime('%Y%m%d')}_{domain}.nc"
        if fpath.exists():
            ds = xr.open_dataset(fpath)
            data_ug = ds['BC_total_deposition'].values * 1e9
            peak_data[row['date'].strftime('%Y%m%d')] = (ds, data_ug)

    vmin = 1e-6
    vmax = 1e-3
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for col, (_, row) in enumerate(top_days.iterrows()):
        date_str = row['date'].strftime('%Y%m%d')
        
        ax = make_map_ax(fig, (2, 2, col + 1), extent=extent, title=f"{row['date'].strftime('%d %b %Y')}", 
                         fontsize=14, 
                         show_left_ylabels=(col % 2 == 0),
                         show_bottom_xlabels=(col >= 2))
        
        if date_str not in peak_data: continue

        ds, data_ug = peak_data[date_str]
        lon2d, lat2d = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
        ax.pcolormesh(lon2d, lat2d, np.where(data_ug <= 0, np.nan, data_ug), norm=norm, cmap=CMAP_DEPO, zorder=1)
        add_region_boundaries(ax)
        add_khumbu_marker(ax)
        ds.close()

    plt.subplots_adjust(left=0.05, right=0.84, top=0.92, bottom=0.08, wspace=0.02, hspace=0.15)

    cbar_ax = fig.add_axes([0.86, 0.10, 0.02, 0.80]) 
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DEPO), cax=cbar_ax)
    cbar.set_label('Daily BC Deposition (µg m⁻²)', fontsize=14, fontweight='bold')
    
    plt.savefig(EDITED_FIG_DIR / f'Fig3_peak_day_source_maps_{domain}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# =============================================================================
# FIGURE 4: Seasonal Total
# =============================================================================

def plot_seasonal_total(domain='coarse'):
    print(f"  Plotting Figure 4: Seasonal total map ({domain})...")
    all_files = sorted((RESULTS_DIR / f"daily/{domain}").glob(f"BC_deposition_2021*_{domain}.nc"))
    if not all_files: return

    total_data, lats, lons = None, None, None
    for fpath in all_files:
        ds = xr.open_dataset(fpath)
        d = ds['BC_total_deposition'].values
        if total_data is None:
            total_data, lats, lons = d.copy(), ds['latitude'].values, ds['longitude'].values
        else:
            total_data += np.where(np.isnan(d), 0, d)
        ds.close()

    total_data_ug = total_data * 1e9

    fig, ax = plt.subplots(1, 1, figsize=(14, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([55, 110, 10, 45] if domain == 'coarse' else [72, 98, 18, 38], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#d6e8f5', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor='#666666', linestyle='--', zorder=2)

    data_plot = np.where(total_data_ug < 1e-4, np.nan, total_data_ug)
    
    vmin = 1e-4
    vmax = 1e-1
    
    pcm = ax.pcolormesh(np.meshgrid(lons, lats)[0], np.meshgrid(lons, lats)[1], data_plot,
                        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='hot_r', zorder=1, alpha=0.9)

    add_region_boundaries(ax)
    add_khumbu_marker(ax)
    
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()
    ax.plot([min_lon, max_lon, max_lon, min_lon, min_lon], 
            [min_lat, min_lat, max_lat, max_lat, min_lat], 
            color='black', linestyle=':', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=5)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    cbar = fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.025, pad=0.04, shrink=0.85)
    cbar.set_label('Seasonal Grid-Cell BC Deposition (µg m⁻²)', fontsize=14, fontweight='bold')
    
    legend_elements = [
        mpatches.Patch(facecolor='none', edgecolor='#3498db', lw=1.5, label='Indo-Gangetic Plain'),
        mpatches.Patch(facecolor='none', edgecolor='#2ecc71', lw=1.5, linestyle='--', label='Nepal'),
        plt.Line2D([0], [0], color='black', linestyle=':', lw=1.5, label='Model Domain Boundary'),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='cyan', markeredgecolor='black', markersize=12, label='Khumbu Glacier Receptor')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12, framealpha=0.95)

    plt.savefig(EDITED_FIG_DIR / f'Fig4_seasonal_total_source_map_{domain}.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# FIGURE 5: Domain Comparison
# =============================================================================

def plot_domain_comparison():
    print("  Plotting Figure 5: Domain comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    dfs = {dom: pd.read_csv(RESULTS_DIR / f"timeseries/BC_deposition_timeseries_{dom}.csv", parse_dates=['date']).sort_values('date') for dom in ['coarse', 'nested']}
    scale = 1e9

    for ax, (month, dates) in zip(axes, {'March': ('2021-03-01', '2021-03-31'), 'April': ('2021-04-01', '2021-04-30'), 'May': ('2021-05-01', '2021-05-31')}.items()):
        for dom, ls in [('coarse', '-'), ('nested', '--')]:
            sub = dfs[dom][(dfs[dom]['date'] >= dates[0]) & (dfs[dom]['date'] <= dates[1])]
            ax.plot(sub['date'], sub['total_kg_m2'] * scale, linestyle=ls, linewidth=2.0, color=MONTH_COLORS[month], label=f'{dom.capitalize()} ({"0.25°" if dom == "coarse" else "0.10°"})')
            ax.fill_between(sub['date'], 0, sub['total_kg_m2'] * scale, alpha=0.15, color=MONTH_COLORS[month])
        
        ax.set_title(month, fontsize=14, fontweight='bold', loc='left', pad=8)
        ax.set_ylabel('BC Deposition\n(µg m⁻²)', fontsize=14)
        ax.grid(True, axis='y', alpha=0.4, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=12, loc='upper right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[-1].set_xlabel('Date (2021)', fontsize=14)
    plt.tight_layout()
    plt.savefig(EDITED_FIG_DIR / 'Fig5_coarse_vs_nested_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print(" Generating EDITED Figures...")
    print("=" * 65)
    plot_timeseries()
    for d in ['coarse', 'nested']: plot_monthly_maps(d)
    for d in ['coarse', 'nested']: plot_peak_day_maps(d)
    for d in ['coarse', 'nested']: plot_seasonal_total(d)
    plot_domain_comparison()
    print(f"\n All figures saved in: {Path.cwd()}")
