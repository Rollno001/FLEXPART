#!/usr/bin/env python3
"""
=============================================================================
STEP 6B: Plotting Spatial Concentration Maps (2x2 Grid)
         Publication Quality - Matches Deposition Figure 3
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = Path.cwd()
RESULTS_DIR = BASE_DIR / "RESULTS" / "concentration"
EDITED_FIG_DIR = BASE_DIR / "RESULTS" / "figures" / "edited_figures"
EDITED_FIG_DIR.mkdir(parents=True, exist_ok=True)

KHUMBU_LON = 86.825
KHUMBU_LAT = 27.955
SHP_DIR = BASE_DIR / "shpfiles"

# Use a slightly different colormap to visually distinguish Concentration from Deposition
CMAP_CONC = plt.get_cmap('hot_r') 

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

# =============================================================================
# HELPERS
# =============================================================================

def make_map_ax(fig, pos, extent=None, title='', fontsize=14, show_left_ylabels=True, show_bottom_xlabels=True):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
        
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
# FIGURE: Peak Days (2x2 GRID)
# =============================================================================

def plot_peak_day_maps(domain='coarse', n_peaks=4):
    print(f"  Plotting Concentration Peak Maps 2x2 ({domain})...")
    
    # 1. Load the concentration CSV
    csv_path = RESULTS_DIR / f"timeseries/BC_concentration_timeseries_{domain}.csv"
    if not csv_path.exists():
        print(f"❌ Missing {csv_path.name}")
        return
        
    df = pd.read_csv(csv_path, parse_dates=['date'])
    
    # 2. Find the top 4 days based on BC_conc_ng_m3
    top_days = df[df['status'] == 'ok'].nlargest(n_peaks, 'BC_conc_ng_m3')
    extent = [60, 105, 15, 45] if domain == 'coarse' else [75, 95, 20, 35]
    
    fig = plt.figure(figsize=(11, 8))

    peak_data = {}
    for _, row in top_days.iterrows():
        date_str = row['date'].strftime('%Y%m%d')
        fpath = RESULTS_DIR / f"daily/{domain}/BC_concentration_{date_str}_{domain}.nc"
        
        if fpath.exists():
            ds = xr.open_dataset(fpath)
            # The netcdf has it in kg m-3, we multiply by 1e12 to get ng m-3 (as noted in step 5)
            data_val = ds['BC_concentration'].values * 1e12 
            peak_data[date_str] = (ds, data_val)

    # 3. Dynamic Color Scale (Logarithmic)
    vmin = 1e-2 
    vmax = 100   # Set to 100 ng/m3 to capture extreme peaks
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for col, (_, row) in enumerate(top_days.iterrows()):
        date_str = row['date'].strftime('%Y%m%d')
        
        ax = make_map_ax(fig, (2, 2, col + 1), extent=extent, title=f"{row['date'].strftime('%d %b %Y')}", 
                         fontsize=14, 
                         show_left_ylabels=(col % 2 == 0),
                         show_bottom_xlabels=(col >= 2))
        
        if date_str not in peak_data: continue

        ds, data_val = peak_data[date_str]
        lon2d, lat2d = np.meshgrid(ds['longitude'].values, ds['latitude'].values)
        
        ax.pcolormesh(lon2d, lat2d, np.where(data_val <= vmin, np.nan, data_val), norm=norm, cmap=CMAP_CONC, zorder=1)
        add_region_boundaries(ax)
        add_khumbu_marker(ax)
        ds.close()

    plt.subplots_adjust(left=0.05, right=0.84, top=0.92, bottom=0.08, wspace=0.02, hspace=0.15)

    cbar_ax = fig.add_axes([0.86, 0.10, 0.02, 0.80]) 
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_CONC), cax=cbar_ax)
    cbar.set_label('Daily BC Concentration (ng m⁻³)', fontsize=14, fontweight='bold')
    
    out_file = EDITED_FIG_DIR / f'FigX_peak_day_concentration_maps_{domain}.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✅ Saved: {out_file.name}")
    plt.close()

if __name__ == '__main__':
    print("=" * 65)
    print(" Generating Concentration Peak Day Maps (2x2 Grid)")
    print("=" * 65)
    for d in ['coarse', 'nested']:
        plot_peak_day_maps(d)
