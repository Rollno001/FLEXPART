#!/usr/bin/env python3
"""
=============================================================================
STEP 7B (FINAL): Plotting FLEXPART Emission Sensitivity Footprints
                 2x3 Panel (Coarse vs Nested)
                 Includes: Locked Colorbars, Bounding Boxes, Inner Labels
=============================================================================
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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

BASE_DIR = Path.cwd()
CONC_DIRS = {
    'March': BASE_DIR / 'CONCENTRATION/gridtime_all_march',
    'April': BASE_DIR / 'CONCENTRATION/gridtime_all_april',
    'May':   BASE_DIR / 'CONCENTRATION/gridtime_all_may',
}

SHP_DIR = BASE_DIR / "Path.cwd()"
OUT_DIR = BASE_DIR / "Path.cwd()"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KHUMBU_LON = 86.825
KHUMBU_LAT = 27.955

CMAP_FOOTPRINT = plt.get_cmap('YlGnBu') 

# Global Colorbar Limits (Locked across all subplots)
VMIN = 1.0     
VMAX = 10000.0   

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 14})

# =============================================================================
# HELPERS
# =============================================================================

def make_map_ax(fig, pos, extent=None, title='', ylabel='', show_left_ylabels=True, show_bottom_xlabels=True):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    if extent: 
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
    ax.add_feature(cfeature.LAND, facecolor='#e0e0e0', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='#d6e8f5', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#333333', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='#666666', linestyle='--', zorder=2)
    
    # Thin, subtle gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='grey', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = show_left_ylabels
    gl.bottom_labels = show_bottom_xlabels
    
    tick_interval = 10 if (extent and (extent[1] - extent[0] > 30)) else 5
    gl.xlocator = mticker.MultipleLocator(tick_interval)
    gl.ylocator = mticker.MultipleLocator(tick_interval)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style, gl.ylabel_style = {'size': 10}, {'size': 10}
    
    if title: 
        ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    
    # Add domain label to the far left plots
    if ylabel:
        ax.text(-0.15, 0.5, ylabel, va='center', ha='center', rotation='vertical', 
                transform=ax.transAxes, fontsize=14, fontweight='bold')
        
    return ax

def add_khumbu_marker(ax):
    ax.plot(KHUMBU_LON, KHUMBU_LAT, marker='*', color='magenta', markersize=12,
            transform=ccrs.PlateCarree(), zorder=10, markeredgecolor='black', markeredgewidth=0.8)

def add_region_boundaries(ax):
    import cartopy.io.shapereader as shpreader
    regions = {'Nepal': ('#e74c3c', 1.2, '--')} 
    for region, (color, lw, ls) in regions.items():
        shp_path = SHP_DIR / f"{region}.shp"
        if shp_path.exists():
            reader = shpreader.Reader(str(shp_path))
            for geom in reader.geometries():
                ax.add_geometries([geom], ccrs.PlateCarree(), facecolor='none', edgecolor=color, linewidth=lw, linestyle=ls, zorder=3)

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def get_monthly_footprint(month_name, domain):
    folder = CONC_DIRS[month_name]
    if not folder.exists(): return None, None, None
    
    all_files = sorted(folder.glob("grid_time_*.nc"))
    if domain == "nested":
        files = [f for f in all_files if f.name.endswith("_nest.nc")]
    else:
        files = [f for f in all_files if not f.name.endswith("_nest.nc")]
    
    if not files: return None, None, None
        
    monthly_footprint = None
    lats, lons = None, None
    
    for fpath in files:
        try:
            ds = xr.open_dataset(fpath)
            es = ds['spec001_mr'].values
            es_surface = np.sum(es, axis=(0, 2))[0, 0, :, :] 
            
            if monthly_footprint is None:
                monthly_footprint = es_surface.copy()
                lats = ds['latitude'].values
                lons = ds['longitude'].values
            else:
                monthly_footprint += es_surface
            ds.close()
        except Exception:
            pass
            
    return monthly_footprint, lats, lons

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    print("=" * 65)
    print(" Generating Publication 2x3 Panel: Emission Footprints")
    print("=" * 65)

    fig = plt.figure(figsize=(16, 9))
    norm = mcolors.LogNorm(vmin=VMIN, vmax=VMAX)
    
    months = ['March', 'April', 'May']
    domains = ['coarse', 'nested']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    plot_idx = 1
    for row, domain in enumerate(domains):
        extent = [60, 105, 15, 45] if domain == 'coarse' else [75, 95, 20, 35]
        ylabel = 'Coarse Domain (0.25°)' if domain == 'coarse' else 'Nested Domain (0.10°)'
        
        for col, month in enumerate(months):
            print(f"  Processing {domain.upper()} - {month}...")
            data, lats, lons = get_monthly_footprint(month, domain)
            
            # Formatting logic (Only month name in title now)
            title = month if row == 0 else ""
            show_left = (col == 0)
            show_bottom = (row == 1)
            y_label_text = ylabel if col == 0 else ""
            
            ax = make_map_ax(fig, (2, 3, plot_idx), extent=extent, title=title, 
                             ylabel=y_label_text, show_left_ylabels=show_left, 
                             show_bottom_xlabels=show_bottom)
            
            if data is not None:
                lon2d, lat2d = np.meshgrid(lons, lats)
                plot_data = np.where(data <= VMIN, np.nan, data)
                pcm = ax.pcolormesh(lon2d, lat2d, plot_data, norm=norm, cmap=CMAP_FOOTPRINT, zorder=1)
                
            add_region_boundaries(ax)
            add_khumbu_marker(ax)
            
            # --- CRITICAL FIX 1: Add inner panel labels (a, b, c...) ---
            ax.text(0.03, 0.95, panel_labels[plot_idx-1], transform=ax.transAxes, 
                    fontsize=14, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'), zorder=20)

            # --- CRITICAL FIX 2: Draw nested bounding box on coarse maps ---
            if domain == 'coarse':
                ax.add_patch(mpatches.Rectangle(xy=[75, 20], width=20, height=15,
                                                facecolor='none', edgecolor='black',
                                                linewidth=1.8, linestyle='--',
                                                transform=ccrs.PlateCarree(), zorder=15))
            
            # Add legend only to the first panel (removed the zorder argument here)
            if plot_idx == 1:
                legend_elements = [
                    mpatches.Patch(facecolor='none', edgecolor='#e74c3c', lw=1.2, linestyle='--', label='Nepal'),
                    plt.Line2D([0], [0], color='black', linestyle='--', lw=1.8, label='Nested Domain Bounds'),
                    plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='magenta', markeredgecolor='black', markersize=10, label='Khumbu Glacier')
                ]
                ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.95)
                
            plot_idx += 1

    # Adjust layout to make room for the horizontal colorbar at the bottom
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.18, wspace=0.1, hspace=0.1)

    # Add single, shared horizontal colorbar
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03]) # [left, bottom, width, height]
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP_FOOTPRINT), cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Surface Residence Time Footprint (seconds)', fontsize=13, fontweight='bold')
    
    # Save
    out_file = OUT_DIR / 'FigX_FLEXPART_Footprint_2x3.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n   Saved Figure: {out_file.name}")
    plt.close()

if __name__ == '__main__':
    main()
