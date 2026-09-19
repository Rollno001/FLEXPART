#!/usr/bin/env python3
"""
=============================================================================
STEP 7: Age-Class Resolved BC Analysis at Khumbu Glacier
        

        Age classes defined in this  FLEXPART runs:
        Class 0:  6hr    (0.25 days)  → very local / same day
        Class 1: 12hr    (0.5 days)
        Class 2: 24hr    (1 day)
        Class 3: 48hr    (2 days)
        Class 4: 72hr    (3 days)
        Class 5: 96hr    (4 days)
        Class 6: 120hr   (5 days)
        Class 7: 168hr   (7 days)
        Class 8: 240hr   (10 days)   → most distant source possible

        What we do:
        For each age class separately, compute:
          D_age [kg m-2] = Σ_t Σ_x Σ_y [ ES_age(x,y,t) × E_BC(x,y) / H × dt ]
          C_age [kg m-3] = Σ_t Σ_x Σ_y [ ES_age(x,y,t) × E_BC(x,y) / H ]

        This tells us: of all the BC that arrived at Khumbu on a given day,
        what fraction left its source region 1 day ago vs 3 days ago vs 7 days ago?

Outputs:
  RESULTS/ageclass/
    BC_ageclass_timeseries_deposition_{domain}.csv
    BC_ageclass_timeseries_concentration_{domain}.csv
    BC_ageclass_source_maps_{date}_{domain}.nc  (for peak days)
    Fig11_ageclass_analysis.png
    Fig12_ageclass_source_maps_peak_day.png
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path.home() / "Path.cwd()"

# Age class definitions (seconds → human-readable labels)
AGE_CLASSES_SEC = [21600, 43200, 86400, 172800, 259200, 345600, 432000, 604800, 864000]
AGE_LABELS = ['6hr', '12hr', '1day', '2day', '3day', '4day', '5day', '7day', '10day']
AGE_DAYS   = [0.25,  0.5,   1.0,   2.0,   3.0,   4.0,   5.0,   7.0,   10.0]

# Grouped bins for cleaner visualization
# "How many days of travel time?"
AGE_GROUPS = {
    '0–1 day\n(very local)':   [0, 1, 2],    # 6hr, 12hr, 24hr
    '1–3 days\n(regional)':    [3, 4],        # 48hr, 72hr
    '3–5 days\n(long-range)':  [5, 6],        # 96hr, 120hr
    '5–10 days\n(very distant)':[7, 8],       # 168hr, 240hr
}
GROUP_COLORS = {
    '0–1 day\n(very local)':    '#2ecc71',
    '1–3 days\n(regional)':     '#f39c12',
    '3–5 days\n(long-range)':   '#e74c3c',
    '5–10 days\n(very distant)':'#8e44ad',
}

# FLEXPART directories
FLEXPART_DIRS = {
    'dry': {
        'March': BASE_DIR / 'March_dry',
        'April': BASE_DIR / 'April_dry',
        'May':   BASE_DIR / 'May_dry',
    }
}
CONC_DIRS = {
    'March': BASE_DIR / 'CONCENTRATION/gridtime_all_march',
    'April': BASE_DIR / 'CONCENTRATION/gridtime_all_april',
    'May':   BASE_DIR / 'CONCENTRATION/gridtime_all_may',
}
ENSEM_DIRS = {
    'coarse': BASE_DIR / 'ENSEM_REGRID/coarse_0.25deg',
    'nested': BASE_DIR / 'ENSEM_REGRID/nested_0.1deg',
}
OUT_DIR  = BASE_DIR / 'RESULTS' / 'ageclass'
FIG_DIR  = BASE_DIR / 'RESULTS' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

DT     = 10800.0
H_SURF = 50.0
KHUMBU_LON, KHUMBU_LAT = 86.825, 27.955
SHP_DIR = BASE_DIR / "shpfiles"

# =============================================================================
# FILE HELPERS
# =============================================================================

def get_depo_file(date_str, domain='coarse'):
    month_num  = int(date_str[4:6])
    month_name = {3:'March',4:'April',5:'May'}[month_num]
    folder = FLEXPART_DIRS['dry'][month_name] / date_str
    suffix = '_nest' if domain == 'nested' else ''
    fname  = folder / f"grid_drydep_{date_str}000000{suffix}.nc"
    return fname if fname.exists() else None

def get_conc_file(date_str, domain='coarse'):
    month_num  = int(date_str[4:6])
    month_name = {3:'March',4:'April',5:'May'}[month_num]
    folder = CONC_DIRS[month_name]
    suffix = '_nest' if domain == 'nested' else ''
    fname  = folder / f"grid_time_{date_str}000000{suffix}.nc"
    return fname if fname.exists() else None

def get_ensemfire(date_str, domain='coarse'):
    key  = 'coarse' if domain == 'coarse' else 'nested'
    tag  = 'coarse' if domain == 'coarse' else 'nested'
    fname = ENSEM_DIRS[key] / f"EnsemFire_BC_{tag}_{date_str}.nc"
    return fname if fname.exists() else None

def load_bc_emissions(ensemfire_file):
    with nc.Dataset(ensemfire_file) as f:
        bc = f.variables['BC'][:]
        if hasattr(bc, 'filled'): bc = bc.filled(0.0)
        bc = np.squeeze(bc)
        bc = np.where(bc < 0, 0.0, bc)
    return bc

def load_sensitivity_by_ageclass(flexpart_file):
    """
    Load spec001_mr WITHOUT summing age classes.
    Returns array shape: (nageclass=9, time=80, lat, lon)
    """
    with nc.Dataset(flexpart_file) as f:
        es = f.variables['spec001_mr'][:]
        if hasattr(es, 'filled'): es = es.filled(0.0)
        # shape: (9, 1, 80, 8, lat, lon)
        es = es[:, 0, :, 0, :, :]   # drop pointspec, take height=0 (50m)
        # shape: (9, 80, lat, lon)
        lats = f.variables['latitude'][:]
        lons = f.variables['longitude'][:]
    return es, lats, lons

# =============================================================================
# CORE: Compute deposition/concentration per age class for one day
# =============================================================================

def compute_ageclass_deposition(es_by_age, bc_flux):
    """
    es_by_age shape: (9, 80, lat, lon) — units: m (dry dep sensitivity)
    bc_flux   shape: (lat, lon) — units: kg m-2 s-1
    Returns: array shape (9,) — deposition per age class [kg m-2]
    """
    bc_vol = bc_flux / H_SURF  # [kg m-3 s-1]
    depo_per_age = np.zeros(9)
    for a in range(9):
        # ES_a shape: (80, lat, lon)
        # sum over time (80) and space (lat,lon)
        depo_per_age[a] = np.sum(es_by_age[a] * bc_vol[np.newaxis,:,:] * DT)
    return depo_per_age  # [kg m-2]

def compute_ageclass_concentration(es_by_age, bc_flux):
    """
    es_by_age: units 's'
    Returns: array shape (9,) — concentration per age class [kg m-3]
    """
    bc_vol = bc_flux / H_SURF  # [kg m-3 s-1]
    conc_per_age = np.zeros(9)
    for a in range(9):
        conc_per_age[a] = np.sum(es_by_age[a] * bc_vol[np.newaxis,:,:])
    return conc_per_age  # [kg m-3]

def compute_ageclass_source_maps(es_by_age, bc_flux, mode='deposition'):
    """
    Returns source map per age class: shape (9, lat, lon)
    """
    bc_vol = bc_flux / H_SURF
    maps = np.zeros((9,) + bc_flux.shape)
    for a in range(9):
        if mode == 'deposition':
            maps[a] = np.sum(es_by_age[a] * bc_vol[np.newaxis,:,:] * DT, axis=0)
        else:
            maps[a] = np.sum(es_by_age[a] * bc_vol[np.newaxis,:,:], axis=0)
    return maps

# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def run_ageclass_analysis(domain='coarse'):
    print(f"\n{'='*60}")
    print(f" Age-Class Analysis — Domain: {domain.upper()}")
    print(f"{'='*60}")

    all_dates = [d.strftime('%Y%m%d')
                 for d in pd.date_range('2021-03-01','2021-05-31')]

    depo_records = []
    conc_records = []

    # Also track peak day source maps for plotting
    peak_depo_date = None
    peak_conc_date = None
    peak_depo_val  = 0
    peak_conc_val  = 0

    for date_str in all_dates:
        depo_file  = get_depo_file(date_str, domain)
        conc_file  = get_conc_file(date_str, domain)
        ensem_file = get_ensemfire(date_str, domain)

        if ensem_file is None:
            continue

        bc_flux = load_bc_emissions(ensem_file)

        # --- DEPOSITION ---
        if depo_file:
            try:
                es_depo, lats, lons = load_sensitivity_by_ageclass(depo_file)
                depo_ages = compute_ageclass_deposition(es_depo, bc_flux)
                total_depo = np.sum(depo_ages)

                row = {'date': date_str}
                for a in range(9):
                    row[f'depo_age{a}_{AGE_LABELS[a]}_kg_m2'] = float(depo_ages[a])
                row['depo_total_kg_m2'] = float(total_depo)
                depo_records.append(row)

                if total_depo > peak_depo_val:
                    peak_depo_val  = total_depo
                    peak_depo_date = date_str
                    peak_depo_maps = compute_ageclass_source_maps(es_depo, bc_flux, 'deposition')
                    peak_depo_lats, peak_depo_lons = lats, lons

                print(f"  {date_str} depo: total={total_depo*1e12:.2f} ng m⁻²  "
                      f"| age breakdown (ng m⁻²): "
                      + "  ".join([f"{AGE_LABELS[a]}:{depo_ages[a]*1e12:.3f}"
                                   for a in range(9)]))
            except Exception as e:
                print(f"  {date_str} depo ERROR: {e}")

        # --- CONCENTRATION ---
        if conc_file:
            try:
                es_conc, lats_c, lons_c = load_sensitivity_by_ageclass(conc_file)
                conc_ages = compute_ageclass_concentration(es_conc, bc_flux)
                total_conc = np.sum(conc_ages)

                row = {'date': date_str}
                for a in range(9):
                    row[f'conc_age{a}_{AGE_LABELS[a]}_ng_m3'] = float(conc_ages[a] * 1e12)
                row['conc_total_ng_m3'] = float(total_conc * 1e12)
                conc_records.append(row)

                if total_conc > peak_conc_val:
                    peak_conc_val  = total_conc
                    peak_conc_date = date_str
                    peak_conc_maps = compute_ageclass_source_maps(es_conc, bc_flux, 'concentration')
                    peak_conc_lats, peak_conc_lons = lats_c, lons_c

            except Exception as e:
                print(f"  {date_str} conc ERROR: {e}")

    # Save CSVs
    ts_dir = OUT_DIR / 'timeseries'
    ts_dir.mkdir(parents=True, exist_ok=True)

    df_depo = pd.DataFrame(depo_records)
    df_depo['date'] = pd.to_datetime(df_depo['date'], format='%Y%m%d')
    df_depo.to_csv(ts_dir / f"BC_ageclass_deposition_{domain}.csv", index=False)

    df_conc = pd.DataFrame(conc_records)
    df_conc['date'] = pd.to_datetime(df_conc['date'], format='%Y%m%d')
    df_conc.to_csv(ts_dir / f"BC_ageclass_concentration_{domain}.csv", index=False)

    print(f"\n  ✅ Timeseries saved to {ts_dir}")
    print(f"  Peak deposition day : {peak_depo_date}  ({peak_depo_val*1e12:.2f} ng m⁻²)")
    print(f"  Peak concentration day: {peak_conc_date}  ({peak_conc_val*1e12:.4f} ng m⁻³)")

    return (df_depo, df_conc,
            peak_depo_date, peak_depo_maps, peak_depo_lats, peak_depo_lons,
            peak_conc_date, peak_conc_maps, peak_conc_lats, peak_conc_lons)

# =============================================================================
# FIGURE 11: Age-Class Time Series — Stacked Bar Chart
# =============================================================================

def plot_ageclass_timeseries(df_depo, df_conc, domain='coarse'):
    print("  Plotting Figure 11: Age-class stacked bar charts...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('Age-Class Resolved BC at Khumbu Glacier — Transport Time Analysis\n'
                 f'Pre-monsoon 2021  |  Domain: '
                 f'{"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
                 fontsize=13, fontweight='bold', y=0.98)

    cmap = plt.cm.get_cmap('Spectral_r', 9)
    age_colors = [cmap(i/8) for i in range(9)]

    for ax_idx, (df, mode, ylabel, scale, unit) in enumerate([
        (df_depo, 'deposition', 'BC Deposition (ng m⁻²)', 1e12, 'ng m⁻²'),
        (df_conc, 'concentration', 'BC Concentration (ng m⁻³)', 1.0, 'ng m⁻³'),
    ]):
        ax = axes[ax_idx]
        df_ok = df.sort_values('date').copy()
        dates = df_ok['date'].values

        # Build stacked bars for each age class
        bottom = np.zeros(len(df_ok))
        for a in range(9):
            if mode == 'deposition':
                col = f'depo_age{a}_{AGE_LABELS[a]}_kg_m2'
                vals = df_ok[col].values * scale
            else:
                col = f'conc_age{a}_{AGE_LABELS[a]}_ng_m3'
                vals = df_ok[col].values

            vals = np.where(vals < 0, 0, vals)
            ax.bar(dates, vals, bottom=bottom, color=age_colors[a],
                   alpha=0.9, width=np.timedelta64(20,'h'),
                   label=f'{AGE_LABELS[a]}', edgecolor='none')
            bottom += vals

        ax.set_ylabel(ylabel, fontsize=10)
        ax.yaxis.grid(True, alpha=0.4, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
        for m_start in ['2021-04-01', '2021-05-01']:
            ax.axvline(pd.Timestamp(m_start), color='#aaaaaa', linewidth=1,
                       linestyle=':', zorder=1)

        # Stacked bar legend
        handles = [mpatches.Patch(color=age_colors[a], label=f'{AGE_LABELS[a]}')
                   for a in range(9)]
        ax.legend(handles=handles, fontsize=8, ncol=9,
                  loc='upper right', framealpha=0.8,
                  title='Transport age (time since leaving source region)',
                  title_fontsize=8)
        title = 'Dry Deposition' if mode == 'deposition' else 'Atmospheric Concentration'
        ax.set_title(title, fontsize=10, style='italic', loc='left', pad=4)

    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[1].set_xlabel('Date', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f'Fig11_ageclass_timeseries_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {out.name}")


# =============================================================================
# FIGURE 12: Age-Class Pie Charts — mean percentage contribution
# =============================================================================

def plot_ageclass_pies(df_depo, df_conc, domain='coarse'):
    print("  Plotting Figure 12: Age-class pie charts (mean % contribution)...")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Mean % Contribution by Transport Age to Khumbu BC\n'
                 f'Pre-monsoon 2021 | {"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
                 fontsize=13, fontweight='bold')

    cmap = plt.cm.get_cmap('Spectral_r', 9)
    age_colors = [cmap(i/8) for i in range(9)]

    # Months + full season
    periods = [
        ('March', '2021-03-01', '2021-03-31'),
        ('April', '2021-04-01', '2021-04-30'),
        ('May',   '2021-05-01', '2021-05-31'),
        ('Mar–May\n(Full Season)', '2021-03-01', '2021-05-31'),
    ]

    for ax, (label, s, e) in zip(axes, periods):
        mask = (df_depo['date'] >= s) & (df_depo['date'] <= e)
        sub  = df_depo[mask]

        totals = np.zeros(9)
        for a in range(9):
            col = f'depo_age{a}_{AGE_LABELS[a]}_kg_m2'
            if col in sub.columns:
                totals[a] = sub[col].sum()

        grand = totals.sum()
        if grand <= 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        pcts = totals / grand * 100

        # Group small slices for clarity
        threshold = 2.0  # merge age classes < 2% into "other"
        merged_vals   = []
        merged_labels = []
        merged_colors = []
        other_val = 0.0
        for a in range(9):
            if pcts[a] >= threshold:
                merged_vals.append(totals[a])
                merged_labels.append(f'{AGE_LABELS[a]}\n({pcts[a]:.1f}%)')
                merged_colors.append(age_colors[a])
            else:
                other_val += totals[a]
        if other_val > 0:
            merged_vals.append(other_val)
            merged_labels.append(f'<2%\n({other_val/grand*100:.1f}%)')
            merged_colors.append('#cccccc')

        wedges, texts = ax.pie(merged_vals, colors=merged_colors,
                               startangle=90, counterclock=False,
                               wedgeprops={'edgecolor': 'white', 'linewidth': 0.8})
        ax.legend(wedges, merged_labels, loc='lower center',
                  bbox_to_anchor=(0.5, -0.35), fontsize=7.5, ncol=2,
                  framealpha=0.8)
        ax.set_title(label, fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    out = FIG_DIR / f'Fig12_ageclass_pies_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {out.name}")


# =============================================================================
# FIGURE 13: Age-class source maps for peak deposition day
# =============================================================================

def plot_ageclass_source_maps(peak_date, age_maps, lats, lons,
                               domain='coarse', mode='deposition'):
    print(f"  Plotting Figure 13: Age-class source maps for peak {mode} day ({peak_date})...")

    extent = [60,105,15,55] if domain == 'coarse' else [75,95,20,35]

    # Plot selected age classes: 1day, 2day, 3day, 5day, 7day, 10day
    selected = [2, 3, 4, 6, 7, 8]
    labels   = [AGE_LABELS[a] for a in selected]

    fig = plt.figure(figsize=(20, 8))
    label_str = f"{datetime.strptime(peak_date,'%Y%m%d').strftime('%d %B %Y')}"
    mode_str  = 'Dry Deposition' if mode == 'deposition' else 'Concentration'
    fig.suptitle(f'Age-Class Source Contribution Maps — Peak {mode_str} Day: {label_str}\n'
                 f'Khumbu Glacier | '
                 f'{"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
                 fontsize=13, fontweight='bold')

    # Common colour scale across all age panels
    vmaxes = []
    for a in selected:
        d = age_maps[a]
        if np.any(d > 0):
            vmaxes.append(np.nanpercentile(d[d>0], 99))
    vmax = max(vmaxes) if vmaxes else 1e-14
    norm = mcolors.LogNorm(vmin=vmax*1e-5, vmax=vmax)

    for col, (a, lbl) in enumerate(zip(selected, labels)):
        if isinstance((1, 6, col+1), tuple):
            ax = fig.add_subplot(1, 6, col+1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND,      facecolor='#f5f0e8', zorder=0)
        ax.add_feature(cfeature.OCEAN,     facecolor='#d6e8f5', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='#555', zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.4, edgecolor='#999',
                       linestyle='--', zorder=2)

        gl = ax.gridlines(draw_labels=(col==0), linewidth=0.3, color='grey',
                          alpha=0.4, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}

        data = age_maps[a]
        lon2d, lat2d = np.meshgrid(lons, lats)
        data_plot = np.where(data <= 0, np.nan, data)
        ax.pcolormesh(lon2d, lat2d, data_plot,
                      norm=norm, cmap='YlOrRd',
                      transform=ccrs.PlateCarree(), zorder=1)

        ax.plot(KHUMBU_LON, KHUMBU_LAT, marker='*', color='cyan', markersize=9,
                transform=ccrs.PlateCarree(), zorder=10,
                markeredgecolor='black', markeredgewidth=0.5)

        # Fraction of total
        total_all = np.sum([age_maps[i] for i in range(9)])
        frac = np.sum(data)/total_all*100 if total_all > 0 else 0
        ax.set_title(f'Age: {lbl}\n{frac:.1f}% of total',
                     fontsize=9, fontweight='bold', pad=5)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.012, 0.65])
    sm = plt.cm.ScalarMappable(norm=norm, cmap='YlOrRd')
    cbar = fig.colorbar(sm, cax=cbar_ax)
    unit = 'kg m⁻²' if mode == 'deposition' else 'kg m⁻³'
    cbar.set_label(f'BC {mode_str} ({unit})', fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    plt.subplots_adjust(left=0.02, right=0.91, top=0.86, bottom=0.05, wspace=0.05)
    suffix = 'depo' if mode == 'deposition' else 'conc'
    out = FIG_DIR / f'Fig13_ageclass_sourcemaps_{suffix}_{peak_date}_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✅ Saved: {out.name}")


# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

def print_summary_table(df_depo, df_conc, domain):
    print(f"\n{'='*65}")
    print(f" SUMMARY: Mean % Contribution by Age Group — {domain.upper()}")
    print(f"{'='*65}")
    print(f" (Dry Deposition)")
    print(f" {'Age Group':<25} {'March':>10} {'April':>10} {'May':>10} {'Season':>10}")
    print(f" {'-'*65}")

    periods = {
        'March': ('2021-03-01','2021-03-31'),
        'April': ('2021-04-01','2021-04-30'),
        'May':   ('2021-05-01','2021-05-31'),
        'Season':('2021-03-01','2021-05-31'),
    }

    for group_name, indices in AGE_GROUPS.items():
        row_str = f" {group_name.replace(chr(10),' '):<25}"
        for pname, (s, e) in periods.items():
            mask  = (df_depo['date'] >= s) & (df_depo['date'] <= e)
            sub   = df_depo[mask]
            group_sum = 0.0
            total_sum = 0.0
            for a in range(9):
                col = f'depo_age{a}_{AGE_LABELS[a]}_kg_m2'
                if col in sub.columns:
                    v = sub[col].sum()
                    total_sum += v
                    if a in indices:
                        group_sum += v
            pct = group_sum/total_sum*100 if total_sum > 0 else 0
            row_str += f" {pct:>9.1f}%"
        print(row_str)
    print(f"{'='*65}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print(" Age-Class BC Analysis — Khumbu Glacier")
    print(" 'Kati tadha dekhi ko BC aaipugyo?'")
    print("=" * 65)

    for domain in ['coarse', 'nested']:
        results = run_ageclass_analysis(domain)
        (df_depo, df_conc,
         peak_depo_date, peak_depo_maps, peak_depo_lats, peak_depo_lons,
         peak_conc_date, peak_conc_maps, peak_conc_lats, peak_conc_lons) = results

        print_summary_table(df_depo, df_conc, domain)
        plot_ageclass_timeseries(df_depo, df_conc, domain)
        plot_ageclass_pies(df_depo, df_conc, domain)

        if peak_depo_date:
            plot_ageclass_source_maps(peak_depo_date, peak_depo_maps,
                                      peak_depo_lats, peak_depo_lons,
                                      domain, 'deposition')
        if peak_conc_date:
            plot_ageclass_source_maps(peak_conc_date, peak_conc_maps,
                                      peak_conc_lats, peak_conc_lons,
                                      domain, 'concentration')

    print("\n" + "="*65)
    print(" ALL AGE-CLASS FIGURES COMPLETE")
    print(f" Saved in: {FIG_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
