#!/usr/bin/env python3
"""
=============================================================================
STEP 8: Regional Contribution Analysis — Nepal, IGP, Tibetan Plateau
        

        Uses the already-computed source maps from:
          RESULTS/daily/{domain}/BC_deposition_YYYYMMDD_{domain}.nc
          RESULTS/concentration/daily/{domain}/BC_concentration_YYYYMMDD_{domain}.nc

        Masks each source map with the three shapefiles:
          IGP.shp, Nepal.shp, Tibetan_Plateau.shp

        For each day and region, computes:
          Regional contribution [%] = sum(source_map inside region) / sum(source_map total)

Outputs:
  RESULTS/regional/
    BC_regional_deposition_{domain}.csv
    BC_regional_concentration_{domain}.csv
    Fig14_regional_contribution_timeseries.png
    Fig15_regional_contribution_bars.png
    Fig16_regional_contribution_seasonal_pies.png
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import shapely
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = Path.home() / "Path.cwd()"
DEPO_DIR    = BASE_DIR / "RESULTS" / "daily"
CONC_DIR    = BASE_DIR / "RESULTS" / "concentration" / "daily"
SHP_DIR     = BASE_DIR / "shpfiles"
OUT_DIR     = BASE_DIR / "RESULTS" / "regional"
FIG_DIR     = BASE_DIR / "RESULTS" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = ['IGP', 'Nepal', 'Tibetan_Plateau']
REGION_COLORS = {
    'IGP':              '#e74c3c',
    'Nepal':            '#2ecc71',
    'Tibetan_Plateau':  '#3498db',
    'Outside_all':      '#95a5a6',
}
REGION_LABELS = {
    'IGP':             'Indo-Gangetic Plain',
    'Nepal':           'Nepal',
    'Tibetan_Plateau': 'Tibetan Plateau',
    'Outside_all':     'Outside all regions',
}

MONTHS = {
    'March': ('2021-03-01','2021-03-31'),
    'April': ('2021-04-01','2021-04-30'),
    'May':   ('2021-05-01','2021-05-31'),
}

# =============================================================================
# BUILD REGION MASKS
# =============================================================================

def build_region_mask(lats, lons, shp_path):
    """
    Build a boolean 2D mask (lat × lon) for grid cells whose centres
    fall inside the shapefile polygon.
    Returns: 2D bool array, shape (nlat, nlon)
    """
    reader   = shpreader.Reader(str(shp_path))
    geometries = list(reader.geometries())
    union = unary_union(geometries)

    nlat, nlon = len(lats), len(lons)
    mask = np.zeros((nlat, nlon), dtype=bool)

    # Vectorised containment check using shapely prepared geometry
    prep = shapely.prepare(union)
    for j, lat in enumerate(lats):
        for i, lon in enumerate(lons):
            mask[j, i] = prep.contains(Point(lon, lat))

    return mask


def build_all_masks(lats, lons):
    """Build masks for all three regions. Returns dict of 2D bool arrays."""
    masks = {}
    for region in REGIONS:
        shp_path = SHP_DIR / f"{region}.shp"
        if not shp_path.exists():
            print(f"    Shapefile not found: {shp_path}")
            masks[region] = np.zeros((len(lats), len(lons)), dtype=bool)
        else:
            print(f"  Building mask for {region}...")
            masks[region] = build_region_mask(lats, lons, shp_path)
            n_cells = np.sum(masks[region])
            print(f"    → {n_cells} grid cells inside {region}")
    return masks


# =============================================================================
# COMPUTE REGIONAL CONTRIBUTIONS FOR ONE SOURCE MAP
# =============================================================================

def compute_regional_fractions(source_map, masks):
    """
    source_map: 2D array (lat, lon) — source contribution values
    masks: dict of 2D bool arrays

    Returns dict with:
      - absolute sum for each region
      - percentage of total for each region
      - 'Outside_all': fraction not in any of the three regions
    """
    total = np.nansum(source_map)
    if total <= 0:
        return {r: {'abs': 0.0, 'pct': 0.0} for r in REGIONS + ['Outside_all']}

    result = {}
    covered = np.zeros(source_map.shape, dtype=bool)

    for region in REGIONS:
        mask = masks[region]
        region_sum = np.nansum(source_map[mask])
        result[region] = {
            'abs': float(region_sum),
            'pct': float(region_sum / total * 100)
        }
        covered |= mask

    outside_sum = np.nansum(source_map[~covered])
    result['Outside_all'] = {
        'abs': float(outside_sum),
        'pct': float(outside_sum / total * 100)
    }

    return result


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def run_regional_analysis(domain='coarse'):
    print(f"\n{'='*60}")
    print(f" Regional Contribution Analysis — Domain: {domain.upper()}")
    print(f"{'='*60}")

    # --- Load all daily deposition files to get lat/lon once ---
    depo_files = sorted((DEPO_DIR / domain).glob(f"BC_deposition_2021*_{domain}.nc"))
    conc_files = sorted((CONC_DIR / domain).glob(f"BC_concentration_2021*_{domain}.nc"))

    if not depo_files and not conc_files:
        print(f"   No files found in {DEPO_DIR/domain}")
        return None, None

    # Get lat/lon from first file
    ref_file = depo_files[0] if depo_files else conc_files[0]
    ref_ds   = xr.open_dataset(ref_file)
    lats = ref_ds['latitude'].values
    lons = ref_ds['longitude'].values
    ref_ds.close()

    # Build masks ONCE (this is the slow step — ~1-2 min for coarse)
    print("\n  Building region masks (this takes 1-2 minutes)...")
    masks = build_all_masks(lats, lons)

    # --- DEPOSITION ---
    depo_records = []
    print(f"\n  Processing {len(depo_files)} deposition files...")
    for fpath in depo_files:
        date_str = fpath.name.split('_')[2]  # BC_deposition_YYYYMMDD_domain.nc
        ds = xr.open_dataset(fpath)
        source_map = ds['BC_total_deposition'].values
        ds.close()

        fracs = compute_regional_fractions(source_map, masks)
        total = np.nansum(source_map)

        row = {
            'date':         pd.to_datetime(date_str, format='%Y%m%d'),
            'total_kg_m2':  float(total),
            'total_ng_m2':  float(total * 1e12),
        }
        for region in REGIONS + ['Outside_all']:
            row[f'{region}_abs_kg_m2'] = fracs[region]['abs']
            row[f'{region}_pct']       = fracs[region]['pct']
        depo_records.append(row)

        if total > 0:
            igp_pct   = fracs['IGP']['pct']
            nepal_pct = fracs['Nepal']['pct']
            tp_pct    = fracs['Tibetan_Plateau']['pct']
            out_pct   = fracs['Outside_all']['pct']
            print(f"  {date_str}: total={total*1e12:.2e} ng m⁻² | "
                  f"IGP={igp_pct:.1f}% | Nepal={nepal_pct:.1f}% | "
                  f"TP={tp_pct:.1f}% | Outside={out_pct:.1f}%")

    df_depo = pd.DataFrame(depo_records).sort_values('date')
    csv_path = OUT_DIR / f"BC_regional_deposition_{domain}.csv"
    df_depo.to_csv(csv_path, index=False)
    print(f"   Deposition regional CSV saved: {csv_path.name}")

    # --- CONCENTRATION ---
    conc_records = []
    print(f"\n  Processing {len(conc_files)} concentration files...")
    for fpath in conc_files:
        date_str = fpath.name.split('_')[2]
        ds = xr.open_dataset(fpath)
        source_map = ds['BC_concentration'].values
        ds.close()

        fracs = compute_regional_fractions(source_map, masks)
        total = np.nansum(source_map)

        row = {
            'date':         pd.to_datetime(date_str, format='%Y%m%d'),
            'total_kg_m3':  float(total),
            'total_ng_m3':  float(total * 1e12),
        }
        for region in REGIONS + ['Outside_all']:
            row[f'{region}_abs_kg_m3'] = fracs[region]['abs']
            row[f'{region}_pct']       = fracs[region]['pct']
        conc_records.append(row)

    df_conc = pd.DataFrame(conc_records).sort_values('date')
    csv_path = OUT_DIR / f"BC_regional_concentration_{domain}.csv"
    df_conc.to_csv(csv_path, index=False)
    print(f"   Concentration regional CSV saved: {csv_path.name}")

    # Print summary
    print_regional_summary(df_depo, df_conc, domain)

    return df_depo, df_conc


# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

def print_regional_summary(df_depo, df_conc, domain):
    print(f"\n{'='*65}")
    print(f" REGIONAL CONTRIBUTION SUMMARY — {domain.upper()}")
    print(f"{'='*65}")
    print(f" {'Region':<22} {'March':>10} {'April':>10} {'May':>10} {'Season':>10}")
    print(f" {'-'*65}")

    periods = [
        ('March',  '2021-03-01','2021-03-31'),
        ('April',  '2021-04-01','2021-04-30'),
        ('May',    '2021-05-01','2021-05-31'),
        ('Season', '2021-03-01','2021-05-31'),
    ]

    print(" [DRY DEPOSITION]")
    for region in REGIONS + ['Outside_all']:
        row_str = f" {REGION_LABELS[region]:<22}"
        for pname, s, e in periods:
            mask = (df_depo['date'] >= s) & (df_depo['date'] <= e)
            sub = df_depo[mask]
            # weighted mean % (weighted by total deposition magnitude)
            if sub['total_ng_m2'].sum() > 0:
                w_mean_pct = (sub[f'{region}_abs_kg_m2'].sum() /
                              sub['total_kg_m2'].sum() * 100)
            else:
                w_mean_pct = 0.0
            row_str += f" {w_mean_pct:>9.1f}%"
        print(row_str)

    print(f"\n [CONCENTRATION]")
    for region in REGIONS + ['Outside_all']:
        row_str = f" {REGION_LABELS[region]:<22}"
        for pname, s, e in periods:
            mask = (df_conc['date'] >= s) & (df_conc['date'] <= e)
            sub = df_conc[mask]
            if sub['total_ng_m3'].sum() > 0:
                w_mean_pct = (sub[f'{region}_abs_kg_m3'].sum() /
                              sub['total_kg_m3'].sum() * 100)
            else:
                w_mean_pct = 0.0
            row_str += f" {w_mean_pct:>9.1f}%"
        print(row_str)
    print(f"{'='*65}")


# =============================================================================
# FIGURE 14: Regional Contribution Time Series
# =============================================================================

def plot_regional_timeseries(df_depo, df_conc, domain='coarse'):
    print("  Plotting Figure 14: Regional contribution time series...")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Regional BC Source Contribution to Khumbu Glacier\n'
                 f'Pre-monsoon 2021 | '
                 f'{"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
                 fontsize=13, fontweight='bold')

    for ax_idx, (df, mode, col_suffix, unit) in enumerate([
        (df_depo, 'Dry Deposition', '_abs_kg_m2', 'ng m⁻²'),
        (df_conc, 'Concentration',  '_abs_kg_m3', 'ng m⁻³'),
    ]):
        ax = axes[ax_idx]
        df_ok = df.sort_values('date').copy()

        # Stacked area chart showing absolute contribution by region
        regions_order = ['IGP', 'Nepal', 'Tibetan_Plateau', 'Outside_all']
        bottom = np.zeros(len(df_ok))
        scale  = 1e12

        for region in regions_order:
            col = f'{region}{col_suffix}'
            if col not in df_ok.columns:
                continue
            vals = df_ok[col].values * scale
            vals = np.where(vals < 0, 0, vals)
            ax.fill_between(df_ok['date'], bottom, bottom + vals,
                            alpha=0.8, color=REGION_COLORS[region],
                            label=REGION_LABELS[region])
            ax.plot(df_ok['date'], bottom + vals,
                    color=REGION_COLORS[region], linewidth=0.5, alpha=0.6)
            bottom += vals

        ax.set_ylabel(f'BC {mode} ({unit})', fontsize=10)
        ax.set_title(mode, fontsize=10, style='italic', loc='left', pad=4)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=9)
        for m_start in ['2021-04-01', '2021-05-01']:
            ax.axvline(pd.Timestamp(m_start), color='#aaaaaa',
                       linewidth=1, linestyle=':', zorder=1)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.85, ncol=2)

    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[1].set_xlabel('Date', fontsize=10)

    plt.tight_layout(rect=[0,0,1,0.96])
    out = FIG_DIR / f'Fig14_regional_timeseries_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"     Saved: {out.name}")


# =============================================================================
# FIGURE 15: Seasonal Pie Charts per Region
# =============================================================================

def plot_regional_pies(df_depo, df_conc, domain='coarse'):
    print("  Plotting Figure 15: Regional contribution pie charts...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Regional BC Source Contribution (%) to Khumbu Glacier\n'
                 f'Pre-monsoon 2021 | '
                 f'{"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
                 fontsize=13, fontweight='bold')

    periods = [
        ('March',    '2021-03-01','2021-03-31'),
        ('April',    '2021-04-01','2021-04-30'),
        ('May',      '2021-05-01','2021-05-31'),
        ('Mar–May',  '2021-03-01','2021-05-31'),
    ]
    regions_order = ['IGP','Nepal','Tibetan_Plateau','Outside_all']
    colors = [REGION_COLORS[r] for r in regions_order]
    labels = [REGION_LABELS[r] for r in regions_order]

    for row_idx, (df, col_suffix, row_label) in enumerate([
        (df_depo, '_abs_kg_m2', 'Dry Deposition'),
        (df_conc, '_abs_kg_m3', 'Concentration'),
    ]):
        for col_idx, (pname, s, e) in enumerate(periods):
            ax = axes[row_idx][col_idx]
            mask = (df['date'] >= s) & (df['date'] <= e)
            sub  = df[mask]

            vals = []
            for region in regions_order:
                col = f'{region}{col_suffix}'
                vals.append(sub[col].sum() if col in sub.columns else 0.0)

            total = sum(vals)
            if total <= 0:
                ax.text(0.5,0.5,'No data',ha='center',va='center',
                        transform=ax.transAxes)
                continue

            pcts   = [v/total*100 for v in vals]
            pie_labels = [f'{l}\n({p:.1f}%)' for l, p in zip(labels, pcts)]

            wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                               counterclock=False,
                               wedgeprops={'edgecolor':'white','linewidth':0.8})

            ax.set_title(f'{pname}', fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=70)

        # Legend on last column
        axes[row_idx][3].legend(wedges, pie_labels,
                                loc='center left',
                                bbox_to_anchor=(1.02, 0.5),
                                fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = FIG_DIR / f'Fig15_regional_pies_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"     Saved: {out.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print(" Regional Contribution Analysis — Khumbu Glacier")
    print(" Nepal vs IGP vs Tibetan Plateau")
    print("=" * 65)

    for domain in ['coarse', 'nested']:
        df_depo, df_conc = run_regional_analysis(domain)
        if df_depo is not None:
            print(f"\n  Generating figures for {domain}...")
            plot_regional_timeseries(df_depo, df_conc, domain)
            plot_regional_pies(df_depo, df_conc, domain)

    print("\n" + "="*65)
    print(" ALL REGIONAL FIGURES COMPLETE")
    print(f" Saved in: {FIG_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
