#!/usr/bin/env python3
"""
STEP 8: Regional Contribution Analysis — Nepal, IGP, Tibetan Plateau
Fixed version using geopandas sjoin
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from shapely.geometry import Point
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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
    'IGP':             '#e74c3c',
    'Nepal':           '#2ecc71',
    'Tibetan_Plateau': '#3498db',
    'Outside_all':     '#95a5a6',
}
REGION_LABELS = {
    'IGP':             'Indo-Gangetic Plain',
    'Nepal':           'Nepal',
    'Tibetan_Plateau': 'Tibetan Plateau',
    'Outside_all':     'Outside all regions',
}
MONTHS = {
    'March': ('2021-03-01', '2021-03-31'),
    'April': ('2021-04-01', '2021-04-30'),
    'May':   ('2021-05-01', '2021-05-31'),
}

# ─────────────────────────────────────────────
# MASK BUILDING — geopandas sjoin (fast)
# ─────────────────────────────────────────────

def build_region_mask(lats, lons, shp_path):
    gdf_region = gpd.read_file(str(shp_path))
    if gdf_region.crs is None:
        gdf_region = gdf_region.set_crs('EPSG:4326')
    else:
        gdf_region = gdf_region.to_crs('EPSG:4326')

    nlat, nlon = len(lats), len(lons)
    lon2d, lat2d = np.meshgrid(lons, lats)

    pts = gpd.GeoDataFrame(
        {'idx': np.arange(nlat * nlon)},
        geometry=[Point(float(x), float(y))
                  for x, y in zip(lon2d.ravel(), lat2d.ravel())],
        crs='EPSG:4326'
    )

    joined = gpd.sjoin(pts, gdf_region[['geometry']],
                       how='left', predicate='within')
    inside = ~joined['index_right'].isna()
    return inside.values.reshape(nlat, nlon)


def build_all_masks(lats, lons):
    masks = {}
    for region in REGIONS:
        shp_path = SHP_DIR / f"{region}.shp"
        if not shp_path.exists():
            print(f"  WARNING: {shp_path} not found — region set to zero")
            masks[region] = np.zeros((len(lats), len(lons)), dtype=bool)
        else:
            print(f"  Building mask: {region} ...", end=' ', flush=True)
            masks[region] = build_region_mask(lats, lons, shp_path)
            print(f"{int(np.sum(masks[region]))} cells")
    return masks


# ─────────────────────────────────────────────
# REGIONAL FRACTIONS
# ─────────────────────────────────────────────

def compute_regional_fractions(source_map, masks):
    total = float(np.nansum(source_map))
    if total <= 0:
        return {r: {'abs': 0.0, 'pct': 0.0} for r in REGIONS + ['Outside_all']}
    result  = {}
    covered = np.zeros(source_map.shape, dtype=bool)
    for region in REGIONS:
        s = float(np.nansum(source_map[masks[region]]))
        result[region] = {'abs': s, 'pct': s / total * 100}
        covered |= masks[region]
    s = float(np.nansum(source_map[~covered]))
    result['Outside_all'] = {'abs': s, 'pct': s / total * 100}
    return result


# ─────────────────────────────────────────────
# PROCESS FILES
# ─────────────────────────────────────────────

def process_files(files, masks, var_hint, scale_label):
    records = []
    for fpath in sorted(files):
        parts    = fpath.stem.split('_')
        date_str = parts[2]
        ds = xr.open_dataset(fpath)
        # pick first data variable
        varname = list(ds.data_vars)[0]
        source_map = np.squeeze(ds[varname].values)
        ds.close()

        fracs = compute_regional_fractions(source_map, masks)
        total = float(np.nansum(source_map))
        row   = {'date': pd.to_datetime(date_str, format='%Y%m%d'),
                 f'total_{var_hint}': total,
                 f'total_{var_hint}_scaled': total * 1e12}
        for region in REGIONS + ['Outside_all']:
            row[f'{region}_abs'] = fracs[region]['abs']
            row[f'{region}_pct'] = fracs[region]['pct']
        records.append(row)

        if total > 0:
            print(f"  {date_str}: total={total*1e12:9.2f} {scale_label} | "
                  f"IGP={fracs['IGP']['pct']:5.1f}% | "
                  f"Nepal={fracs['Nepal']['pct']:5.1f}% | "
                  f"TP={fracs['Tibetan_Plateau']['pct']:4.1f}% | "
                  f"Outside={fracs['Outside_all']['pct']:5.1f}%")

    return pd.DataFrame(records).sort_values('date')


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def print_summary(df, label, domain):
    periods = [
        ('March',  '2021-03-01', '2021-03-31'),
        ('April',  '2021-04-01', '2021-04-30'),
        ('May',    '2021-05-01', '2021-05-31'),
        ('Season', '2021-03-01', '2021-05-31'),
    ]
    print(f"\n{'='*65}")
    print(f" {label} — {domain.upper()}")
    print(f"{'='*65}")
    print(f"  {'Region':<26} {'March':>9} {'April':>9} {'May':>8} {'Season':>9}")
    print(f"  {'-'*63}")
    for region in REGIONS + ['Outside_all']:
        row_str = f"  {REGION_LABELS[region]:<26}"
        total_col = [c for c in df.columns if c.startswith('total_') and not c.endswith('_scaled')][0]
        for _, s, e in periods:
            mask = (df['date'] >= s) & (df['date'] <= e)
            sub  = df[mask]
            grand = sub[total_col].sum()
            pct = sub[f'{region}_abs'].sum() / grand * 100 if grand > 0 else 0.0
            row_str += f" {pct:8.1f}%"
        print(row_str)
    print(f"{'='*65}")


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

def plot_pies(df_depo, df_conc, domain):
    periods = [
        ('March',   '2021-03-01', '2021-03-31'),
        ('April',   '2021-04-01', '2021-04-30'),
        ('May',     '2021-05-01', '2021-05-31'),
        ('Mar-May', '2021-03-01', '2021-05-31'),
    ]
    regions_order = ['IGP', 'Nepal', 'Tibetan_Plateau', 'Outside_all']
    colors  = [REGION_COLORS[r] for r in regions_order]
    labels  = [REGION_LABELS[r] for r in regions_order]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f'Regional BC Source Contribution (%) to Khumbu Glacier\n'
        f'Pre-monsoon 2021 | {"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
        fontsize=13, fontweight='bold')

    for row_idx, (df, total_col) in enumerate([
        (df_depo, [c for c in df_depo.columns if c.startswith('total_') and not c.endswith('_scaled')][0]),
        (df_conc, [c for c in df_conc.columns if c.startswith('total_') and not c.endswith('_scaled')][0]),
    ]):
        row_label = 'Dry Deposition' if row_idx == 0 else 'Concentration'
        for col_idx, (pname, s, e) in enumerate(periods):
            ax   = axes[row_idx][col_idx]
            mask = (df['date'] >= s) & (df['date'] <= e)
            sub  = df[mask]
            vals = [sub[f'{r}_abs'].sum() for r in regions_order]
            total = sum(vals)
            if total <= 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(pname)
                continue
            pcts = [v / total * 100 for v in vals]
            pie_labels = [f'{l}\n({p:.1f}%)' for l, p in zip(labels, pcts)]
            wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                               counterclock=False,
                               wedgeprops={'edgecolor': 'white', 'linewidth': 0.8})
            ax.set_title(f'{pname}', fontsize=10, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, labelpad=70)
        axes[row_idx][3].legend(
            wedges, pie_labels,
            loc='center left', bbox_to_anchor=(1.02, 0.5),
            fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = FIG_DIR / f'Fig15_regional_pies_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out.name}")


def plot_timeseries(df_depo, df_conc, domain):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(
        f'Regional BC Source Contribution to Khumbu Glacier\n'
        f'Pre-monsoon 2021 | {"Coarse 0.25°" if domain=="coarse" else "Nested 0.1°"}',
        fontsize=13, fontweight='bold')

    for ax_idx, (df, row_label, scale, unit) in enumerate([
        (df_depo, 'Dry Deposition', 1e12, 'ng m⁻²'),
        (df_conc, 'Concentration',  1e12, 'ng m⁻³'),
    ]):
        ax = axes[ax_idx]
        df_s = df.sort_values('date').copy()
        bottom = np.zeros(len(df_s))
        for region in ['IGP', 'Nepal', 'Tibetan_Plateau', 'Outside_all']:
            col  = f'{region}_abs'
            if col not in df_s.columns:
                continue
            vals = df_s[col].values * scale
            vals = np.where(vals < 0, 0, vals)
            ax.fill_between(df_s['date'], bottom, bottom + vals,
                            alpha=0.8, color=REGION_COLORS[region],
                            label=REGION_LABELS[region])
            bottom += vals

        ax.set_ylabel(f'BC {row_label} ({unit})', fontsize=10)
        ax.set_title(row_label, fontsize=10, style='italic', loc='left', pad=4)
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        for m in ['2021-04-01', '2021-05-01']:
            ax.axvline(pd.Timestamp(m), color='#aaaaaa', linewidth=1,
                       linestyle=':', zorder=1)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.85, ncol=2)

    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    axes[1].set_xlabel('Date', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / f'Fig14_regional_timeseries_{domain}.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_domain(domain):
    print(f"\n{'='*60}")
    print(f" Domain: {domain.upper()}")
    print(f"{'='*60}")

    depo_files = sorted((DEPO_DIR / domain).glob(f"BC_deposition_2021*_{domain}.nc"))
    conc_files = sorted((CONC_DIR / domain).glob(f"BC_concentration_2021*_{domain}.nc"))
    print(f"  Found {len(depo_files)} deposition files, {len(conc_files)} concentration files")

    if not depo_files and not conc_files:
        print("  No files found — skipping.")
        return

    ref_ds = xr.open_dataset(depo_files[0] if depo_files else conc_files[0])
    lats   = ref_ds['latitude'].values
    lons   = ref_ds['longitude'].values
    ref_ds.close()
    print(f"  Grid: {len(lats)} lat × {len(lons)} lon")

    masks = build_all_masks(lats, lons)

    if depo_files:
        print(f"\n  --- DEPOSITION ---")
        df_depo = process_files(depo_files, masks, 'kg_m2', 'ng m⁻²')
        df_depo.to_csv(OUT_DIR / f"BC_regional_deposition_{domain}.csv", index=False)
        print_summary(df_depo, 'DRY DEPOSITION', domain)
    else:
        df_depo = pd.DataFrame()

    if conc_files:
        print(f"\n  --- CONCENTRATION ---")
        df_conc = process_files(conc_files, masks, 'kg_m3', 'ng m⁻³')
        df_conc.to_csv(OUT_DIR / f"BC_regional_concentration_{domain}.csv", index=False)
        print_summary(df_conc, 'CONCENTRATION', domain)
    else:
        df_conc = pd.DataFrame()

    if not df_depo.empty and not df_conc.empty:
        print(f"\n  Generating figures...")
        plot_timeseries(df_depo, df_conc, domain)
        plot_pies(df_depo, df_conc, domain)


def main():
    print("=" * 65)
    print(" Regional Contribution Analysis — Khumbu Glacier")
    print(" Nepal  |  IGP  |  Tibetan Plateau  |  Outside")
    print("=" * 65)
    for domain in ['coarse', 'nested']:
        run_domain(domain)
    print("\n" + "=" * 65)
    print(" DONE — figures saved in RESULTS/figures/")
    print("=" * 65)


if __name__ == '__main__':
    main()
