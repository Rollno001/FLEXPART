#!/usr/bin/env python3
"""
=============================================================================
STEP 5: Compute BC Concentration at Khumbu Glacier from FLEXPART backward
        concentration runs (IND_RECEPTOR=1) and EnsemFire wildfire BC emissions

Research: Black Carbon from wildfires → Khumbu Glacier, Nepal
          Pre-monsoon 2021 (March, April, May)

Method:   Stohl et al. (2005) + Eckhardt et al. (2017) Table 1
          For IND_SOURCE=1, IND_RECEPTOR=1 backward mode:
          C [kg m⁻³] = Σ_t Σ_x Σ_y [ ES(x,y,t) [s] × E_BC(x,y) [kg m⁻² s⁻¹] / H [m] ]

          Where:
          - ES   = emission sensitivity (spec001_mr, units: s)
                   summed over all 9 age classes, at height index 0 (surface)
          - E_BC = EnsemFire BC flux regridded to FLEXPART grid [kg m⁻² s⁻¹]
          - H    = 50m (lowest FLEXPART output level, converts surface flux to
                   volumetric emission rate [kg m⁻³ s⁻¹])
          - No dt needed: ES units [s] × E_vol [kg m⁻³ s⁻¹] = C [kg m⁻³] directly

          Note on release height: particles released 10–500m AGL
          (Z1=10, Z2=500 in your concentration run script), so surface
          level sensitivity (50m) is appropriate.

Final units: ng m⁻³ (multiply kg m⁻³ × 1e12)
Comparable to: NCO-P Pyramid Station measurements (5079m, ~10km from Khumbu)

Outputs:
  - RESULTS/concentration/daily/     : daily BC concentration maps (coarse + nested)
  - RESULTS/concentration/monthly/   : monthly mean BC concentration maps
  - RESULTS/concentration/timeseries/: daily time series CSV
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path.home() / "Path.cwd()"

# CONCENTRATION file directories (different structure from deposition runs)
CONC_DIRS = {
    'March': BASE_DIR / 'CONCENTRATION/gridtime_all_march',
    'April': BASE_DIR / 'CONCENTRATION/gridtime_all_april',
    'May':   BASE_DIR / 'CONCENTRATION/gridtime_all_may',
}

# EnsemFire regridded BC directories (already created in Step 2)
ENSEM_DIRS = {
    'coarse': BASE_DIR / 'ENSEM_REGRID/coarse_0.25deg',
    'nested': BASE_DIR / 'ENSEM_REGRID/nested_0.1deg',
}

# Output directory
OUT_DIR = BASE_DIR / 'RESULTS' / 'concentration'

# Physics
H_SURF = 50.0   # meters — lowest FLEXPART output level

# Study months
MONTHS = {
    'March': ('20210301', '20210331'),
    'April': ('20210401', '20210430'),
    'May':   ('20210501', '20210531'),
}

# =============================================================================
# HELPERS
# =============================================================================

def get_conc_file(date_str, domain='coarse'):
    """
    Find concentration grid_time file for a given date and domain.
    Files are named: grid_time_YYYYMMDD000000.nc  (coarse)
                     grid_time_YYYYMMDD000000_nest.nc  (nested)
    """
    month_num = int(date_str[4:6])
    month_name = {3: 'March', 4: 'April', 5: 'May'}[month_num]
    folder = CONC_DIRS[month_name]

    if domain == 'nested':
        fname = folder / f"grid_time_{date_str}000000_nest.nc"
    else:
        fname = folder / f"grid_time_{date_str}000000.nc"

    return fname if fname.exists() else None


def get_ensemfire_file(date_str, domain='coarse'):
    """Get regridded EnsemFire BC file."""
    if domain == 'coarse':
        fname = ENSEM_DIRS['coarse'] / f"EnsemFire_BC_coarse_{date_str}.nc"
    else:
        fname = ENSEM_DIRS['nested'] / f"EnsemFire_BC_nested_{date_str}.nc"
    return fname if fname.exists() else None


def load_concentration_sensitivity(conc_file):
    """
    Load FLEXPART concentration emission sensitivity (IND_RECEPTOR=1).
    units: 's'
    Returns ES summed over all age classes, at surface level (height idx 0).
    Shape: (n_timesteps, n_lat, n_lon)
    """
    with nc.Dataset(conc_file) as f:
        # spec001_mr shape: (nageclass=9, pointspec=1, time=80, height=8, lat, lon)
        es = f.variables['spec001_mr'][:]
        if hasattr(es, 'filled'):
            es = es.filled(0.0)

        # Sum over all 9 age classes (axis 0)
        es_summed = np.sum(es, axis=0)   # shape: (1, time, height, lat, lon)
        es_summed = es_summed[0]          # shape: (time, height, lat, lon)

        # Take surface level (height index 0 = 50m)
        es_surf = es_summed[:, 0, :, :]  # shape: (time, lat, lon)

        lats = f.variables['latitude'][:]
        lons = f.variables['longitude'][:]

    return es_surf, lats, lons


def load_bc_emissions(ensemfire_file):
    """Load regridded EnsemFire BC emission flux [kg m⁻² s⁻¹]."""
    with nc.Dataset(ensemfire_file) as f:
        bc = f.variables['BC'][:]
        if hasattr(bc, 'filled'):
            bc = bc.filled(0.0)
        bc = np.squeeze(bc)               # shape: (lat, lon)
        bc = np.where(bc < 0, 0.0, bc)
        lats = f.variables['lat'][:]
        lons = f.variables['lon'][:]
    return bc, lats, lons


def compute_daily_concentration(es_surf, bc_flux):
    """
    Compute BC concentration at Khumbu for one day.

    Formula:
      C [kg m⁻³] = Σ_t Σ_x Σ_y [ ES(x,y,t) [s] × (E_BC(x,y) / H) [kg m⁻³ s⁻¹] ]

    Returns:
      conc_total [kg m⁻³]: scalar total BC concentration
      conc_map   [lat,lon]: spatial source contribution map [kg m⁻³]
    """
    # Convert BC surface flux to volumetric emission rate [kg m⁻³ s⁻¹]
    bc_vol = bc_flux / H_SURF

    # For each timestep: ES(t) [s] × E_vol [kg m⁻³ s⁻¹] → [kg m⁻³]
    conc_per_step = es_surf * bc_vol[np.newaxis, :, :]  # shape: (time, lat, lon)

    # Sum over all timesteps → source contribution map
    conc_map = np.sum(conc_per_step, axis=0)  # shape: (lat, lon) [kg m⁻³]

    # Total concentration: sum over all source grid cells
    conc_total = np.sum(conc_map)  # scalar [kg m⁻³]

    return conc_total, conc_map


def save_daily_conc_map(conc_map, lats, lons, date_str, domain):
    """Save daily BC concentration source map as NetCDF."""
    out_dir = OUT_DIR / 'daily' / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"BC_concentration_{date_str}_{domain}.nc"

    ds = xr.Dataset({
        'BC_concentration': xr.DataArray(
            conc_map.astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
            attrs={
                'units': 'kg m-3',
                'long_name': 'BC concentration source contribution at Khumbu',
                'description': 'ES_conc [s] x E_BC [kg m-3 s-1], summed over 10-day backward trajectory',
                'note': 'Multiply by 1e12 to convert to ng m-3'
            }
        ),
    }, attrs={
        'title': 'BC concentration source contribution at Khumbu Glacier',
        'receptor': 'Khumbu Glacier, Nepal (86.80-86.85E, 27.94-27.97N)',
        'date': date_str,
        'method': 'Stohl et al. (2005) + Eckhardt et al. (2017) — IND_RECEPTOR=1',
        'emissions': 'EnsemFire v1.0',
        'transport': 'FLEXPART v11.0 backward simulation with GFS 0.25deg',
        'domain': domain,
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M")}'
    })
    ds.to_netcdf(fname)
    return fname


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print(" BC Concentration Computation — Khumbu Glacier, Nepal")
    print(" Pre-monsoon 2021 (March–May)")
    print(" IND_RECEPTOR=1 | Units: ng m⁻³")
    print("=" * 65)

    all_dates = []
    for d in pd.date_range('2021-03-01', '2021-05-31'):
        all_dates.append(d.strftime('%Y%m%d'))

    for domain in ['coarse', 'nested']:
        print(f"\n{'='*65}")
        print(f" Domain: {domain.upper()}")
        print(f"{'='*65}")

        records = []
        processed = 0
        skipped = 0

        for date_str in all_dates:
            conc_file  = get_conc_file(date_str, domain)
            ensem_file = get_ensemfire_file(date_str, domain)

            missing = []
            if conc_file  is None: missing.append('concentration')
            if ensem_file is None: missing.append('ensemfire')

            if missing:
                skipped += 1
                print(f"  SKIP {date_str}: missing {', '.join(missing)}")
                records.append({
                    'date': date_str,
                    'BC_conc_kg_m3': np.nan,
                    'BC_conc_ng_m3': np.nan,
                    'status': f"missing_{'+'.join(missing)}"
                })
                continue

            try:
                es_surf, lats, lons = load_concentration_sensitivity(conc_file)
                bc_flux, _, _       = load_bc_emissions(ensem_file)

                conc_total, conc_map = compute_daily_concentration(es_surf, bc_flux)

                conc_ng = conc_total * 1e12  # convert to ng m⁻³

                save_daily_conc_map(conc_map, lats, lons, date_str, domain)

                processed += 1
                print(f"  ✅ {date_str}: C = {conc_ng:.4f} ng m⁻³  ({conc_total:.4e} kg m⁻³)")

                records.append({
                    'date': date_str,
                    'BC_conc_kg_m3': float(conc_total),
                    'BC_conc_ng_m3': float(conc_ng),
                    'status': 'ok'
                })

            except Exception as e:
                print(f"  ❌ {date_str}: ERROR — {e}")
                skipped += 1
                records.append({
                    'date': date_str,
                    'BC_conc_kg_m3': np.nan,
                    'BC_conc_ng_m3': np.nan,
                    'status': f'error: {str(e)}'
                })

        # Save time series CSV
        ts_dir = OUT_DIR / 'timeseries'
        ts_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        csv_path = ts_dir / f"BC_concentration_timeseries_{domain}.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n  Summary ({domain}): {processed} processed, {skipped} skipped")
        print(f"  Saved: {csv_path}")

        # Print quick statistics
        ok = df[df['status'] == 'ok']
        if len(ok) > 0:
            print(f"\n  === Statistics (ng m⁻³) ===")
            print(f"  Mean  : {ok['BC_conc_ng_m3'].mean():.4f}")
            print(f"  Median: {ok['BC_conc_ng_m3'].median():.4f}")
            print(f"  Max   : {ok['BC_conc_ng_m3'].max():.4f}  "
                  f"on {ok.loc[ok['BC_conc_ng_m3'].idxmax(), 'date'].strftime('%d %B %Y')}")
            print(f"  Min   : {ok['BC_conc_ng_m3'].min():.6f}")
            for month, (s, e) in MONTHS.items():
                mask = (ok['date'] >= s) & (ok['date'] <= e)
                sub = ok[mask]
                if len(sub) > 0:
                    print(f"  {month} mean: {sub['BC_conc_ng_m3'].mean():.4f} ng m⁻³")

    # -------------------------------------------------------------------------
    # Monthly means
    # -------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(" Computing monthly mean concentration maps...")

    for domain in ['coarse', 'nested']:
        daily_dir = OUT_DIR / 'daily' / domain
        monthly_dir = OUT_DIR / 'monthly' / domain
        monthly_dir.mkdir(parents=True, exist_ok=True)

        for month_name, (start, end) in MONTHS.items():
            month_files = sorted(daily_dir.glob(f"BC_concentration_{start[:6]}*_{domain}.nc"))
            if not month_files:
                print(f"  ⚠️  No files for {month_name} {domain}")
                continue

            datasets = [xr.open_dataset(f) for f in month_files]
            combined = xr.concat(datasets, dim='time')
            monthly_mean = combined.mean(dim='time')
            monthly_mean.attrs['title'] = f"Monthly mean BC concentration — {month_name} 2021"
            monthly_mean.attrs['n_days'] = len(month_files)

            out_fname = monthly_dir / f"BC_concentration_monthly_{start[:6]}_{domain}.nc"
            monthly_mean.to_netcdf(out_fname)
            for ds in datasets:
                ds.close()
            print(f"  ✅ {month_name} {domain}: {len(month_files)} days → {out_fname.name}")

    print(f"\n{'='*65}")
    print(" ALL DONE!")
    print(f" Results: {OUT_DIR}")
    print(f"   daily/      — daily BC concentration source maps")
    print(f"   monthly/    — monthly mean maps")
    print(f"   timeseries/ — daily time series CSV (ng m⁻³)")
    print()
    print(" NEXT: Run python3 06_plot_concentration.py for concentration figures")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
