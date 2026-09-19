#!/usr/bin/env python3
"""
=============================================================================
STEP 3: Compute BC Deposition at Khumbu Glacier from FLEXPART backward runs
        and EnsemFire wildfire BC emissions

Research: Black Carbon from wildfires → Khumbu Glacier, Nepal
          Pre-monsoon 2021 (March, April, May)

Method:   Eckhardt et al. (2017), GMD 10, 4605-4618
          D [kg m-2] = Σ_t Σ_x Σ_y [ ES(x,y,t) [m] × E_BC(x,y) [kg m-2 s-1] × dt [s] ]

          Where:
          - ES   = emission sensitivity (spec001_mr), summed over all 9 age classes
                   height index 0 (lowest level = 50m)
          - E_BC = EnsemFire BC flux regridded to FLEXPART grid [kg m-2 s-1]
          - dt   = 10800 s (3-hourly FLEXPART output step)

          Note: ES units in file are labeled "s m3 kg-1" but because
                IND_RECEPTOR=4 (dry) and IND_RECEPTOR=3 (wet), FLEXPART
                has multiplied by deposition velocity internally, making
                effective units = meters [m]. See Eckhardt et al. Table 1.

Outputs:
  - RESULTS/daily/     : daily dry, wet, total BC deposition NetCDF maps
  - RESULTS/monthly/   : monthly mean BC deposition maps
  - RESULTS/timeseries/: daily BC deposition time series CSV + NetCDF
  - RESULTS/source_contrib/: source contribution maps (spatial, per day)
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - adjust if needed
# =============================================================================

BASE_DIR = Path.home() / "Path.cwd()"

# FLEXPART output directories
FLEXPART_DIRS = {
    'dry': {
        'March': BASE_DIR / 'March_dry',
        'April': BASE_DIR / 'April_dry',
        'May':   BASE_DIR / 'May_dry',
    },
    'wet': {
        'March': BASE_DIR / 'March_wet',
        'April': BASE_DIR / 'April_wet',
        'May':   BASE_DIR / 'May_wet',
    }
}

# EnsemFire regridded BC directories
ENSEM_DIRS = {
    'coarse': BASE_DIR / 'ENSEM_REGRID/coarse_0.25deg',
    'nested': BASE_DIR / 'ENSEM_REGRID/nested_0.1deg',
}

# Output directory
OUT_DIR = BASE_DIR / 'RESULTS'

# Physics constants
DT = 10800.0        # seconds — FLEXPART 3-hourly output (LOUTSTEP)
H_SURF = 50.0       # meters  — lowest FLEXPART output level (height index 0)
                    # Used to convert surface flux [kg m-2 s-1] to
                    # volumetric emission [kg m-3 s-1] for ES multiplication

# Study period
MONTHS = {
    'March': ('20210301', '20210331'),
    'April': ('20210401', '20210430'),
    'May':   ('20210501', '20210531'),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_flexpart_file(run_type, date_str, domain='coarse'):
    """
    Find FLEXPART output file for a given run type, date, and domain.
    run_type: 'dry' or 'wet'
    date_str: 'YYYYMMDD'
    domain: 'coarse' (global) or 'nested'
    """
    month_num = int(date_str[4:6])
    month_name = {3: 'March', 4: 'April', 5: 'May'}[month_num]

    folder = FLEXPART_DIRS[run_type][month_name] / date_str

    if run_type == 'dry':
        suffix = 'drydep'
        if domain == 'nested':
            fname = folder / f"grid_drydep_{date_str}000000_nest.nc"
        else:
            fname = folder / f"grid_drydep_{date_str}000000.nc"
    else:
        suffix = 'wetdep'
        if domain == 'nested':
            fname = folder / f"grid_wetdep_{date_str}000000_nest.nc"
        else:
            fname = folder / f"grid_wetdep_{date_str}000000.nc"

    return fname if fname.exists() else None


def get_ensemfire_file(date_str, domain='coarse'):
    """Get regridded EnsemFire BC file for a given date."""
    if domain == 'coarse':
        fname = ENSEM_DIRS['coarse'] / f"EnsemFire_BC_coarse_{date_str}.nc"
    else:
        fname = ENSEM_DIRS['nested'] / f"EnsemFire_BC_nested_{date_str}.nc"
    return fname if fname.exists() else None


def load_emission_sensitivity(flexpart_file):
    """
    Load FLEXPART emission sensitivity field.
    Returns array summed over all age classes, at surface level (height idx 0).
    Shape: (n_timesteps, n_lat, n_lon)
    Units after this function: m (effective deposition sensitivity)
    """
    with nc.Dataset(flexpart_file) as f:
        # spec001_mr shape: (nageclass=9, pointspec=1, time=80, height=8, lat, lon)
        es = f.variables['spec001_mr'][:]  # masked array

        # Fill masked values with 0
        if hasattr(es, 'filled'):
            es = es.filled(0.0)

        # Sum over all 9 age classes (axis 0)
        es_summed = np.sum(es, axis=0)  # shape: (1, time, height, lat, lon)

        # Remove pointspec dimension
        es_summed = es_summed[0]  # shape: (time, height, lat, lon)

        # Take surface level (height index 0 = 50m)
        es_surf = es_summed[:, 0, :, :]  # shape: (time, lat, lon)

        # Get coordinates
        lats = f.variables['latitude'][:]
        lons = f.variables['longitude'][:]
        times = f.variables['time'][:]
        time_units = f.variables['time'].units

    return es_surf, lats, lons, times, time_units


def load_bc_emissions(ensemfire_file):
    """
    Load regridded EnsemFire BC emission flux.
    Returns 2D array (lat, lon) in kg m-2 s-1.
    EnsemFire is daily (1 time step), so we squeeze the time dimension.
    """
    with nc.Dataset(ensemfire_file) as f:
        bc = f.variables['BC'][:]  # shape: (1, lat, lon)
        if hasattr(bc, 'filled'):
            bc = bc.filled(0.0)
        bc = np.squeeze(bc)  # shape: (lat, lon)
        # Replace any negative fill values with 0
        bc = np.where(bc < 0, 0.0, bc)
        lats = f.variables['lat'][:]
        lons = f.variables['lon'][:]
    return bc, lats, lons


def compute_daily_deposition(es_surf, bc_flux):
    """
    Compute total BC deposition at Khumbu for one day.

    Formula (Eckhardt et al. 2017, Eq. relating to Table 1):
      D = Σ_t Σ_x Σ_y [ ES(x,y,t) [m] × (E_BC(x,y) [kg m-2 s-1] / H [m]) × dt [s] ]

    Dividing E_BC by H converts surface flux [kg m-2 s-1] to volumetric
    emission rate [kg m-3 s-1], which is what ES [m] × [kg m-3 s-1] × [s]
    needs to give [kg m-2].

    Returns:
      - depo_total [kg m-2]: scalar, total deposition at Khumbu
      - depo_map [lat, lon]: spatial source contribution map [kg m-2 per grid cell area]
    """
    # bc_flux shape: (lat, lon)
    # es_surf shape: (time, lat, lon)

    # Convert BC surface flux to volumetric emission rate
    bc_vol = bc_flux / H_SURF  # [kg m-3 s-1]

    # For each timestep: ES(t) × E_vol × dt
    # bc_vol is constant in time (daily EnsemFire), es_surf varies with t
    # depo_per_step shape: (time, lat, lon)
    depo_per_step = es_surf * bc_vol[np.newaxis, :, :] * DT  # [kg m-2]

    # Source contribution map: sum over time, keep spatial dims
    depo_map = np.sum(depo_per_step, axis=0)  # [kg m-2], shape: (lat, lon)

    # Total deposition at Khumbu: sum over all grid cells
    depo_total = np.sum(depo_map)  # scalar [kg m-2]

    return depo_total, depo_map


def save_daily_map(depo_map_dry, depo_map_wet, lats, lons, date_str, domain):
    """Save daily deposition map (dry + wet + total) as NetCDF."""
    out_dir = OUT_DIR / 'daily' / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"BC_deposition_{date_str}_{domain}.nc"

    depo_total = depo_map_dry + depo_map_wet

    ds = xr.Dataset({
        'BC_dry_deposition': xr.DataArray(
            depo_map_dry.astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
            attrs={'units': 'kg m-2', 'long_name': 'Daily BC dry deposition at Khumbu source regions',
                   'description': 'ES_dry x E_BC x dt, summed over 10-day backward trajectory'}
        ),
        'BC_wet_deposition': xr.DataArray(
            depo_map_wet.astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
            attrs={'units': 'kg m-2', 'long_name': 'Daily BC wet deposition at Khumbu source regions'}
        ),
        'BC_total_deposition': xr.DataArray(
            depo_total.astype(np.float32),
            dims=['latitude', 'longitude'],
            coords={'latitude': lats, 'longitude': lons},
            attrs={'units': 'kg m-2', 'long_name': 'Daily BC total deposition (dry + wet)'}
        ),
    }, attrs={
        'title': 'BC deposition source contribution map at Khumbu Glacier',
        'receptor': 'Khumbu Glacier, Nepal (86.80-86.85E, 27.94-27.97N)',
        'date': date_str,
        'method': 'Eckhardt et al. (2017) GMD 10, 4605-4618',
        'emissions': 'EnsemFire v1.0',
        'transport': 'FLEXPART v11.0 backward simulation with GFS 0.25deg',
        'domain': domain,
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M")} by compute_bc_deposition.py'
    })
    ds.to_netcdf(fname)
    return fname


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def main():
    print("=" * 65)
    print(" BC Deposition Computation — Khumbu Glacier, Nepal")
    print(" Pre-monsoon 2021 (March–May)")
    print(" Following: Eckhardt et al. (2017) GMD 10, 4605-4618")
    print("=" * 65)

    # Generate full date list for pre-monsoon 2021
    all_dates = []
    for d in pd.date_range('2021-03-01', '2021-05-31'):
        all_dates.append(d.strftime('%Y%m%d'))

    # Storage for time series
    records = []

    for domain in ['coarse', 'nested']:
        print(f"\n{'='*65}")
        print(f" Processing domain: {domain.upper()}")
        print(f"{'='*65}")

        domain_records = []
        skipped = 0
        processed = 0

        for date_str in all_dates:
            # Find required files
            dry_file  = get_flexpart_file('dry', date_str, domain)
            wet_file  = get_flexpart_file('wet', date_str, domain)
            ensem_file = get_ensemfire_file(date_str, domain)

            # Skip if any file missing
            missing = []
            if dry_file  is None: missing.append('dry')
            if wet_file  is None: missing.append('wet')
            if ensem_file is None: missing.append('ensemfire')

            if missing:
                skipped += 1
                print(f"  SKIP {date_str}: missing {', '.join(missing)}")
                domain_records.append({
                    'date': date_str,
                    'dry_total_kg_m2': np.nan,
                    'wet_total_kg_m2': np.nan,
                    'total_kg_m2': np.nan,
                    'status': f"missing_{'+'.join(missing)}"
                })
                continue

            try:
                # Load emission sensitivity (dry)
                es_dry, lats, lons, times, tunits = load_emission_sensitivity(dry_file)

                # Load emission sensitivity (wet)
                es_wet, _, _, _, _ = load_emission_sensitivity(wet_file)

                # Load BC emissions (same for both, daily)
                bc_flux, _, _ = load_bc_emissions(ensem_file)

                # Compute deposition
                dry_total, dry_map = compute_daily_deposition(es_dry, bc_flux)
                wet_total, wet_map = compute_daily_deposition(es_wet, bc_flux)
                total = dry_total + wet_total

                # Save daily spatial map
                save_daily_map(dry_map, wet_map, lats, lons, date_str, domain)

                processed += 1
                print(f"  ✅ {date_str}: dry={dry_total:.4e}  wet={wet_total:.4e}  "
                      f"total={total:.4e} kg m-2")

                domain_records.append({
                    'date': date_str,
                    'dry_total_kg_m2':   float(dry_total),
                    'wet_total_kg_m2':   float(wet_total),
                    'total_kg_m2':       float(total),
                    'status': 'ok'
                })

            except Exception as e:
                print(f"  ❌ {date_str}: ERROR — {e}")
                skipped += 1
                domain_records.append({
                    'date': date_str,
                    'dry_total_kg_m2': np.nan,
                    'wet_total_kg_m2': np.nan,
                    'total_kg_m2': np.nan,
                    'status': f'error: {str(e)}'
                })

        # Save time series CSV for this domain
        ts_dir = OUT_DIR / 'timeseries'
        ts_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(domain_records)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        csv_path = ts_dir / f"BC_deposition_timeseries_{domain}.csv"
        df.to_csv(csv_path, index=False)

        print(f"\n  Summary ({domain}): {processed} days processed, {skipped} skipped")
        print(f"  Time series saved: {csv_path}")

        records.append((domain, df))

    # -------------------------------------------------------------------------
    # Compute monthly means
    # -------------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(" Computing monthly mean deposition maps...")
    print(f"{'='*65}")

    for domain in ['coarse', 'nested']:
        daily_dir = OUT_DIR / 'daily' / domain
        monthly_dir = OUT_DIR / 'monthly' / domain
        monthly_dir.mkdir(parents=True, exist_ok=True)

        for month_name, (start, end) in MONTHS.items():
            month_files = sorted(daily_dir.glob(f"BC_deposition_{start[:6]}*_{domain}.nc"))

            if not month_files:
                print(f"  ⚠️  No files for {month_name} {domain}, skipping")
                continue

            datasets = [xr.open_dataset(f) for f in month_files]
            combined = xr.concat(datasets, dim='time')
            monthly_mean = combined.mean(dim='time')

            out_fname = monthly_dir / f"BC_deposition_monthly_{start[:6]}_{domain}.nc"
            monthly_mean.attrs['title'] = f"Monthly mean BC deposition — {month_name} 2021"
            monthly_mean.attrs['n_days'] = len(month_files)
            monthly_mean.to_netcdf(out_fname)

            # Close datasets
            for ds in datasets:
                ds.close()

            print(f"  ✅ {month_name} {domain}: {len(month_files)} days averaged → {out_fname.name}")

    print(f"\n{'='*65}")
    print(" ALL DONE!")
    print(f" Results saved in: {OUT_DIR}")
    print(f"   daily/      — daily deposition maps (dry, wet, total)")
    print(f"   monthly/    — monthly mean maps")
    print(f"   timeseries/ — daily time series CSV files")
    print(f"{'='*65}")
    print()
    print(" NEXT STEP: Run python3 04_plot_results.py to generate figures")


if __name__ == '__main__':
    main()
