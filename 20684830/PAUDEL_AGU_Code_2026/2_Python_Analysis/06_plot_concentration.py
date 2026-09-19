#!/usr/bin/env python3
"""
=============================================================================
STEP 6: Concentration vs Deposition Timeline (Meteorology Style)
        Pre-monsoon 2021 (March, April, May) - Q1 JOURNAL EDITED VERSION

* STRIPPED DOWN: Generates ONLY the final dual-axis timeline.
* POLISHED: Z-order corrected, axes anchored to zero.
* PUBLICATION READY: Removed internal title and redundant legends.
=============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR    = Path.home() / "Path.cwd()"
RESULTS_DIR = BASE_DIR / "RESULTS"
CONC_DIR    = RESULTS_DIR / "concentration"
DEPO_DIR    = RESULTS_DIR            
FIG_DIR     = RESULTS_DIR / "Path.cwd()"
EDITED_FIG_DIR = FIG_DIR / "Path.cwd()"
EDITED_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Update global font parameters for publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2 # Thicker frame
})

# =============================================================================
# FIGURE GENERATOR
# =============================================================================

def plot_meteorology_timeline(domain='coarse'):
    print(f"  Plotting Final Publication Timeline ({domain})...")

    # Load and merge data
    csv_c = CONC_DIR / f"timeseries/BC_concentration_timeseries_{domain}.csv"
    df_c = pd.read_csv(csv_c, parse_dates=['date']).sort_values('date')

    csv_d = DEPO_DIR / f"timeseries/BC_deposition_timeseries_{domain}.csv"
    df_d = pd.read_csv(csv_d, parse_dates=['date']).sort_values('date')
    df_d = df_d[df_d['status'] == 'ok'].copy()

    df = pd.merge(df_c, df_d, on='date', how='inner')
    dates = df['date']
    conc = df['BC_conc_ng_m3']
    depo = df['total_kg_m2'] * 1e12  # Convert kg/m2 to ng/m2

    # Styling colors
    color_c = '#C0392B' # Deep Red
    color_d_fill = '#AED6F1' # Light Blue Fill
    color_d_edge = '#2874A6' # Dark Blue Edge for crispness

    # Create figure (slightly shorter since we removed the title)
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    ax2 = ax1.twinx()
    
    # --- PLOT BARS FIRST (ax2) ---
    ax2.bar(dates, depo, color=color_d_fill, edgecolor=color_d_edge, linewidth=0.8, 
            alpha=0.85, width=0.8)
    ax2.set_ylabel('Surface Deposition Flux (ng m⁻²)', color=color_d_edge, fontweight='bold', labelpad=12)
    ax2.tick_params(axis='y', colors=color_d_edge, width=1.2)
    ax2.spines['right'].set_color(color_d_edge)
    ax2.spines['right'].set_linewidth(1.5)

    # --- PLOT LINE OVER TOP (ax1) ---
    ax1.plot(dates, conc, color=color_c, linewidth=2.5)
    ax1.set_ylabel('Atmospheric Concentration (ng m⁻³)', color=color_c, fontweight='bold', labelpad=12)
    ax1.tick_params(axis='y', colors=color_c, width=1.2)
    ax1.spines['left'].set_color(color_c)
    ax1.spines['left'].set_linewidth(1.5)
    
    # ==========================================
    # PUBLICATION POLISH TWEAKS
    # ==========================================
    
    # 1. Fix the Z-order: Bring ax1 (the line) to the front and make its background transparent
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # 2. Fix the floating zero and add 15% headroom to the top of the axes
    ax1.set_ylim(0, conc.max() * 1.15)
    ax2.set_ylim(0, depo.max() * 1.15)
    
    # ==========================================
    
    # --- X-AXIS FORMATTING ---
    date_loc = mdates.WeekdayLocator(byweekday=0) # Ticks every Monday
    date_fmt = mdates.DateFormatter('%d %b')
    ax1.xaxis.set_major_locator(date_loc)
    ax1.xaxis.set_major_formatter(date_fmt)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    
    # Clean grid
    ax1.grid(True, axis='x', color='grey', alpha=0.3, linestyle='--')
    ax1.grid(True, axis='y', color='grey', alpha=0.15, linestyle='-')
    
    # Save
    out = EDITED_FIG_DIR / f'Fig10_concentration_vs_deposition_{domain}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    ✅ Saved Publication Figure: {out.name}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print(" Generating Publication-Ready Timeline Figures")
    print(f" Output Directory: {EDITED_FIG_DIR}")
    print("=" * 65)
    print()

    for domain in ['coarse', 'nested']:
        plot_meteorology_timeline(domain)

    print()
    print("=" * 65)
    print(" COMPLETE")
    print("=" * 65)
