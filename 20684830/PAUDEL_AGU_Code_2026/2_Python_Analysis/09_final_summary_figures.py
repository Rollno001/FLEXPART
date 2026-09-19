#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try importing scipy safely
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except:
    print(" scipy not found → KDE will fallback to line plot")
    SCIPY_AVAILABLE = False

# =============================================================================
# CONFIG
# =============================================================================

BASE    = Path.home() / "Downloads/FOR_REGRID/Claude_AI"
RES     = BASE / "RESULTS"
FIG     = RES / "figures"
EDITED_FIG_DIR = FIG / "edited_figures"
EDITED_FIG_DIR.mkdir(parents=True, exist_ok=True)

AGE_LABELS = ['6hr','12hr','1day','2day','3day','4day','5day','7day','10day']
DOMAIN     = 'coarse'

def load_csv(path):
    print(f"  Loading: {path}")
    df = pd.read_csv(path, parse_dates=['date'])
    return df.sort_values('date').reset_index(drop=True)

# =============================================================================
# FIG 19 (Q1 FINAL POLISHED)
# =============================================================================

def fig19_transport_age():
    print("  Plotting Fig19: Heatmap & Age Distribution (Final)...")

    df_age = load_csv(RES / f"ageclass/timeseries/BC_ageclass_deposition_{DOMAIN}.csv")

    periods = [('March', '2021-03-01', '2021-03-31'),
               ('April', '2021-04-01', '2021-04-30'),
               ('May',   '2021-05-01', '2021-05-31')]

    age_numeric = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 7, 10])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ============================================================
    # PANEL A: HEATMAP
    # ============================================================
    ax1 = axes[0]

    matrix = np.zeros((len(periods), len(AGE_LABELS)))

    for i, (_, s, e) in enumerate(periods):
        mask = (df_age['date'] >= s) & (df_age['date'] <= e)
        sub = df_age[mask]
        totals = []
        for a in range(9):
            col = f'depo_age{a}_{AGE_LABELS[a]}_kg_m2'
            totals.append(sub[col].sum() if col in sub.columns else 0)

        totals = np.array(totals)
        grand = totals.sum()
        if grand > 0:
            matrix[i, :] = totals / grand * 100

    # Draw the Heatmap
    im = ax1.imshow(matrix, cmap='YlOrBr', aspect='auto', vmin=0, vmax=np.max(matrix))

    ax1.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
    ax1.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
    ax1.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax1.tick_params(which="minor", bottom=False, left=False)

    ax1.set_xticks(np.arange(len(AGE_LABELS)))
    ax1.set_xticklabels(AGE_LABELS, rotation=45, ha='right', fontweight='bold', fontsize=11)
    
    ax1.set_yticks(np.arange(len(periods)))
    ax1.set_yticklabels([p[0] for p in periods], rotation=90, va='center', fontweight='bold', fontsize=12)

    ax1.spines[:].set_visible(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i,j]
            if val > 1.0: 
                text_color = "white" if val > (np.max(matrix) * 0.6) else "#333333"
                ax1.text(j, i, f"{val:.1f}%", ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)

    # Add (a) tag instead of a title
    ax1.text(-0.05, 1.05, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax1)
    cbar_ax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Age-Class Contribution (%)', fontweight='bold', labelpad=10, fontsize=11)
    cbar.outline.set_visible(False)

    # ============================================================
    # PANEL B: KDE + CUMULATIVE
    # ============================================================
    ax2 = axes[1]

    mask_all = (df_age['date'] >= '2021-03-01') & (df_age['date'] <= '2021-05-31')
    sub_all  = df_age[mask_all]

    totals = np.zeros(9)
    for a in range(9):
        col = f'depo_age{a}_{AGE_LABELS[a]}_kg_m2'
        if col in sub_all.columns:
            totals[a] = sub_all[col].sum()

    grand = totals.sum()
    if grand == 0: return
    weights = totals / grand

    if SCIPY_AVAILABLE:
        kde = gaussian_kde(age_numeric, weights=weights)
        x_vals = np.logspace(np.log10(0.2), np.log10(11), 300) 
        y_vals = np.maximum(kde(x_vals), 0)

        ax2.plot(x_vals, y_vals, color='#D35400', lw=2.5, label='Probability Density')
        ax2.fill_between(x_vals, 0, y_vals, color='#E67E22', alpha=0.3)

    # Removed [Log Scale] from label
    ax2.set_xlabel('Transport Age (Days)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Probability Density', fontweight='bold', color='#D35400', fontsize=12)

    ax2.set_xscale('log')
    ax2.set_xlim(0.2, 11)
    ax2.set_xticks([0.25, 0.5, 1, 2, 5, 10])
    
    # Custom formatter to remove trailing zeros (e.g., 1.00 becomes 1)
    from matplotlib.ticker import FuncFormatter
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(True, which='major', alpha=0.3, linestyle='--')

    # Add (b) tag instead of a title
    ax2.text(-0.05, 1.05, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')

    # -------------------------------
    # CUMULATIVE LINE
    # -------------------------------
    ax2b = ax2.twinx()
    cumulative = np.cumsum(weights * 100)

    ax2b.plot(age_numeric, cumulative, color='#1a1a2e', lw=2, marker='o', ms=6, label='Cumulative %')

    ax2b.set_ylabel('Cumulative Contribution (%)', fontweight='bold', color='#1a1a2e', fontsize=12)
    ax2b.set_ylim(0, 105)
    ax2b.spines['top'].set_visible(False)

    ax2b.axhline(50, color='grey', linestyle='--', alpha=0.6, lw=1.2)
    ax2b.axhline(90, color='grey', linestyle=':', alpha=0.6, lw=1.2)
    ax2b.text(6, 53, '50% (Median)', fontsize=10, color='#555', fontweight='bold')
    ax2b.text(8.5, 93, '90%', fontsize=10, color='#555', fontweight='bold')

    plt.tight_layout()
    out = EDITED_FIG_DIR / 'Fig19_FIXED_POLISHED.png'
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"     Saved: {out.name}")
# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("=" * 65)
    print(" Generating Final Q1-Polished Figures")
    print(f" Output Directory: {EDITED_FIG_DIR}")
    print("=" * 65)
    
    # Call the function to actually generate the plot
    fig19_transport_age()
    
    print("\n COMPLETE")
    print("=" * 65)
