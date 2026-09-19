#!/bin/bash
# =============================================================================
# STEP 1: Create CDO grid description files for both FLEXPART output domains
# Research: Black Carbon from wildfires → Khumbu Glacier deposition
# Following: Eckhardt et al. (2017), GMD 10, 4605-4618
# =============================================================================

BASE_DIR="$Path.cwd()"
GRID_DIR="$BASE_DIR/GRID_FILES"
mkdir -p "$GRID_DIR"

echo "============================================================"
echo " Creating CDO grid description files"
echo "============================================================"

# --- Coarse grid (0.25 degree) ---
# OUTLON0=60.0, NUMXGRID=181, DXOUT=0.25 → centers start at 60.125
# OUTLAT0=15.0, NUMYGRID=161, DYOUT=0.25 → centers start at 15.125
cat > "$GRID_DIR/flexpart_coarse_0.25deg.txt" << 'EOF'
gridtype  = lonlat
xsize     = 181
ysize     = 161
xfirst    = 60.125
xinc      = 0.25
yfirst    = 15.125
yinc      = 0.25
EOF

# --- Nested grid (0.1 degree) ---
# OUTLON0N=75.0, NUMXGRIDN=201, DXOUTN=0.1 → centers start at 75.05
# OUTLAT0N=20.0, NUMYGRIDN=151, DYOUTN=0.1 → centers start at 20.05
cat > "$GRID_DIR/flexpart_nested_0.1deg.txt" << 'EOF'
gridtype  = lonlat
xsize     = 201
ysize     = 151
xfirst    = 75.05
xinc      = 0.1
yfirst    = 20.05
yinc      = 0.1
EOF

echo " Grid files created:"
echo "   Coarse : $GRID_DIR/flexpart_coarse_0.25deg.txt"
echo "            → 181×161 cells, 0.25°, lon:60.125-105.125°E, lat:15.125-55.125°N"
echo "   Nested : $GRID_DIR/flexpart_nested_0.1deg.txt"
echo "            → 201×151 cells, 0.10°, lon:75.05-95.05°E,   lat:20.05-35.05°N"

# --- Pre-generate conservative remapping weights ---
# We generate weights ONCE from a sample file and reuse for all days
# This saves huge amounts of time compared to computing weights per file

echo ""
echo "============================================================"
echo " Generating conservative remapping weights (done ONCE)"
echo " This may take 1-3 minutes..."
echo "============================================================"

SAMPLE_FILE="$BASE_DIR/ENSEM_RAW/EnsemFire_v1.0_20210301.nc"

if [ ! -f "$SAMPLE_FILE" ]; then
    echo " ERROR: Sample EnsemFire file not found: $SAMPLE_FILE"
    exit 1
fi

# Extract BC only from sample to speed up weight generation
cdo -s -selvar,BC "$SAMPLE_FILE" "$GRID_DIR/sample_bc.nc"

# Generate weights for COARSE grid (0.25°) - conservative remapping
echo "   Generating weights for coarse grid (0.25°)..."
cdo -s gencon,"$GRID_DIR/flexpart_coarse_0.25deg.txt" \
    "$GRID_DIR/sample_bc.nc" \
    "$GRID_DIR/weights_coarse_0.25deg.nc"
echo "    Coarse weights saved"

# Generate weights for NESTED grid (0.1°) - conservative remapping
echo "   Generating weights for nested grid (0.1°)..."
cdo -s gencon,"$GRID_DIR/flexpart_nested_0.1deg.txt" \
    "$GRID_DIR/sample_bc.nc" \
    "$GRID_DIR/weights_nested_0.1deg.nc"
echo "    Nested weights saved"

# Cleanup sample
rm -f "$GRID_DIR/sample_bc.nc"

echo ""
echo "============================================================"
echo " STEP 1 COMPLETE"
echo " Grid files and remapping weights are ready."
echo " Now run: bash 02_regrid_ensemfire.sh"
echo "============================================================"
