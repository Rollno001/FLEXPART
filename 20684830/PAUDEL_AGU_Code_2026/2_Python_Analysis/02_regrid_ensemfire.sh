#!/bin/bash
# =============================================================================
# STEP 2: Regrid EnsemFire BC emission files to both FLEXPART grid domains
# Uses CDO conservative remapping (correct for flux variables like kg s-1 m-2)
# Must run AFTER 01_setup_grids.sh
# =============================================================================

BASE_DIR="$Path.cwd()"
ENSEM_DIR="$BASE_DIR/ENSEM_RAW"
GRID_DIR="$BASE_DIR/GRID_FILES"

# Output directories for regridded files
OUT_COARSE="$BASE_DIR/ENSEM_REGRID/coarse_0.25deg"
OUT_NESTED="$BASE_DIR/ENSEM_REGRID/nested_0.1deg"

mkdir -p "$OUT_COARSE" "$OUT_NESTED"

# Check weights exist
if [ ! -f "$GRID_DIR/weights_coarse_0.25deg.nc" ] || [ ! -f "$GRID_DIR/weights_nested_0.1deg.nc" ]; then
    echo " ERROR: Remapping weights not found. Please run 01_setup_grids.sh first."
    exit 1
fi

echo "============================================================"
echo " STEP 2: Regridding EnsemFire BC to FLEXPART grids"
echo " Using CDO conservative remapping (correct for flux fields)"
echo "============================================================"
echo ""

# Count files to process
TOTAL=$(ls "$ENSEM_DIR"/EnsemFire_v1.0_2021{03,04,05}??.nc 2>/dev/null | wc -l)
echo " Files to process: $TOTAL (March + April + May 2021)"
echo ""

COUNT=0
FAILED=0

for ENSEM_FILE in "$ENSEM_DIR"/EnsemFire_v1.0_2021{03,04,05}??.nc; do

    [ -f "$ENSEM_FILE" ] || continue

    BASENAME=$(basename "$ENSEM_FILE")
    DATE_STR=${BASENAME:15:8}   # Extract YYYYMMDD from EnsemFire_v1.0_YYYYMMDD.nc

    OUT_C="$OUT_COARSE/EnsemFire_BC_coarse_${DATE_STR}.nc"
    OUT_N="$OUT_NESTED/EnsemFire_BC_nested_${DATE_STR}.nc"

    COUNT=$((COUNT + 1))
    printf " [%3d/%3d] Processing %s ... " "$COUNT" "$TOTAL" "$DATE_STR"

    # Skip if already done (useful for reruns)
    if [ -f "$OUT_C" ] && [ -f "$OUT_N" ]; then
        echo "SKIPPED (already exists)"
        continue
    fi

    # Step A: Extract BC variable only (much faster than processing all variables)
    TEMP_FILE=$(mktemp /tmp/ensem_bc_XXXXXX.nc)

    cdo -s -selvar,BC "$ENSEM_FILE" "$TEMP_FILE" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo " FAILED (selvar)"
        FAILED=$((FAILED + 1))
        rm -f "$TEMP_FILE"
        continue
    fi

    # Step B: Remap to COARSE grid (0.25°) using pre-computed weights
    cdo -s remap,"$GRID_DIR/flexpart_coarse_0.25deg.txt","$GRID_DIR/weights_coarse_0.25deg.nc" \
        "$TEMP_FILE" "$OUT_C" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo " FAILED (remap coarse)"
        FAILED=$((FAILED + 1))
        rm -f "$TEMP_FILE" "$OUT_C"
        continue
    fi

    # Step C: Remap to NESTED grid (0.1°) using pre-computed weights
    cdo -s remap,"$GRID_DIR/flexpart_nested_0.1deg.txt","$GRID_DIR/weights_nested_0.1deg.nc" \
        "$TEMP_FILE" "$OUT_N" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo " FAILED (remap nested)"
        FAILED=$((FAILED + 1))
        rm -f "$TEMP_FILE" "$OUT_N"
        continue
    fi

    rm -f "$TEMP_FILE"
    echo " done"

done

echo ""
echo "============================================================"
echo " STEP 2 COMPLETE"
echo " Processed : $((COUNT - FAILED)) / $TOTAL files successfully"
if [ $FAILED -gt 0 ]; then
    echo "   Failed  : $FAILED files"
fi
echo ""
echo " Regridded files saved to:"
echo "   Coarse: $OUT_COARSE"
echo "   Nested: $OUT_NESTED"
echo ""
echo " Now verify the output with:"
echo "   ncdump -h $OUT_COARSE/EnsemFire_BC_coarse_20210301.nc"
echo "   ncdump -h $OUT_NESTED/EnsemFire_BC_nested_20210301.nc"
echo ""
echo " Then run: python3 03_compute_bc_deposition.py"
echo "============================================================"
