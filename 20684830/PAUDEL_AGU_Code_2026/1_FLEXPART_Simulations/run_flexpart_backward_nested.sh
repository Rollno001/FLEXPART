#!/bin/bash

# FLEXPART base directories
BASE_DIR=$PWD
OPTIONS_DIR="$BASE_DIR/options"
INPUTS_DIR="$BASE_DIR/inputs"

# If FLEXPART exists but flexpart does not, rename it
if [[ -f "$BASE_DIR/FLEXPART" && ! -f "$BASE_DIR/flexpart" ]]; then
    echo "Renaming FLEXPART -> flexpart"
    mv "$BASE_DIR/FLEXPART" "$BASE_DIR/flexpart"
fi

# Ensure executable permission
chmod +x "$BASE_DIR/flexpart"

# Dates
START=20210509
END=20210531
STEP=1   # days

# Retry settings
MAX_RETRIES=7
RETRY_DELAY=10  # seconds to wait before retrying

DATES=()
current="$START"

while [[ "$current" -le "$END" ]]; do
  DATES+=("$current")
  current=$(date -d "${current} +${STEP} days" +"%Y%m%d")
done

for DATE in "${DATES[@]}"; do
  echo "=== Running backward simulation for $DATE ==="

  # Compute dates
  RECEPTOR_DATE=$DATE
  START_DATE=$(date -d "$DATE -10 days" +"%Y%m%d")
  RELEASE_START=$(date -d "$DATE -1 day" +"%Y%m%d")

  OUTPUT_DIR="$BASE_DIR/output/nested/$DATE"
  mkdir -p "$OUTPUT_DIR"

  # -------------------
  # COMMAND file
  # -------------------
  cat > "$OPTIONS_DIR/COMMAND" <<EOF
&COMMAND
 LDIRECT = -1,
 IBDATE = $START_DATE, IBTIME = 000000,
 IEDATE = $RECEPTOR_DATE, IETIME = 000000,
 LOUTSTEP = 10800,
 LOUTAVER = 10800,
 LOUTSAMPLE = 90,
 LSYNCTIME = 90,
 CTL = 0.1,
 IFINE = 4,
 IOUT = 9,
 IPOUT = 0,
 LSUBGRID = 1,
 LCONVECTION = 1,
 LAGESPECTRA = 1,
 IPIN = 0,
 IOUTPUTFOREACHRELEASE = 1,
 IND_SOURCE = 1,
 IND_RECEPTOR = 4,
 NESTED_OUTPUT = 1,
/
EOF

  # -------------------
  # RELEASES file
  # -------------------
  cat > "$OPTIONS_DIR/RELEASES" <<EOF
&RELEASES_CTRL
 NSPEC = 1,
 SPECNUM_REL = 40,
/

&RELEASE
 IDATE1 = $RELEASE_START, ITIME1 = 000000,
 IDATE2 = $RELEASE_START, ITIME2 = 235959,
 LON1 = 86.80, LON2 = 86.85,
 LAT1 = 27.94, LAT2 = 27.97,
 Z1 = 0.0, Z2 = 30.0,
 ZKIND = 1,
 MASS = 1.0,
 PARTS = 100000,
 COMMENT = "Khumbu_BC_$DATE",
/
EOF

  # -------------------
  # pathnames file
  # -------------------
  cat > "$BASE_DIR/pathnames" <<EOF
./options/
./output/nested/$DATE/
./inputs/
./inputs/AVAILABLE
EOF

  # -------------------
  # Run FLEXPART with retry on error
  # -------------------
  attempt=1
  success=false
  
  while [[ $attempt -le $MAX_RETRIES ]]; do
    echo "==================== Start Simulation (Attempt $attempt/$MAX_RETRIES) ============="
    
    ./flexpart
    EXIT_CODE=$?
    
    if [[ $EXIT_CODE -ne 0 ]]; then
      echo "  FLEXPART failed (exit code: $EXIT_CODE) on attempt $attempt for $DATE"
      
      if [[ $attempt -lt $MAX_RETRIES ]]; then
        echo " Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
        
        # Clean up partial output before retry
        rm -f "$OUTPUT_DIR"/*.nc "$OUTPUT_DIR"/grid_* 2>/dev/null
        
        ((attempt++))
      else
        echo " Max retries ($MAX_RETRIES) reached for $DATE. Moving to next date."
        break
      fi
    else
      echo " Simulation completed successfully for $DATE"
      success=true
      break
    fi
  done

  if [[ "$success" == "false" ]]; then
    echo " Failed to complete simulation for $DATE after $attempt attempt(s)"
  fi

  echo "==================== Finished $DATE ==============="
done

echo " All simulations completed!"
