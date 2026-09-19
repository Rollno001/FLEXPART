# Python and FLEXPART Scripts for Pre-monsoon Khumbu Glacier Black Carbon Analysis

# Author: Aashutosh Paudel
# Associated Manuscript: [Bridging Regional Fires and Glacial Black Carbon Loading Through Atmospheric Transport at Khumbu Glacier]

# Description
This repository contains the code necessary to reproduce the FLEXPART emission sensitivity simulations and subsequent Python-based geospatial analysis for Black Carbon transport to the Khumbu Glacier (Pre-monsoon 2021).

# FLEXPART Model Version & Compilation
Simulations were performed using **FLEXPART v11** (or update with your exact version). The model source code, compilation instructions,
and required libraries (netCDF, eccodes) can be found at the official FLEXPART repository: https://www.flexpart.eu/ 
Users must compile the `flexpart` executable on their local system before running the scripts provided here.

# Directory Structure

**1_FLEXPART_Simulations/**
Contains the bash automation script (`run_flexpart_backward_nested.sh`) and the configuration files. 
* `options_dry/` and `options_wet/`: Contain the parameter files for the respective dry and wet deposition simulation runs.
* `AVAILABLE_template.txt`: A sample layout of the meteorological input files.

**2_Data_Preprocessing/**
Contains the bash scripts used to prepare the wildfire emission inventories prior to Python analysis. Requires CDO (Climate Data Operators).
* `01_setup_grids.sh`: Generates CDO grid description files and conservative remapping weights for the 0.25° and 0.10° domains.
* `02_regrid_ensemfire.sh`: Regrids the raw EnsemFire Black Carbon emissions to match the FLEXPART spatial grids.

**3_Python_Analysis/**
Contains the sequential Python scripts used to process the raw FLEXPART output alongside the regridded EnsemFire emissions.
* `03_compute_bc_deposition.py`: Calculates daily BC surface deposition.
* `04_plot_results.py`: Generates deposition timelines and spatial maps.
* `05_compute_bc_concentration.py`: Calculates daily atmospheric BC concentration.
* `07b_plot_sensitivity_footprint_panels.py`: Generates the 2x3 panel of emission sensitivity footprints.
* `08_regional_fixed.py`: Calculates source contribution percentages (Nepal vs. IGP).
