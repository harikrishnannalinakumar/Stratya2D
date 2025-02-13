# Stratya2D - Kinematic Decompaction and Backstripping

Straya2D enhances traditional basin analysis by extending 1D decompaction and backstripping methodologies to a 2D framework using seismic cross-sections. This Python-based tool leverages image processing techniques to integrate horizon extraction, depth normalisation, and Monte Carlo Simulation for uncertainty quantification.

## Features

- **Backstripping and Decompaction**: Performs backstepping using Monte Carlo simulation.
- **Horizon Extraction**: Automatically extracts and processes seismic horizons from PNG or JPEG images.
- **2D Decompaction and Backstripping**: Calculates changes in depositional thickness over time across a 2D seismic cross-section.
- **Monte Carlo Simulation**: Quantifies uncertainties in tectonic subsidence and depositional thickness.
- **Visualisation**: Provides dynamic 2D visualisations of basin evolution and horizon dynamics.
- **Horizon Flattening**: Adjusts seismic horizons to a common reference level, enabling clearer stratigraphic interpretation and basin evolution analysis.

## How to set up?

1) Download the repository by cloning it or manually downloading the folder.
   
2) Ensure Python is installed on your machine.

## Usage

### Configure parameters
1) Open ``main.py`` and input the following parameters:
  - Tectonic subsidence calculations (Monte Carlo simulation).
  - Horizon extraction and smoothing settings. *(Default values generally work well in most cases.)*
  - Vertical and horizontal distance normalisation for seismic data.
  - Well location.
    
### Input image with horizons marked
2) Place an input image with marked horizons inside: ``input/{figure_name}`` *(You can use the default image provided in the folder for testing.)*

### Run the Code
3) Execute ```python main.py```
