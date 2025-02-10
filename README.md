# Stratya2D - Kinematic Decompaction and Backstripping

Straya2D enhances traditional basin analysis by extending 1D decompaction and backstripping methodologies to a 2D framework using seismic cross-sections. This Python-based tool leverages image processing techniques to integrate horizon extraction, depth normalisation, and Monte Carlo Simulation for uncertainty quantification.

## Features

- **Horizon Extraction**: Automatically extracts and processes seismic horizons from PNG or JPEG images.
- **2D Decompaction and Backstripping**: Calculates changes in depositional thickness over time across a 2D seismic cross-section.
- **Monte Carlo Simulation**: Quantifies uncertainties in tectonic subsidence and depositional thickness.
- **Visualisation**: Provides dynamic 2D visualisations of basin evolution and horizon dynamics.
- **Horizon Flattening**: Adjusts seismic horizons to a common reference level, enabling clearer stratigraphic interpretation and basin evolution analysis.

## How to set up?

Download the repository folder and run the code using in any machine with python installed. 

## Usage

1) Provide an input image with the horizons marked.
   
2) Open input.py and specify the main parameters for tectonic subsidence calculation, including Monte Carlo simulation. Configure parameters for horizon extraction, smoothing, and distance normalisation, as well as the well location. *(Default values generally work well in most cases.)*
   
3) Run stratya.py. If the extracted horizons do not align correctly, adjust the horizon extraction parameters in Step 2 and re-run the process.

4) For Horizon Flattening, run flattening.py.

 For testing purposes choose the default image in the folder.


