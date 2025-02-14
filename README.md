# Stratya2D - Kinematic Decompaction and Backstripping

Straya2D enhances traditional basin analysis by extending 1D decompaction and backstripping methodologies to a 2D framework using seismic cross-sections. This Python-based tool leverages image processing techniques to integrate horizon extraction, depth normalisation, and Monte Carlo Simulation for uncertainty quantification.

## Features

- **Backstripping and Decompaction**: Uses Monte Carlo simulation for uncertainty estimation.
- **Horizon Extraction**: Automatically detects and processes seismic horizons from PNG or JPEG images.
- **2D Decompaction and Backstripping**: Calculates changes in depositional thickness over time across a 2D seismic cross-section.
- **Monte Carlo Simulation**: Quantifies uncertainties in tectonic subsidence and sediment compaction.
- **Visualisation**: Provides dynamic 2D visualisations of basin evolution and horizon dynamics.
- **Horizon Flattening**: Adjusts seismic horizons to a common reference level, enabling clearer stratigraphic interpretation and basin evolution analysis.

## Installation

1) Clone the Repository
   
   ```bash
   git clone https://github.com/harikrishnannalinakumar/Stratya2D.git
   cd Stratya2D
   ```
2) Install the Dependencies

   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have Python installed before running this command.)*
   
## User Guide

### Configure parameters
1) Open ``main.py`` and input the following parameters:
  - Tectonic subsidence calculations (Monte Carlo simulation).
  - Horizon extraction and smoothing settings *(default values generally work well in most cases).*
  - Vertical and horizontal distance normalisation for seismic data.
  - Well location.
    
### Input image with horizons marked
2) Place an input image with marked horizons inside: ``input/{figure_name}``
   <br> *(You can use the default image provided in the folder for testing.)*

### Run the Code
3) Execute
   
   ```bash
     python main.py
    ```
