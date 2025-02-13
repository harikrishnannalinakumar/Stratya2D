#%%
# -----------------------------------------------------------------------------
### Inputs requied for carrying out backstripping (Current values are random)

lith = ["Quarzite","Anhydrite","Sandstone","Dolomite","Limestones","Sandstone",
        "Chalk","Shales"]       # Lithology
poro = [0.2,0.05,0.49,0.2,0.4,0.49,0.7,0.63] # Surface porosity
const = [0.3,0.2,0.27,0.60,0.60,0.27,0.71,0.51]    # Exponential constant (km-1)
density = [2650,2960,2650,2870,2710,2650,2710,2720] # Grain density (Kg/m3)

decomp_depth_age_start = ['260']                 # Deposition age of unit 1
age = ['245', '210','160','145','100', '80','45','0']  # Age of the layer (Ma)

y1 = [2450,2100,1650,1330,1000,600,260,0]    # Top - Thickness (lower) (m)
y2 = [3220,2450,2100,1650,1330,1000,600,260] # Bottom - Thickness (m)  ( y2 > y1) 
# Paleo and Sea_level will have one more value than the number of units. 
# This represents the sea level at the time of deposition of the first unit
# (i.e) for the deposition of the first unit (value at the decomp_depth_start_age). 
# The first value corresponds to the Older unit and the last value corresponds to the Younger unit.

paleo = [-20,0,20,10,20,20,200,300,350]  # Paleobathymetry and Eustaic sea level 
# relative to present day Wd (m)
sea_level = [10,0,0,-20,-40,70,80,100,50] # Sea level realtive to today Del_sl (m)

density_water = 1030               # Desnity of water
density_mantle = 3300               # Density of mantle

# -----------------------------------------------------------------------------
### Inputs for the Monte Carlo estimation

## Upper and Lower limits for porosity, exponential constant, density, paleo 
# and sea level

poro_lower_limits = [0.15, 0.02, 0.40, 0.12, 0.3, 0.4, 0.6, 0.6 ]  
poro_upper_limits = [0.4, 0.25, 0.55,  0.29, 0.5, 0.55, 0.77, 0.7] 

const_lower_limits = [0.2,  0.1,     0.2,   0.4,    0.4,  0.25,    0.65,  0.45]
const_upper_limits = [0.6,  0.35,    0.6,   0.8,    0.8,  0.35,    1.0,   0.8 ]

density_lower_limits = [2600, 2900, 2600, 2800, 2650, 2600, 2650, 2700,]
density_upper_limits = [2700, 3000, 2700, 2900, 2750, 2700, 2750, 2800]

paleo_upper_limits = [-10,10,  30,  20,  30,  30,  210, 310, 360,]
paleo_lower_limits = [-30,-10, -10, -10, -10, -10, 190, 290, 340, ]

sea_level_lower_limits = [20,-10, -10, -30, -50, 60, 70,  90, 40, ]
sea_level_upper_limits = [ -10,20,  10,   10, 0,   10,  80, 90, 110]

# -----------------------------------------------------------------------------

#%% Inputs for image processing

fig = 'test' ## Name of the input fugure under 'input/figure/'

max_depth = 5.350   # Maximum depth of the seismic lines in Kilometers
max_distance = 200.37 # Maximum distance of the sesimic line in Kilometers.

well = "Sample Well"  ## Insert the well name

well_loc = 131 # Well location in Kilometer
well_length = 3.781 # Well  depth in Kilometers 
well_name = well

### Inputs for savgol filter

polyorder = 2 ## Default value works in most cases.


#%% For interpolating horizons to a common X-range (start and end points)
#
# `start_range` and `end_range` define the X-axis range where horizons 
# will be interpolated. 
#
# - `start_range`: Any horizon starting **before** this value **won’t be interpolated**.
# - `end_range`: Any horizon ending **after** this value **won’t be interpolated**.
# The default values wokrs in most cases

start_range = 1  # Minimum X values for interplation in km
end_range = 20   # Maximum X value for interpolation in km

#%% Input for Horizon flattening

## Horizon number for carrying out Horizon flattening

flat_horizon = "1"

# %% For saving the images

# save_images = True
save_images = False

#%% Running the code 

import time
from tqdm import tqdm  

steps = [
    ("Carrying out Backstripping", "backstripping"),
    ("Horizon extraction and Normalisation", "horizon_reco"),
    ("Horizon processing", "data_adj"),
    ("Integrating Backstripping results", "backstrip_int"),
    ("Plotting", "plot_results"),
    ("Horizon flattening", "horizon_flatt")
]

if __name__ == "__main__":
    print("\n .... ***Starting process***....\n")  
    start_time = time.time()

    with tqdm(total=len(steps), desc="Progress", unit="step") as pbar:
        for step_name, module_name in steps:
            print(f"\n\n Step: {step_name}...")  
            # time.sleep(0.5)  
            __import__(module_name)  
            pbar.update(1)  

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  
    
    print("\n\n .....***Process Completed Successfully!***..... \n")
    print(f" Total Time Elapsed: {elapsed_time:.2f} minutes")
