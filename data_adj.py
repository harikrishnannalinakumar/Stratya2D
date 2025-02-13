import os
import pandas as pd
import matplotlib.pyplot as plt

from main import well_length,well_loc,well_name,fig,max_depth,start_range,end_range
max_depth = max_depth*1000

#%% ## For interpolating the x and y axis to the minimum value
def y_adjust(start_x_range, end_x_range):
   
    directory_path = f'result//csv//{fig}//'
   
    filepaths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('_smoothed.csv')]
    if not filepaths:
        return []

    start_end_values = {}
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df = df.drop_duplicates(subset='X', keep='first')
        start_x = df['X'].min()
        end_x = df['X'].max()
        start_end_values[filepath] = (start_x, end_x)

    global_min_start = min([val[0] for val in start_end_values.values()])
    global_max_end = max([val[1] for val in start_end_values.values()])
    
    adjusted_files = []
    for filepath, (start, end) in start_end_values.items():
        df = pd.read_csv(filepath)
        # Ensure no duplicates in 'X' column
        df = df.drop_duplicates(subset='X', keep='first')
        df.set_index('X', inplace=True)
        adjusted_start = start if start > start_range else global_min_start
        adjusted_end = end if end < end_range else global_max_end
        df_interpolated = df.reindex(range(adjusted_start, adjusted_end + 1)).interpolate(method='linear', limit_direction='both').reset_index()
        adjusted_path = filepath.replace(".csv", "_adjusted.csv")
        df_interpolated.to_csv(adjusted_path, index=False)
        adjusted_files.append(adjusted_path)

    return adjusted_files, global_min_start, global_max_end
#%% ## Making the X axis same across all the sheets and making y values NaN if dosent exist

def std_x_range(files, min_x, max_x):
    standardized_files = []

    for file in files:
        df = pd.read_csv(file)
        df.set_index('X', inplace=True)
        # Adjust x-axis range without interpolating y-values.
        # Missing y-values will be filled with NaN by default.
        df_standardized = df.reindex(range(min_x, max_x + 1)).reset_index()
        # Overwrite the existing file
        df_standardized.to_csv(file, index=False)
        standardized_files.append(file)
        
    return standardized_files
#%% ## Combine the sheet and create the difference

def compute_differences(files):

    files = sorted(files, reverse=True) # Reversing to process the horizons in reverse order
    

    dfs = [pd.read_csv(file).set_index('X') for file in files]

    combined_df = pd.concat(dfs, axis=1)
    component_names = [os.path.basename(file).replace("_adjusted.csv", "") for file in files]

    combined_df.columns = component_names
    combined_df.columns = [f"Layer {i}" for i in range(len(files), 0, -1)]

    # Calculate the differences
    diff_dfs = []
    for i in range(len(component_names), 1, -1):
        # Check for NaN in the base layer (e.g. Layer 4)
        base_layer = combined_df[f"Layer {i-1}"]
        nan_mask = base_layer.isnull()
        
        # If there are NaN values and it's not the last layer being considered, then replace those NaN values with the next layer's values
        if nan_mask.sum() > 0 and i > 2:
            base_layer = base_layer.where(~nan_mask, combined_df[f"Layer {i-2}"])
    
        diff_col = (combined_df[f"Layer {i}"] - base_layer)
        
        diff_dfs.append(diff_col.rename(f"Layer {i} - Layer {i-1}"))

    result_df = pd.concat(diff_dfs + [combined_df["Layer 1"]], axis=1).reset_index()
    result_df.to_csv(f"result//csv//{fig}//combined_differences.csv", index=False)

    return result_df
#%%
def plot_interpolated(files):
    plt.figure(figsize=(10, 6))
    horizon_count = 1 
    for file in files:
        df = pd.read_csv(file)
        label = f"Horizon {horizon_count}"
        plt.plot(df['X']/1000, df['Smooth_Y'], label=label)
        horizon_count += 1  # Increment the counter for the next file
    plt.vlines(x=well_loc, ymin=0, ymax=well_length*1000, colors='k', linestyles='solid')
    plt.ylim(-50.0, max_depth+100)
    plt.plot(well_loc, -10, marker='X', markersize=8, color='k', label=well_name)
    plt.title('Smooth and Extrapolated horizons')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')   
    plt.grid(False)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.legend(loc='lower right')
    plt.show()
    
#%%

# Sample usage with given start and end x-value ranges for demonstration
start_range = start_range
end_range = end_range
adj_df, min_x, max_x = y_adjust(start_range, end_range)
std_df = std_x_range(adj_df, min_x, max_x)
#%%
combined_result = compute_differences(adj_df)
# plot_interpolated(adj_df)
