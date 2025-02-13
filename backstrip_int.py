import numpy as np
import csv
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import main as inp 

well_length = inp.well_length
well_loc = inp.well_loc

fig = inp.fig

filename = 'result//monte_carlo//averages//Decompacted_thickness_avg.csv'
filename1 = 'result//csv//Calculated_Decompacted_thickness_1.csv'
filename2 = f'result//csv//{fig}//combined.csv'
directory = "result/csv/updated_values"

output_directory = "result//csv//updated_values//"


#%% ## Merging the CSV column will all the values

def combine_files(files):
    files = sorted(files, reverse=True)
    
    dfs = [pd.read_csv(file).set_index('X') for file in files]

    combined_df = pd.concat(dfs, axis=1)
    
    component_names = [os.path.basename(file).replace("_adjusted.csv", "") for file in files]
    combined_df.columns = [f"Layer {i}" for i in range(len(files), 0, -1)]

    result_df = combined_df.reset_index()
    result_df.to_csv(f"result//csv//{fig}//combined.csv", index=False)

    return result_df

def get_files(fig):
    files = []

    label = 1
    while True:
        filename = f'result//csv//{fig}//line_points_component_{label}_smoothed_adjusted.csv'
        if os.path.exists(filename):
            # print(f"Found: {filename}")
            files.append(filename)
            label += 1
        else:
            break

    return files

#%%
file = get_files(fig)
combined_df = combine_files(file)

#%% ## Reformatting the Decompacted thickness

# Read the CSV file into a DataFrame
df = pd.read_csv(filename)

df = df[df.columns[::-1]]

df = df.iloc[::-1]

df.rename(columns={"Unnamed: 0": "Layers"}, inplace=True)

cols = ['Layers'] + sorted([col for col in df.columns if col != 'Layers'], key=float, reverse=True)
df = df[cols]

df.to_csv(filename1, index=False)

#%% ## Making the csv and renaming the columns according to the calcaulted_decompacted thickness
    ## while considering erosion as well
    
calculated_df_path = filename1
calculated_df_new = pd.read_csv(calculated_df_path)

labels_all_zeros = calculated_df_new[calculated_df_new.iloc[:, 1:].sum(axis=1) == 0]["Layers"].tolist()

combined_df_path = filename2
combined_df_new = pd.read_csv(combined_df_path)

labels_new = calculated_df_new["Layers"].tolist()
#%%
def rename_col(combined_df, labels, zero_labels):
    # Starting from the second column (i.e., index 1)
    col_index = 1
    for label in labels:
        if label in zero_labels:
            # Insert a zero column
            combined_df.insert(col_index, label, 0)
            col_index += 1
        else:
            combined_df.columns.values[col_index] = label
            col_index += 1
    return combined_df

final_combined_df = rename_col(combined_df_new, labels_new, labels_all_zeros)

output_filename_combined_revised = 'result/csv/combined_difference_renamed.csv'
final_combined_df.to_csv(output_filename_combined_revised, index=False)

#%% ## Putting the well location and normalising the data

combined_diff_df = pd.read_csv('result//csv//combined_difference_renamed.csv')
filename1_df = pd.read_csv(filename1)


zero_columns = combined_diff_df.columns[(combined_diff_df == 0).all()] 
non_zero_columns = combined_diff_df.columns[(combined_diff_df != 0).any()]    
combined_diff_df = combined_diff_df.drop(columns=zero_columns)
new_column_names = ['X'] + list(range(len(non_zero_columns) - 1)) # 'X' is preserved for the first column
combined_diff_df.columns = new_column_names

#%%

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

#%%

decompacted_depth_df = filename1_df
extracted_values_df = combined_diff_df

def normalization(df, well_locs, y_values, columns_to_transform):
    X = np.array(well_locs).reshape(-1, 1)
    y = np.array(y_values).reshape(-1, 1)

    reg_model = LinearRegression().fit(X, y)

    transformed_df = df.copy()
    for col in columns_to_transform:
        transformed_df[col] = transformed_df[col].apply(lambda x: reg_model.coef_[0][0] * x + reg_model.intercept_[0])
        
    transformed_df = transformed_df[['X'] + list(columns_to_transform)]

    return transformed_df
first_col_mat = extracted_values_df.loc[extracted_values_df['X'] == well_loc, extracted_values_df.columns[1]].iloc[0]
expected_value_first = decompacted_depth_df.iloc[0,1]
ratio = expected_value_first / first_col_mat

timestep1_df = extracted_values_df[['X', extracted_values_df.columns[1]]].copy()
timestep1_df[extracted_values_df.columns[1]] *= ratio


#%%
# Handle the first time step
header = decompacted_depth_df.columns[1]
timestep1_df.to_csv(f'result/csv/updated_values/modified_Layer_loc_at_{header}.csv', index=False)

# Updated code for handling subsequent timesteps
for col_index in range(2, len(extracted_values_df.columns)):
    data_col_index = col_index 

    if data_col_index < len(decompacted_depth_df.columns):
            value_match = extracted_values_df.loc[extracted_values_df['X'] == well_loc, extracted_values_df.columns[1:col_index + 1]].values[0]
            
        # Get the corresponding column in decompacted_depth_df and slice it to match the length of value_match
            expected_values = decompacted_depth_df.iloc[:, data_col_index].values[:len(value_match)]
            timestep_df = normalization(extracted_values_df, value_match, expected_values, extracted_values_df.columns[1:col_index + 1])
            header = decompacted_depth_df.columns[data_col_index]
            timestep_df.to_csv(f'result/csv/updated_values/modified_Layer_loc_at_{header}.csv', index=False)
    else:
        print(f"Skipping column index {col_index} as it is out of bounds for decompacted_depth_df.")