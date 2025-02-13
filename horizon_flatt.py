import math
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import main as inp
import os

# %%

# Enter the age to be viewed.
file_path = "result/csv/updated_values/modified_Layer_loc_at_0.csv"
df = pd.read_csv(file_path)

flatten_layer = inp.flat_horizon
flatten_layer = str(int(flatten_layer) - 1)

fig = inp.fig

#%%

def save_plot(filename):
    if inp.save_images:
        plt.savefig(filename,bbox_inches='tight',dpi=600)

# %%
def process_file( column_name, output_file_path, df):
    if column_name in df.columns:
        first_column = df.columns[0]
        for col in df.columns:
            if col != column_name and col != first_column:
                df[col] = df[col] - df[column_name]

        df[column_name] = 0

        df.to_csv(output_file_path, index=False)
    else:
        print(f"Error: Column '{column_name}' not found. Available columns are: {df.columns.tolist()}")

output_path = f"result/horizon_flattening/{fig}/{flatten_layer}_flattened.csv"  

output_dir = os.path.dirname(output_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
          
result = process_file( flatten_layer, output_path, df)
result

# %%
def read_and_plot(file_path):
    df = pd.read_csv(file_path)

    x = df.iloc[:, 0]
    y_columns = df.columns[1:]  

    plt.figure(figsize=(10, 6))

    for col in y_columns:
        plt.plot(x/1000, df[col], label=f'Horizon {int(col)+1}')

    plt.xlabel('Distance (km)')  
    plt.ylabel('Depth (m)') 
    plt.title(f'Horizons after flattening Horizon {int(flatten_layer) + 1}') 
    plt.legend(loc='lower left')
    # plt.ylim(bottom=0)
    plt.gca().invert_yaxis()
    plt.show()
    save_plot(f"result/horizon_flattening/{fig}/{(int(flatten_layer))+1}_flat.png")

read_and_plot(output_path)
