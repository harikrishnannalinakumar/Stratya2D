import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os
from matplotlib import cm 
import warnings
import main as inp
from main import well_length,well_loc,well_name,max_depth


#%%
def save_plot(filename):
    if inp.save_images:
        plt.savefig(filename,bbox_inches='tight',dpi=600)

#%%

warnings.filterwarnings("ignore")

max_depth = max_depth*1000
well_length = well_length*1000

fig = inp.fig

direct = "result//csv//updated_values//"
fig_dir = "result//figure//"

file_list = [file for file in os.listdir(direct) if file.startswith("modified_Layer_loc_")]

file_list.sort(key=lambda x: int(x.split('at_')[1].split('.csv')[0]), reverse=True)

cmap_name = 'turbo'

cmap = cm.get_cmap(cmap_name, len(file_list)) 

cmap = cm.get_cmap(cmap_name)

colors = cmap(np.linspace(0, 1, len(file_list)))  # Generate colors dynamically

#%% ## Plotting the decompaction at every interval.

def plot_decompaction(file_path, times, next_file=None, last_file=False,fig_dir=fig_dir):
    df = pd.read_csv(file_path)
    
    zero_cols = df.columns[df.eq(0).all()]
    df = df.drop(columns=zero_cols)
    columns = df.columns[1:]    
    
    timestep = int(file_path.split('at_')[1].split('.csv')[0])
    plt.figure()
    df_copy = df.copy()
    cols_to_consider = columns[::-1]
    max_m = len(cols_to_consider)
    
    topmost_horizon = cols_to_consider[0]
    
    plt.title(f'{timestep} MA before present')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')
    plt.plot(df_copy['X']/1000, df_copy[topmost_horizon], color='k',linewidth=0.45)
    plt.fill_between(df_copy['X']/1000, df_copy[topmost_horizon], 0, 
                 facecolor=colors[times % len(colors)],  
                 label=f'Layer {int(topmost_horizon)+1}', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    _, upper_y = plt.ylim()
    plt.ylim(-50, max_depth)
    plt.legend(loc='lower left')
    plt.grid(False)
    plt.gca().invert_yaxis()
    
    plt.show()

    for i in range(len(cols_to_consider)-1):
        col_base = cols_to_consider[i+1]
        col_top = cols_to_consider[i]
        m = max_m-i
        color = colors[m-2]
        # plt.plot(df_copy['X'], df_copy[col_base], color=color)
        plt.plot(df_copy['X']/1000, df_copy[col_base], color='k',linewidth=0.45)
        plt.fill_between(df_copy['X']/1000, df_copy[col_base], df_copy[col_top], facecolor=color, alpha=0.5, label=f'Layer {int(col_base)+1}')

    plt.title(f'{timestep} MA before present')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)')
    _, upper_y = plt.ylim()
    plt.ylim(-50, max_depth)
    plt.legend(loc='lower left')
    plt.grid(False)
    plt.gca().invert_yaxis()
    plt.show()
    
    if last_file:  
        plt.vlines(x=well_loc, ymin=0, ymax=well_length, colors='k', linestyles='solid')
        plt.plot(well_loc, -10, marker='X', markersize=8, color='k', label=well_name)
        plt.legend(loc='lower left')
        
    fig_dir =fig_dir+f'For Timestep {timestep}'+'.png'
    save_plot(fig_dir)

times = 0
for idx, file_name in enumerate(file_list):
    file_path = os.path.join(direct, file_name)
    next_file = pd.read_csv(os.path.join(direct, file_list[idx+1])).columns if idx+1 < len(file_list) else None
    last_file = (idx == len(file_list) - 1)  # Check if it's the last file
    plot_decompaction(file_path, times, next_file, last_file,fig_dir)
    times = times+1

#%% ##Creating an animation of the decompaction

def animation(file_idx, ax):
    df = pd.read_csv(os.path.join(direct, file_list[file_idx]))

    zero_cols = df.columns[df.eq(0).all()]
    df = df.drop(columns=zero_cols)
    
    columns = df.columns[1:]
    
    ax.clear()
    df_copy = df.copy()
    cols_to_consider = columns[::-1]
    max_m = len(cols_to_consider)
    
    for i, col in enumerate(cols_to_consider):
        nan_mask = df[col].isnull()
        df_copy.loc[nan_mask, col] = np.nan
        m = max_m - i 
        color = cmap(m)
        ax.plot(df_copy['X']/1000, df_copy[col], label=col.split(" - ")[0],color=color)
    
    if file_idx + 1 < len(file_list):  
        next_df = pd.read_csv(os.path.join(direct, file_list[file_idx + 1]))
        zero_cols_next = next_df.columns[next_df.eq(0).all()]
        next_df = next_df.drop(columns=zero_cols_next)
        next_columns = next_df.columns[1:]
        ax.axhline(y=0, color='k', linestyle='--', label=next_columns[-1].split(" - ")[0])  
        
    timestep = int(file_list[file_idx].split('at_')[1].split('.csv')[0])
    ax.set_title(f'Timestep {timestep}')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (m)')
    plt.ylim(-50, max_depth)
    ax.legend()
    ax.grid(False)
    ax.invert_yaxis()

# fig, ax = plt.subplots()
# ani = FuncAnimation(fig, animation, frames=range(len(file_list)), fargs=(ax,), interval=1000, repeat=False)
# plt.show()