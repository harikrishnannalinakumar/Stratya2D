# %%

## This section implements the backstripping algorithm based on the equation from Allen and Allen (2005).

import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

import main as inp
import os 

well = inp.well

# %%
def save_plot(filename):
    if inp.save_images:
        output_dir = os.path.dirname(filename)
        if not os.path.exists(output_dir):
         os.makedirs(output_dir)
        
        plt.savefig(filename,bbox_inches='tight',dpi=600)
#%%

lith =    inp.lith
               
poro =    inp.poro       
const =   inp.const        
c = np.divide(const, 1000)      # Changing the unit of Compaction coefficient to m-1

density = inp.density              

decomp_depth_age_start = inp.decomp_depth_age_start                                   # Deposition age of unit 1
age = inp.age 
y1 =  inp.y1
y2 =  inp.y2 
d1=np.add(y1,y2)/2
int_guess = d1

paleo =     inp.paleo     
sea_level = inp.sea_level
  
density_water = inp.density_water               # Desnity of water
density_mantle = inp.density_mantle              # Density of mantle

poro_lower_limits = inp.poro_lower_limits
poro_upper_limits = inp.poro_upper_limits
      
const_lower_limits = inp.const_lower_limits
const_upper_limits = inp.const_upper_limits

density_lower_limits = inp.density_lower_limits
density_upper_limits = inp.density_upper_limits

paleo_upper_limits = inp.paleo_upper_limits
paleo_lower_limits = inp.paleo_lower_limits

sea_level_lower_limits = inp.sea_level_lower_limits
sea_level_upper_limits = inp.sea_level_upper_limits


n = (len(lith))                   # Finding the number of units in the model
epsilon0 = 1e-6                 # Defining an error to check for value convergence
b_dep = 0.0                           # Bottom layer of the previous layer. Required to calculate the total column thickness


# %%
def var_upp_low_limits(original_values, lower_limits, upper_limits):
    varied_values = []
    for value, lower_limit, upper_limit in zip(original_values, lower_limits, upper_limits):
        if lower_limit is not None and upper_limit is not None:
            varied_value = np.random.uniform(lower_limit, upper_limit)
            varied_values.append(varied_value)
        else:
            varied_values.append(value)  # Keep original value if limits are not specified
    return varied_values

# %%
def filter():
    ## For filtering and saving the data
    df = pd.read_csv(f'result/{well}/decompacted_thick.csv')

    df['k'] = pd.to_numeric(df['k'], errors='coerce').fillna(0)

    max_i = df["i"].max()

    filtered_df = df[(df["i"].isin(range(0, max_i + 1))) & (df["k"] == 0)][["i", "tectonic_subsidence"]]
    
    filtered_df_sorted = filtered_df.sort_values(by=['i'], ascending=True)

    filtered_df_sorted.to_csv(f'result//{well}//filtered_tectonic_subsidence.csv', index=False)
    df1 = df.copy()
    
    averages = df1.groupby(['i', 'k']).mean()
    max_values = df1.groupby(['i', 'k']).max()
    min_values = df1.groupby(['i', 'k']).min()
    
    avg_file = 'result/monte_carlo/averages/averages.csv'
    output_dir = os.path.dirname(avg_file)
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)
     
    averages.to_csv(avg_file)
    
    min_file = 'result/monte_carlo/min values/min_values.csv'
    output_dir = os.path.dirname(min_file)
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)
     
    min_values.to_csv(min_file)
    
    max_file = 'result/monte_carlo/max values/max_values.csv'
    output_dir = os.path.dirname(max_file)
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)
    
    max_values.to_csv(max_file)
   
    avg = pd.read_csv('result/monte_carlo/averages/averages.csv')
    avg['k'] = pd.to_numeric(avg['k'], errors='coerce').fillna(avg['i'])

    max_df = pd.read_csv('result/monte_carlo/max values/max_values.csv')
    max_df['k'] = pd.to_numeric(max_df['k'], errors='coerce').fillna(max_df['i'])

    min_df = pd.read_csv('result/monte_carlo/min values/min_values.csv')
    min_df['k'] = pd.to_numeric(min_df['k'], errors='coerce').fillna(min_df['i'])

    for header in avg.columns[2:]:
        pivoted_df = avg.pivot(index='k', columns='i', values=header)
        pivoted_df = pivoted_df.iloc[::-1]
        pivoted_df = pivoted_df.fillna(0)
        pivoted_df.to_csv(f'result/monte_carlo/averages/avg_val_{header}.csv')

    for header in max_df.columns[2:]:
        pivoted_df = max_df.pivot(index='k', columns='i', values=header)
        pivoted_df = pivoted_df.iloc[::-1]
        pivoted_df = pivoted_df.fillna(0)
        pivoted_df.to_csv(f'result/monte_carlo/max values/max_val_{header}.csv')

    for header in min_df.columns[2:]:
        pivoted_df = min_df.pivot(index='k', columns='i', values=header)
        pivoted_df = pivoted_df.iloc[::-1]
        pivoted_df = pivoted_df.fillna(0)
        pivoted_df.to_csv(f'result/monte_carlo/min values/min_val_{header}.csv')

    average_decomp_thickness = pd.read_csv('result/monte_carlo/averages/avg_val_Decompacted thickness.csv', index_col=0)
    max_decomp_thickness = pd.read_csv('result/monte_carlo/max values/max_val_Decompacted thickness.csv', index_col=0)
    min_decomp_thickness = pd.read_csv('result/monte_carlo/min values/min_val_Decompacted thickness.csv', index_col=0)

    def zero_check(row):
        # print('Check')
        return row.eq(0).all() 
    
    zero_row = []
    for row_index, row in average_decomp_thickness.iterrows():
        if zero_check(row):
            zero_row.append(row_index)
            print(row_index)

    for i in range(1,len(average_decomp_thickness)):
        for j in range(len(average_decomp_thickness.columns)):
            if i == 0 and j == 0:
                continue
            average_decomp_thickness.iloc[i, j] += average_decomp_thickness.iloc[max(i - 1, 0), j]
            max_decomp_thickness.iloc[i, j] += max_decomp_thickness.iloc[max(i - 1, 0), j]
            min_decomp_thickness.iloc[i, j] += min_decomp_thickness.iloc[max(i - 1, 0), j]

    average_decomp_thickness.loc[zero_row] = 0
    max_decomp_thickness.loc[zero_row] = 0
    min_decomp_thickness.loc[zero_row] = 0
    
    average_decomp_thickness.to_csv('result/monte_carlo/averages/Decompacted_thickness_avg.csv')
    df = pd.read_csv('result/monte_carlo/averages/Decompacted_thickness_avg.csv')
    df.columns = [''] + age
    df.to_csv('result/monte_carlo/averages/Decompacted_thickness_avg.csv',index=False)
    
    max_decomp_thickness.to_csv('result/monte_carlo/max values/Decompacted_thickness_max.csv')
    df = pd.read_csv('result/monte_carlo/max values/Decompacted_thickness_max.csv')
    df.columns = [''] + age
    df.to_csv('result/monte_carlo/max values/Decompacted_thickness_max.csv',index=False)
    
    min_decomp_thickness.to_csv('result/monte_carlo/min values/Decompacted_thickness_min.csv')
    df = pd.read_csv('result/monte_carlo/min values/Decompacted_thickness_min.csv')
    df.columns = [''] + age
    df.to_csv('result/monte_carlo/min values/Decompacted_thickness_min.csv',index=False)

# %%
def cal_plot_rates():

    df1 = pd.read_csv(f'result/{well}/tect_sub_rate_sorted.csv')
    age_value = int(decomp_depth_age_start[0])
    new_row = {'age': age_value, 'max_subsidence': 0.0, 'min_subsidence': 0.0, 'mean_subsidence': 0.0}
    df1 = df1.append(new_row, ignore_index=True)
    df1.to_csv(f'result/{well}/tect_sub_rate_sorted_modified.csv', index=False)
    
    df = pd.read_csv(f'result/{well}/tect_sub_rate_sorted_modified.csv')
   
    df = df.sort_values(by='age', ascending=True)

    mean_rates = []
    max_rates = []
    min_rates = []

    for i in range(len(df) - 1):
        age_diff = abs(df.iloc[i]['age'] - df.iloc[i+1]['age'])
        mean_rate = (df.iloc[i]['mean_subsidence'] - df.iloc[i+1]['mean_subsidence']) / age_diff
        max_rate = (df.iloc[i]['max_subsidence'] - df.iloc[i+1]['max_subsidence']) / age_diff
        min_rate = (df.iloc[i]['min_subsidence'] - df.iloc[i+1]['min_subsidence']) / age_diff

        mean_rates.append(mean_rate)
        max_rates.append(max_rate)
        min_rates.append(min_rate)

    rates_df = pd.DataFrame({
        'age': df['age'].iloc[:-1],
        'mean_rate': mean_rates,
        'max_rate': max_rates,
        'min_rate': min_rates
    })

    rates_df.to_csv(f'result/{well}/tect_sub_rate_calculated.csv', index=False)

def plot_rates():

    df = pd.read_csv(f'result/{well}/tect_sub_rate_calculated.csv')
    df['age'] = df['age'].astype(int)
    df['error'] = abs(df['max_rate'] - df['min_rate'])

    plt.figure(figsize=(10, 6))
    tect_sub_label = False    
    
    for i in range(len(df) - 1):
            label = 'Rate of Tectonic subsidence' if not tect_sub_label else None
            tect_sub_label = True
            plt.plot(df['age'][i:i+2], df['mean_rate'][i:i+2], color='black', marker='o',label=label)

    plt.errorbar(df['age'], df['mean_rate'], yerr=df['error'], fmt='o', color='black', ecolor='red', capsize=5)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.75)

    plt.title('Rate of Tectonic subsidence')
    plt.xlabel('Age (Ma)')
    plt.ylabel('Mean Rate (m/Ma)')
    # plt.grid(True) 

    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right')

    save_plot(f'result//figure//rates_{well}.png')
    plt.show()

# %%
def calculate_min_max():
    # Calculating the minimum and maximum values for the limits in the plot
    filtered_df = pd.read_csv(f'result/{well}/filtered_tectonic_subsidence.csv')
    filtered_df = filtered_df[filtered_df['i'].isin(range(len(age)))]
    filtered_df['age'] = filtered_df['i'].apply(lambda x: age[x])
    filtered_df.drop('i', axis=1, inplace=True)

    max_values = filtered_df.groupby('age')['tectonic_subsidence'].max().reset_index(name='max_subsidence')
    min_values = filtered_df.groupby('age')['tectonic_subsidence'].min().reset_index(name='min_subsidence')
    mean_values = filtered_df.groupby('age')['tectonic_subsidence'].mean().reset_index(name='mean_subsidence')

    extremes_df = pd.merge(max_values, min_values, on='age')
    extremes_df = pd.merge(extremes_df, mean_values, on='age')

    return extremes_df

def sort_filtered():
    df = pd.read_csv(f'result/{well}/tect_sub_rate.csv')
    df['age'] = df['age'].replace('0', '-1').astype(int)
    df = df.sort_values(by='age')
    df['age'] = df['age'].replace(-1, '0').astype(str)
    df.to_csv(f'result/{well}/tect_sub_rate_sorted.csv', index=False)    

def new_plot():
    df = pd.read_csv(f'result/{well}/tect_sub_rate_sorted.csv')
    df['age'] = df['age'].astype(int)
    df['error'] = df['max_subsidence'] - df['min_subsidence']
    df['error_lower'] = df['mean_subsidence'] - df['min_subsidence']
    df['error_upper'] = df['max_subsidence'] - df['mean_subsidence']

    age_list = list(map(int, age))
    plt.figure(figsize=(10, 6))

    tect_sub_label = False    
    for i in range(len(df) - 1):
            label = 'Tectonic Subsidence' if not tect_sub_label else None
            tect_sub_label = True
            plt.plot(df['age'][i:i+2], df['mean_subsidence'][i:i+2], color='black', marker='o',label=label)
            
    plt.errorbar(df['age'], df['mean_subsidence'], yerr=[df['error_lower'], df['error_upper']], fmt='o', color='black', ecolor='red', capsize=5)
    
    plt.title('Tectonic Subsidence plot')
    plt.xlabel('Age (Ma)')
    plt.ylabel('Tectonic Subsidence (m)')

    plt.legend()
    plt.gca().invert_xaxis() 
    plt.gca().invert_yaxis()
    save_plot(f"result//figure//tect_sub_{well}.png")
    plt.show()

# %%
def monte_first_decomp(i, y1, y2, varied_poro, varied_const, b_dep,guess,bulk_density,paleo_depth,eustatic_sea_level):
    c = varied_const[i]/1000
    density = bulk_density[i]
    # Calculate d_thick
    d_thick = (y2[i] - y1[i]) - (varied_poro[i] / c) * (math.exp(-c * y1[i]) - math.exp(-c * y2[i])) + (varied_poro[i] / c) * (math.exp(-c * b_dep) - math.exp(-c * guess)) + b_dep
    t = 0 
    epsilon = abs(guess - d_thick)
    # print('guess',guess)
    while epsilon > epsilon0:
        t += 1
        guess = d_thick
        d_thick = (y2[i] - y1[i]) - (varied_poro[i] / c) * (math.exp(-c * y1[i]) - math.exp(-c * y2[i])) + (varied_poro[i] / c) * (math.exp(-c * b_dep) - math.exp(-c * guess)) + b_dep
        epsilon = abs(guess - d_thick)
        if t > 1e6: 
            raise Exception("Limit of iterations exceeded. Epsilon = %f" % epsilon)  
    # Calculate poronew based on d_thick
    poronew = (varied_poro[i] / c) * ((math.exp(-c * b_dep) - math.exp(-c * d_thick)) / (d_thick - b_dep))
    bulk_den = (poronew * density_water) + ((1 - poronew) * density)
    tot_den = bulk_den * d_thick/1000
    if(i==0):
        col_thickness = d_thick/1000
        bulk_den_col = bulk_den * col_thickness/col_thickness
        tect_sub = tectonic_subsidence(col_thickness, bulk_den_col, i, paleo_depth,eustatic_sea_level)
    else:
        col_thickness = 0
        bulk_den_col = 0
        tect_sub = 0
    return round(d_thick, 3), round(poronew, 2), round(bulk_den, 2), round(col_thickness, 3), round(bulk_den_col, 2), round(tot_den, 3), round(tect_sub,3)

def monte_second_decomp(curr_row_no, porosity, varied_const, y1, y2, initial_b_dep,den, bulk_density,paleo_depth,eustatic_sea_level):
    k = curr_row_no
    b_dep = initial_b_dep
    prev_lyr_den = den 
    tot_den = 0
    density = bulk_density
    results = []
    varied_poro = porosity

    for i in reversed(range(k)):
        c = varied_const[i]/1000
        # print('second',k,i)
        # print('c',c)
        guess = int_guess[i]
        if y2[i] == y1[i]:  # Checking for unconformity/erosion units
            d_thick = 0.0
            porosity[i] = 0.0
        else:
        # Skip the layer if it's a hiatus layer
            d_thick = (y2[i] - y1[i]) - (varied_poro[i] / c) * (math.exp(-c * y1[i]) - math.exp(-c * y2[i])) + (varied_poro[i] / c) * (math.exp(-c * b_dep) - math.exp(-c * guess)) + b_dep
            t = 0 
            epsilon = abs(guess - d_thick)
            while epsilon > epsilon0:
                t += 1
                guess = d_thick
                d_thick = (y2[i] - y1[i]) - (varied_poro[i] / c) * (math.exp(-c * y1[i]) - math.exp(-c * y2[i])) + (varied_poro[i] / c) * (math.exp(-c * b_dep) - math.exp(-c * guess)) + b_dep
                epsilon = abs(guess - d_thick)
                if t > 1e6: 
                    raise Exception("Limit of iterations exceeded. Epsilon = %f" % epsilon)  
            yt = d_thick

                # Calculate poronew based on d_thick
            poronew = (varied_poro[i] / c) * ((math.exp(-c * b_dep) - math.exp(-c * yt)) / (yt - b_dep))
            bulk_den = (poronew * density_water) + ((1 - poronew) * density[i])
            d_thick= (d_thick - b_dep)
            b_dep= (b_dep + d_thick)
            col_thickness = b_dep/1000
            tot_den = tot_den + (bulk_den * d_thick/1000)

            if(i==0):
                total_lyr_den = tot_den + prev_lyr_den
                sum = total_lyr_den/col_thickness
                bulk_den_col = sum
                tect_sub = tectonic_subsidence(col_thickness, bulk_den_col, k,paleo_depth,eustatic_sea_level)
            else:
                bulk_den_col = 0
                tect_sub = 0

            d_thick = np.nan_to_num(d_thick)
            poronew = np.nan_to_num(poronew)
            bulk_den = np.nan_to_num(bulk_den)
            b_dep = np.nan_to_num(b_dep)
            col_thickness = np.nan_to_num(col_thickness)
            result = (curr_row_no, i, round(varied_poro[i],5), round(c,3), round(d_thick,3), round(poronew,3), round(bulk_den,3), round(b_dep), round(col_thickness,3), round(bulk_den_col,3), round(tect_sub,3))      
            results.append(result)
    return results

def tectonic_subsidence(col_thick, bulk_den_col,i,paleo_depth,eustatic_sea_level ):
    sea_level = eustatic_sea_level
    paleo = paleo_depth
    sea_level = eustatic_sea_level
    paleo = paleo_depth
    tect_subs = ((col_thick*1000)*(density_mantle-bulk_den_col)/(density_mantle-density_water))-(sea_level[i+1]*(density_water/(density_mantle-density_water)))+(paleo[i+1]-sea_level[i+1])
    return tect_subs

# %%

def de():
    num_sim = 1000 ## Number of sims for Monte carlo
    no_of_layers = n
    all_results = []
    d_thick = 0

    array_y1 = np.array(y1)
    array_y2 = np.array(y2)

    for sim in range(num_sim):
        varied_poro = var_upp_low_limits(poro, poro_lower_limits, poro_upper_limits)
        array_poro = np.array(varied_poro)
        varied_const = var_upp_low_limits(const, const_lower_limits, const_upper_limits)
        array_varied_const = np.array(varied_const)
        varied_density = var_upp_low_limits(density, density_lower_limits, density_upper_limits)
        array_varied_density = np.array(varied_density)
        varied_paleo = var_upp_low_limits(paleo, paleo_lower_limits, paleo_upper_limits)
        array_varied_paleo = np.array(varied_paleo)
        varied_sea_level = var_upp_low_limits(sea_level, sea_level_lower_limits, sea_level_upper_limits)
        array_varied_sea_level = np.array(varied_sea_level)
        i = 0

        for i in range(no_of_layers):
            guess = int_guess[i]
            if (y2[i] == y1[i]):  # Skipping the erosional layers
                continue
            else:
                d_thick, poronew, bulk_den, col_thick, bulk_col, tot_den, tect_sub = monte_first_decomp(i, array_y1, array_y2, array_poro, array_varied_const ,b_dep, guess,array_varied_density,array_varied_paleo,array_varied_sea_level)
                all_results.append([i, i , varied_poro[i], varied_const[i] ,d_thick, poronew, bulk_den, b_dep, col_thick, bulk_col, tect_sub])
            if (i!=0):
                second_decomp_results = monte_second_decomp(i, array_poro, array_varied_const, array_y1, array_y2,d_thick, tot_den,array_varied_density,array_varied_paleo,array_varied_sea_level)       
                all_results.extend(second_decomp_results)  # Add all the results from the second function
   
    df_results = pd.DataFrame(all_results, columns=['i','k','Varied Poro', 'Varied Const', 'Decompacted thickness','Poro new','Bulk_density','b_dep','col_thickness','Bulk density col','tectonic_subsidence'])

    csv_filename = f'result/{well}/decompacted_thick.csv'  
   
    output_dir = os.path.dirname(csv_filename)
    if not os.path.exists(output_dir):
     os.makedirs(output_dir)
    
    df_results.to_csv(csv_filename, index=False)

    filter()

# %%
de()
extremes_df = calculate_min_max()

extremes_df.to_csv(f'result/{well}/tect_sub_rate.csv', index=False)
sort_filtered()
new_plot()

# %%
cal_plot_rates()
plot_rates()


