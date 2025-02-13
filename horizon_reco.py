import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import time
import warnings

import main as inp

#%%

fig = inp.fig

max_depth = inp.max_depth  
max_distance = inp.max_distance 

polyorder = inp.polyorder

well_name = inp.well_name
well_loc = inp.well_loc
well_length = inp.well_length

max_depth = max_depth*1000
well_length=well_length*1000
max_distance = max_distance*1000

warnings.filterwarnings("ignore")
start_time = time.time()
#%%     

def save_plot(filename):
    if inp.save_images:
        plt.savefig(filename,bbox_inches='tight',dpi=600)
#%%        
## Input the image 
def run():
    image_path = f"input//{fig}.png"
    image = cv2.imread(f"input//{fig}.png")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError("Error: Image not found.")
    
    ## Doing HSV (Hue, Saturation, Value (Brightness)) conversion and masking to detect the blue colours in the image
    print("Horizon dectection and cleaning")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    lower_blue = np.array([100, 50, 50])        # Implementing the ranges of the blue colour
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    ## Cleaning the mask from the noise 
    
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    cleaned_mask_blue = cv2.morphologyEx(cleaned_mask_blue, cv2.MORPH_CLOSE, kernel)
    
    ## Detecting the lines through Hough transform
    
    lines = cv2.HoughLinesP(cleaned_mask_blue, 1, np.pi/180, threshold=3, minLineLength=3, maxLineGap=3)
    
    ## Filtering the lines. Lines shorter than the length_threshold is filtered out
    
    length_threshold = 1
    
    filtered_lines = [line for line in lines if abs(line[0][2] - line[0][0]) > length_threshold]
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
     ## Drawing detected lines on top of the original image
    
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
    #     plt.plot([x1, x2], [y1, y2], color='green', linewidth=2)
    #     # print(x1,y1,x2,y2)
    # plt.show()
    
    #%% ## The mid-points of the detected lines are calculated. 
        ## These are then normalized to give a representation of scale of the image.
        ## The maximum depth (scale) and the maximum distance of the seismic line in the image is given in Meters. 
    
    depths = [(line[0][1] + line[0][3]) / 2 for line in filtered_lines]
    img_width = [(line[0][0] + line[0][2]) / 2 for line in filtered_lines]
    
    image_height = image.shape[0]
    normalized_depths = [(depth / image_height) * max_depth for depth in depths]
    
    image_width = image.shape[1]
    normalized_x_positions = [(x_pos / image_width) * max_distance for x_pos in img_width]
    
    ## Plotting the extracted horizons 
    
    # plt.figure(figsize=(10, 6))
    # for line in filtered_lines:
    #     x1, y1, x2, y2 = line[0]
    #     # plt.plot([x1, x2], [(y1 / image.shape[0]) * max_depth, (y2 / image.shape[0]) * max_depth], color='blue')
    #     plt.plot([(x1 / image.shape[1]) * max_distance, (x2 / image.shape[1]) * max_distance], 
    #              [(y1 / image.shape[0]) * max_depth, (y2 / image.shape[0]) * max_depth], color='blue')
    # plt.ylim(max_depth, 0)
    # plt.xlabel('Distance (km)')
    # plt.ylabel('Depth (m)')
    # plt.title('Extracted Horizons')
    # plt.grid(True)
    # plt.show()
    # save_plot(f"result//figure//{fig}//Extracted_horizons.png")
    
    #%% ## For splitting the horizons based on connected components
    
    num_labels, labeled_mask = cv2.connectedComponents(cleaned_mask_blue)
    all_lines = []
    
    for label in range(1, num_labels):  # Starting from 1 to ignore the background label 0
        # Create an image with only current label
        isolated = np.where(labeled_mask == label, 255, 0).astype(np.uint8)
    
        # Extract lines from this isolated image
        isolated_lines = cv2.HoughLinesP(isolated, 1, np.pi/180, threshold=3, minLineLength=3, maxLineGap=3)

        if isolated_lines is not None:
            all_lines.extend(isolated_lines)
            
    # print(f"Number of points in the lines: {len(all_lines)}")
    
    ## Isolating each horizons
    # print("Isolating each horizons")
    
    lines_by_component = {}
    for label in range(1, num_labels):  # 0 --> Background label. Starting from 1 to ignore the backgrond label
        # Create an image with only current label
        isolated = np.where(labeled_mask == label, 255, 0).astype(np.uint8)
    
        # Extract lines from this isolated image
        isolated_lines = cv2.HoughLinesP(isolated, 1, np.pi/180, threshold=3, minLineLength=3, maxLineGap=3)
        
        if isolated_lines is not None:
            all_lines.extend(isolated_lines)
            lines_by_component[label] = isolated_lines
    
    # Printing the number of detected lines
    print(f"Number of component labels/Horizons detected: {len(lines_by_component)}")
    
    def get_unique_colors(n):
        colors = plt.cm.jet(np.linspace(0, 1, n))
        return colors
    
    colors = get_unique_colors(len(lines_by_component))
    
    plt.figure(figsize=(10, 6))
    
    for index, (label, lines) in enumerate(lines_by_component.items()):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 = (x1 / image.shape[1]) * max_distance
            x2 = (x2 / image.shape[1]) * max_distance
            y1 = (y1 / image.shape[0]) * max_depth
            y2 = (y2 / image.shape[0]) * max_depth
            plt.plot([x1, x2], [y1, y2], color=colors[index], linewidth=2)
    
    # plt.title('Lines Grouped by Components')
    # plt.xlabel('Distance (km)')
    # plt.ylabel('Depth (m)')
    # # plt.xlim(0, max_distance+100)  # Adjusting the x-axis to fit the image width
    # plt.ylim(-50.0, max_depth+100)  # Adjusting the y-axis to fit the image height and flipping it to match image coordinates
    # plt.gca().invert_yaxis() 
    # plt.show()
    # save_plot(f"result//figure//{fig}//grouping_by_component_splitting.png")
    
    #%%
    # file_count = 0
    # counter = 0
    
    n = len(inp.lith)  
    tot_hor = len(lines_by_component)
    
    if n == tot_hor:
        
        print("Extracting the horizon and saving the coordinates as CSV")
        
        for label, lines in lines_by_component.items():
            component_points_df = pd.DataFrame(columns=["Component", "X", "Y"])
        
            for line in lines:
                x1, y1, x2, y2 = line[0]
                num_points = min(100, abs(x2 - x1) // 2) # Number of points to interpolate along the line
                x_values = np.linspace(x1, x2, num_points)
                y_values = np.linspace(y1, y2, num_points)
        
                x_normalized = (x_values / image.shape[1]) * max_distance
                y_normalized = (y_values / image.shape[0]) * max_depth
        
                for x, y in zip(x_normalized, y_normalized):
                    component_points_df = component_points_df.append({"Component": label, "X": x, "Y": y}, ignore_index=True)
                
            sorted_df = component_points_df.sort_values(by="X")
            filename = f'result//csv//{fig}//line_points_component_{label}.csv'
            output_dir = os.path.dirname(filename) 
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir) 
            sorted_df.to_csv(filename, index=False)
            end_time1 = time.time()
            elapsed_time1 = end_time1 - start_time
            elapsed_time1 = elapsed_time1 / 60
        
            # print(f'Saved Coordinates for Horizon {label} of {tot_hor}')
            
    #%% Cubic Spline interpolation
        from scipy.interpolate import CubicSpline

        def cubic_spline_interpolate(x, y):
          
            cs = CubicSpline(x, y, bc_type='natural')
            x_new = np.linspace(np.min(x), np.max(x), num=len(x)*10)
            y_new = cs(x_new)
            return x_new, y_new
        
        #%% ## Remove duplicates, Extrapolate the values in between and Smoothen the data using Savgol filter
        
        # print("Removing duplicates and applying the filter")
        def data_smooth(filename, hor_no, tot_hor, polyorder):
            data = pd.read_csv(filename)
           
            data = data.drop_duplicates(subset=['X'])
            data['X'] = data['X'].round().astype(int)
            full_range = pd.DataFrame({'X': range(data['X'].min(), data['X'].max() + 1)})
            data = full_range.merge(data, on='X', how='left').interpolate()
            
            data = data[['X', 'Y']]
            
            window_size = max(5, len(data) // 20)  
            if window_size % 2 == 0:
               window_size += 1
                
            data['Smooth_Y'] = savgol_filter(data['Y'], window_size, polyorder,mode='interp')
        
            new_name = filename.replace(f'line_points_component_{hor_no}.csv', f'line_points_component_{hor_no}_smoothed.csv')
        
            data[['X', 'Smooth_Y']].to_csv(new_name, index=False)
            
            # plt.scatter(data['X'], data['Y'], label='Original Data', color='blue', s=5)
            # plt.plot(data['X'], data['Smooth_Y'], label='Smoothed Data', color='red', linewidth=1)
            # plt.title('Original and Smoothed Data Comparison')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.legend()
            # plt.show()
        
        for label in range(1, tot_hor+1):
            filename = f'result/csv/{fig}/line_points_component_{label}.csv'
            data_smooth(filename, label,tot_hor,polyorder)
        #%% ## Plotting the figure
        plt.figure(figsize=(10, 8))
        def horizon_smoothening(filename):
            data = pd.read_csv(filename.replace('.csv', '_smoothed.csv'))
            plt.plot(data['X']/1000, data['Smooth_Y'], label=f"Smoothed Horizon {filename.split('_')[-1].split('.')[0]}")
            plt.vlines(x=well_loc, ymin=0, ymax=well_length, colors='k', linestyles='solid')
            plt.title('Smoothed Horizon')
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (m)')
            plt.ylim(-50.0, max_depth+100) 
            
        def plot_well_marker():
         plt.plot(well_loc, -10, marker='X', markersize=8, color='k', label=well_name)
         
        #%%
        # Iterating over files and plotting smoothed horizons and saving the image
        for label, _ in lines_by_component.items():
            filename = f'result//csv//{fig}//line_points_component_{label}.csv'
            file_to_plot = filename.replace('.csv', '_smoothed.csv')
        
            if os.path.exists(file_to_plot):
                horizon_smoothening(filename)
        plot_well_marker()
        plt.legend()
        plt.gca().invert_yaxis() 
        plt.show()
        save_plot(f"result//figure//{fig}//Smoothed Horizon.png")
        #%%
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time / 60
        
        # print(f"Total: {elapsed_time:.2f} minutes")
        
    else:
        raise ValueError(f"Mismatch: Expected {tot_hor} horizons, but found {n} layers. Please check your inputs.")
        
run()