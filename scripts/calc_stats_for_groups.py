#!/usr/bin/env python

"""calc_stats_for_groups.py: Helper script for aligning climate variables from different sources/instrumentation."""

__author__      = "Fraser King"
__group__       = "University of Michigan"

### IMPORTS
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import xarray as xr
from datetime import timedelta

### GLOBALS
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 15})
palette = ['red', '#cc5500', 'blue', 'purple', '#e8b400', '#009B76', 'gray', 'black']

# LOAD DATA
df = pd.read_csv('../data/pca_group_analysis/finalized_combo.csv')
df = df[df['site'] == 'FIN']
df['time'] = pd.to_datetime(df['time'])

def calc_group_stats(folder_path, era5_path):
    def reshape_to_5min_intervals(nc_file_path):
        nc_data = xr.open_dataset(nc_file_path)
        nc_data = nc_data[['time', 'Zh', 'v', 'width', 'temperature', 'relative_humidity', 'pressure']]
        nc_resampled = nc_data.resample(time='5Min').mean()
        return nc_resampled

    def find_cloud_top_height(zh_column):
        for i, value in enumerate(reversed(zh_column)):
            if value > -40:
                return len(zh_column) - i
        return np.nan

    eof1_list, eof2_list, eof3_list, group_list = [], [], [], []
    date_list, temperature_list, humidity_list, pressure_list = [], [], [], []
    near_surf_reflect_list, reflect_column_list, dv_column_list, sw_column_list, cloud_top_height_list = [], [], [], [], []
    t_column_list, rh_column_list = [], []

    # Load era5 data
    era5_data = xr.open_dataset(era5_path)

    # Main processing loop
    paths = glob.glob(f'{folder_path}/*.nc')
    for path in paths:
        print("Working on path", path)

        resampled_nc_data = reshape_to_5min_intervals(path)

        # Iterate over each time step in the NetCDF file
        for nc_time in resampled_nc_data['time'].values:
            nc_datetime = pd.to_datetime(str(nc_time))

            # ERA5
            target_date =  np.datetime64(nc_time)
            time_differences = abs(era5_data['time'] - target_date)
            closest_index = time_differences.argmin().item()
            closest_time_difference = time_differences.isel(time=closest_index).values

            era5_t_data = np.nan
            era5_q_data = np.nan
            if closest_time_difference / 3600000000000 <= 1:
                era5_t_data = era5_data['t'][closest_index][:].isel(latitude=2, longitude=2).values
                era5_q_data = era5_data['r'][closest_index][:].isel(latitude=2, longitude=2).values
            t_column_list.append(era5_t_data)
            rh_column_list.append(era5_q_data)

            # Finding the closest row in the DataFrame
            closest_row = df.iloc[(df['time'] - nc_datetime).abs().argsort()[:1]]

            date_list.append(nc_datetime)
            if not closest_row.empty and abs(nc_datetime - pd.to_datetime(closest_row['time'].values[0])) <= timedelta(minutes=5):
                eof1_list.append(closest_row['eof1'].values[0])
                eof2_list.append(closest_row['eof2'].values[0])
                eof3_list.append(closest_row['eof3'].values[0])
                group_list.append(closest_row['group'].values[0])

                # Extracting additional variables from NetCDF
                idx = np.where(resampled_nc_data['time'].values == nc_time)[0][0]
                temperature_list.append(resampled_nc_data['temperature'][idx].values)
                humidity_list.append(resampled_nc_data['relative_humidity'][idx].values)
                pressure_list.append(resampled_nc_data['pressure'][idx].values)

                # Extracting reflectivity values
                zh_values = resampled_nc_data['Zh'][idx].values
                dv_values = resampled_nc_data['v'][idx].values
                sw_values = resampled_nc_data['width'][idx].values
                near_surf_reflect_list.append(zh_values[0] if len(zh_values) > 0 else np.nan)
                reflect_column_list.append(zh_values)
                dv_column_list.append(dv_values)
                sw_column_list.append(sw_values)
                cloud_top_height_list.append(find_cloud_top_height(zh_values))
            else:
                # Append NaNs if no matching time or outside the time window
                eof1_list.append(np.nan)
                eof2_list.append(np.nan)
                eof3_list.append(np.nan)
                group_list.append(np.nan)
                temperature_list.append(np.nan)
                humidity_list.append(np.nan)
                pressure_list.append(np.nan)
                near_surf_reflect_list.append(np.nan)
                reflect_column_list.append(np.nan)
                dv_column_list.append(np.nan)
                sw_column_list.append(np.nan)
                cloud_top_height_list.append(np.nan)
        
        print(len(temperature_list))
        print(len(eof1_list))
        print(len(near_surf_reflect_list))
        print(len(reflect_column_list))
        print(len(t_column_list))

    data = {
        'Date': date_list,
        'EOF1': eof1_list,
        'EOF2': eof2_list,
        'EOF3': eof3_list,
        'Group': group_list,
        'Temperature': temperature_list,
        'Relative Humidity': humidity_list,
        'Pressure': pressure_list,
        'Near Surface Reflectivity': near_surf_reflect_list,
        'Reflectivity Column': reflect_column_list,
        'DV Column': dv_column_list,
        'SW Column': sw_column_list,
        'T Column': t_column_list,
        'RH Column': rh_column_list,
        'Cloud Top Height': cloud_top_height_list
    }

    return pd.DataFrame(data)

# Main runloop
if __name__ == '__main__':
    df_stats = calc_group_stats('../data/Finland/', '../data/pca_era5/fin/era5_fin.nc')
    df_stats.dropna(inplace=True)

    print("All done!")