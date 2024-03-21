#!/usr/bin/env python

"""pca_pip_only_preprocess.py: Helper script for extracting relevant microphysical variables from PIP L3 data."""

__author__      = "Fraser King"
__group__       = "University of Michigan"

### IMPORTS
import glob, os
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

### GLOBALS
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.rcParams.update({'font.size': 15})

# calc_various_pca_inputs: Helper function for extracting relevant microphysics variables from PIP L3 at different sites
def calc_various_pca_inputs():
    sites = ['MQT', 'FIN', 'HUR', 'HAUK', 'KIS', 'KO1', 'KO2', 'IMP', 'APX', 'NSA', 'YFB']
    inst = ['006', '004', '008', '007', '007', '002', '003', '003', '007', '010', '003']

    # Path to covnerted files (you can download these on DeepBlue)
    pip_path = '../pip_processing/data/converted/'
        
    pip_dates = []
    for file in glob.glob(os.path.join(pip_path, '**', 'edensity_distributions', '*.nc'), recursive=True):
        pip_dates.append(file[-15:-7])

    site_array = []
    N_0_array = []
    lambda_array = []
    total_particle_array = []
    avg_ed_array = []
    avg_rho_array = []
    avg_sr_array = []
    avg_vvd_array = []
    mwd_array = []
    times = []

    for w,site in enumerate(sites):
        print("Working on site " + site)
        number_of_files = 0
        for date in pip_dates:
            print("Working on day", date)
            
            year = int(date[:4])
            month = int(date[4:6])
            day = int(date[-2:])

            try:
                ds_edensity_lwe_rate = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/adjusted_edensity_lwe_rate/' + inst[w] + date + '_min.nc')
                ds_edensity_distributions = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/edensity_distributions/' + inst[w] + date + '_rho.nc')
                ds_velocity_distributions = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/velocity_distributions/' + inst[w] + date + '_vvd.nc')
                ds_particle_size_distributions = xr.open_dataset(pip_path + str(year) + '_' + site + '/netCDF/particle_size_distributions/' + inst[w] + date + '_psd.nc')
            except FileNotFoundError:
                print("Could not open PIP file")
                continue
            
            dsd_values = ds_particle_size_distributions['psd'].values
            edd_values = ds_edensity_distributions['rho'].values
            vvd_values = ds_velocity_distributions['vvd'].values
            sr_values = ds_edensity_lwe_rate['nrr_adj'].values
            ed_values = ds_edensity_lwe_rate['ed_adj'].values
            bin_centers = ds_particle_size_distributions['bin_centers'].values

            if len(ds_particle_size_distributions.time) != 1440:
                print("PIP data record too short for day, skipping!")
                continue

            ########## PIP CALCULATIONS 
            func = lambda t, a, b: a * np.exp(-b*t)

            # Initialize the datetime object at the start of the day
            current_time = datetime(year, month, day, 0, 0)

            # Loop over each 5-minute block
            count = 0
            for i in range(0, dsd_values.shape[0], 5):
                if i >= 1435:
                    continue

                count += 1
                block_avg = np.mean(dsd_values[i:i+5, :], axis=0)
                valid_indices = ~np.isnan(block_avg)
                block_avg = block_avg[valid_indices]
                valid_bin_centers = bin_centers[valid_indices]

                times.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
                current_time += timedelta(minutes=5)

                if block_avg.size == 0:
                    N_0_array.append(np.nan)
                    lambda_array.append(np.nan)
                    avg_vvd_array.append(np.nan)
                    avg_ed_array.append(np.nan)
                    avg_rho_array.append(np.nan)
                    avg_sr_array.append(np.nan)
                    total_particle_array.append(0)
                    mwd_array.append(np.nan)
                    site_array.append(np.nan)
                    continue

                # Calculate average fallspeed over the 5-minute interval
                vvd_slice = vvd_values[i:i+5, :]
                avg_vvd_array.append(vvd_slice[vvd_slice != 0].mean())

                # Calculate the average L4 density of the 5-minute interval
                avg_ed_array.append(np.nanmean(ed_values[i:i+5]))

                # Calculate the average eDensity of the 5-minute interval
                rho_slice = edd_values[i:i+5]
                avg_rho_array.append(rho_slice[rho_slice != 0].mean())

                # Calculate the average snowfall rate over the 5-minute interval
                avg_sr_array.append(np.nanmean(sr_values[i:i+5]))

                # Calculate total number of particles over the 5-minute interval
                total_particle_array.append(np.nansum(dsd_values[i:i+5, :], axis=(0, 1)))

                # Calculate mean mass diameter over the 5-minute interval
                if edd_values[i:i+5, valid_indices].shape == dsd_values[i:i+5, valid_indices].shape:
                    mass_dist = edd_values[i:i+5, valid_indices] * dsd_values[i:i+5, valid_indices] * (4/3) * np.pi * (valid_bin_centers/2)**3
                    mass_weighted_diameter = np.sum(mass_dist * valid_bin_centers) / np.sum(mass_dist)
                    mwd_array.append(mass_weighted_diameter)
                else:
                    mwd_array.append(np.nan)

                # Curve fit the PSD parameters of N0 and Lambda
                try:
                    popt, pcov = curve_fit(func, valid_bin_centers, block_avg, p0 = [1e4, 2], maxfev=600)
                    if popt[0] > 0 and popt[0] < 10**7 and popt[1] > 0 and popt[1] < 10:
                        N_0_array.append(popt[0])
                        lambda_array.append(popt[1])
                    else:
                        N_0_array.append(np.nan)
                        lambda_array.append(np.nan)

                except RuntimeError:
                    N_0_array.append(np.nan)
                    lambda_array.append(np.nan)

                site_array.append(site)

            number_of_files += 1

    # Store everything into a dataframe and save for ease-of-access
    df = pd.DataFrame(data={'site': site_array, 'time': times, 'n0': N_0_array,  'D0': mwd_array, 'Nt': total_particle_array, \
                            'Fs': avg_vvd_array, 'Sr': avg_sr_array,  'Ed': avg_ed_array, \
                            'Rho': avg_rho_array, 'lambda': lambda_array})
    df.dropna(inplace=True)
    # df.to_csv('../save_path.csv')

# plot_corr: Helper function for plotting variable correlations
def plot_corr(df, size=12):
    corr_df = df.drop(columns=['type'])
    corr = corr_df.corr()
    corr_sum = corr.sum().sort_values(ascending=False)
    corr = corr.loc[corr_sum.index, corr_sum.index]

    # Create a DataFrame correlation plot
    fig, ax = plt.subplots(figsize=(size, size))
    plt.title("PSD Variable Correlation Matrix")
    h = ax.matshow(corr, cmap='bwr', vmin=-1, vmax=1)
    fig.colorbar(h, ax=ax, label='Correlation')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.tight_layout()
    plt.savefig('../images/corr.png')

    sns_plot = sns.pairplot(df, kind="hist", diag_kind="kde", hue='type', height=5, palette=['blue', 'red'], corner=True)
    sns_plot.map_lower(sns.kdeplot, levels=4, color=".2")
    sns_plot.savefig('../images/output_kde.png')

# plot_timeseries: Helper function for plotting variable values over time (useful for finding erroneous periods)
def plot_timeseries(site):
    df = pd.read_csv('../processed/pca_inputs/' + site + '_pip.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna() 
    df = df[(df['Ed'] >= 0)]
    df = df[(df['Ed'] <= 1)]
    df = df[(df['lambda'] <= 2)]
    df['time'] = pd.to_datetime(df['time'])

    df.set_index('time', inplace=True)
    cols = ['Nt', 'n0', 'lambda', 'Ed', 'D0', 'Sr', 'Fs', 'Rho']
    units = ['#', 'm-3 mm-1', 'mm-1', 'g cm-3', 'mm', 'mm hr-1', 'm s-1', 'g cm-3']
    df_rolling = df[cols].rolling(window=1000).mean()

    fig, axs = plt.subplots(4, 2, figsize=(20, 10), sharex=True)
    for i, ax in enumerate(axs.flatten()):
        col = cols[i]
        df_rolling[col].plot(ax=ax, color='black', linewidth=2)
        ax.set_title(col)
        ax.set_ylabel(col + ' (' + units[i] + ')')
        ax.set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('../images/timeseries.png')

# load_and_plot_pca_for_site: Helper function for checking site correlations
def load_and_plot_pca_for_site(site):
    df = pd.read_csv('../processed/pca_inputs/' + site + '_pip.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna()
    df = df[(df['Ed'] >= 0)]
    df = df[(df['Ed'] <= 1)]
    df = df[(df['lambda'] <= 2)]
    df['type'] = df['Rho'].apply(lambda x: 'snow' if x < 0.4 else 'rain')

    df['Log10_n0'] = df['n0'].apply(np.log10)
    df['Log10_lambda'] = df['lambda'].apply(np.log10)
    df['Log10_Ed'] = df['Ed'].apply(np.log10)
    df['Log10_Fs'] = df['Fs'].apply(np.log10)
    df['Log10_Rho'] = df['Rho'].apply(np.log10)
    df['Log10_D0'] = df['D0'].apply(np.log10)
    df['Log10_Sr'] = df['Sr'].apply(np.log10)
    df['Log10_Nt'] = df['Nt'].apply(np.log10)
    df.drop(columns=['n0', 'lambda', 'Nt', 'Ed', 'Fs', 'Rho', 'D0', 'Sr'], inplace=True)
    plot_corr(df)

# Main runloop
if __name__ == '__main__':
    calc_various_pca_inputs()
    # plot_timeseries('MQT')
    # load_and_plot_pca_for_site('MQT')

    print("All done!")

