import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KPIUtility


def plot_FleetRoomAndSetpointTemperature(df_OneCiC, cic):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    plt.plot(df_OneCiC['time_ts'], df_OneCiC['thermostat_otFtRoomTemperature'])
    plt.plot(df_OneCiC['time_ts'], df_OneCiC['thermostat_otFtRoomSetpoint'])
    plt.ylim(15, 30)
    plt.xticks(fontsize=30)
    plt.savefig('../Plots/TrackingError' + str(cic) + '.png')
    plt.close()

def plot_TrackingError(BinCenterForMergeWindow_allCiC_RespectiveBins, TrackingError_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit, lower_limit, mean_value):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Tracking Error (degree)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, TrackingError_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, TrackingError_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, TrackingError_iter)
        plt.scatter(BinCenterForMergeWindow_iter, TrackingError_iter, alpha=0.6, s=70)

    ax.set_ylim([-2, 2])
    plt.savefig('../Plots/TrackingError_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit, color='black', linestyle='-', linewidth=5)

    ax.set_ylim([-0.5, 0.5])
    plt.savefig('../Plots/TrackingError.png')

def plot_RisingTime(BinCenterForMergeWindow_allCiC_RespectiveBins, RisingTime_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_risingtime, lower_limit_risingtime, mean_value_risingtime):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('RisingTime (mins)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, RisingTime_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, RisingTime_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, RisingTime_iter)
        plt.scatter(BinCenterForMergeWindow_iter, RisingTime_iter, alpha=0.6, s=70)

    plt.yscale("log")
    #ax.set_ylim([0, 100])
    plt.savefig('../Plots/RisingTime_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_risingtime, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_risingtime, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_risingtime, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/RisingTime.png')

    # Linear
    plt.yscale("linear")
    ax.set_ylim([10, 2000])
    plt.savefig('../Plots/RisingTime_limitedY.png')

def plot_SetPointTemperatureChange(BinCenterForMergeWindow_allCiC_RespectiveBins, SetPointTemperatureChange_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_setpointTchange, lower_limit_setpointTchange, mean_value_setpointTchange):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('SetPoint Temperature Change (degrees)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, SetPointTChange_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, SetPointTemperatureChange_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, SetPointTChange_iter)
        plt.scatter(BinCenterForMergeWindow_iter, SetPointTChange_iter, alpha=0.6, s=70)

    #plt.yscale("log")
    #ax.set_ylim([0, 100])
    plt.savefig('../Plots/SetPointTemperatureChange_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_setpointTchange, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_setpointTchange, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_setpointTchange, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/SetPointTemperatureChange.png')

def plot_OneCiCOneDayTemperature(df_OneCiC, idx, index_ReachedSteadyState, index_EndOfStableSetPointTemperature, cic):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    plt.title('One CiC in One Day: Parameters Visualization', fontsize=50, fontweight='bold')
    plt.plot(df_OneCiC['time_ts'], df_OneCiC['thermostat_otFtRoomTemperature'], color='blue', label='Room Temperature')
    plt.plot(df_OneCiC['time_ts'], df_OneCiC['thermostat_otFtRoomSetpoint']   , color='green', label='SetPoint Temperature')
    ax.axvline(x=df_OneCiC['time_ts'].iloc[idx], color='red', linestyle='--', linewidth=5, label='Start of Rising Time')
    ax.axvline(x=df_OneCiC['time_ts'].iloc[index_ReachedSteadyState], color='gold', linestyle='--', linewidth=5, label='End of Rising Time & Start of Steady State')
    ax.axvline(x=df_OneCiC['time_ts'].iloc[index_EndOfStableSetPointTemperature], color='magenta', linestyle='--', linewidth=5, label='End of Steady State')
    plt.ylim(15, 30)
    plt.xticks(fontsize=30)
    ax.set_xlabel('Time', color='black')
    ax.set_ylabel('Temperature', color='black')
    plt.legend()
    plt.savefig('../Plots/TrackingError' + str(cic) + '.png')
    plt.close()




