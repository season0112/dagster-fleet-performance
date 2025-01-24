import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KPIUtility 


def plot_HeatingCycle_frequency(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatingCycleChangeCounts_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit, lower_limit, mean_value_all, name):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, HeatingCycleChangeCounts_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatingCycleChangeCounts_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, HeatingCycleChangeCounts_iter)
        plt.scatter(BinCenterForMergeWindow_iter, HeatingCycleChangeCounts_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Heating Cycle Change Counts', color='black')
    ax.set_ylim([0, 50])

    plt.savefig('../Plots/HeatingCycleChangeCounts' + name  +'.png')


def plot_HeatingCycle_percentage(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatingCyclePercentageCounts_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_percentage, lower_limit_percentage, mean_value_percentage_all, name):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, HeatingCyclePercentageCounts_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatingCyclePercentageCounts_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, HeatingCyclePercentageCounts_iter)
        plt.scatter(BinCenterForMergeWindow_iter, HeatingCyclePercentageCounts_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_percentage_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_percentage, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_percentage, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Heating Cycle Percentage', color='black')
    #ax.set_ylim([0, 50])
    plt.savefig('../Plots/HeatingCyclePercentage' + name + '.png')


#### Obsolete

def plot_FunctionStatusNumbers_Thresholds(ThresholdTime, df_load, minimumRecords_list):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(ThresholdTime, [KPIUtility.FindNonZeroPeriodNumbers(df_load, 'functionStatus', minimumRecords) for minimumRecords in minimumRecords_list], 'bs', markersize=20) 
    ax.set_xlabel("Threshold (mins)")
    ax.set_ylabel("Number of FunctionStatus periods")
    plt.savefig('../Plots/functionStatus_Thresholds.png')
    plt.close()


def plot_SupervisoryControlMode(df_load):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), df_load['qc_supervisoryControlMode'], 'bs', markersize=10)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    #ax.text(0.05, 0.95, "2 for heatpump, 3 for heatpump and boiler, 4 for boiler", transform=ax.transAxes, fontsize=40, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    ax.set_ylabel("qc_supervisoryControlMode")
    #ax.set_ylim(-1, 6)
    plt.savefig('../Plots/' + 'qc_supervisoryControlMode' + '.png')
    plt.close()


def plot_functionStatus(df_load):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))

    ax.plot( df_load['time_ts'], df_load['functionStatus'], 'bs', markersize=3, label='Pump+Boiler OFF')

    '''
    ax.plot(pd.to_datetime(df_load['time_ts'][df_load['functionStatus'] == 0], utc=True),
            df_load['functionStatus'][df_load['functionStatus'] == 0],
            'bs', markersize=3, label='Pump+Boiler OFF')  

    #ax.plot(pd.to_datetime(df_load['time_ts'][df_load['functionStatus'] == 1], utc=True),
            df_load['functionStatus'][df_load['functionStatus'] == 1],
            'rs', markersize=3, label='One of or both are ON')  
    '''

    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    ax.set_ylabel("Heat System Running Status")
    plt.legend(loc='best', markerscale=2)
    plt.savefig('../Plots/functionStatus.png')
    plt.close()


def plot_SlidingWindow(window_centers, percentage_counts, change_counts):

    fig, ax = plt.subplots(figsize=(30, 18))
    ax.plot(window_centers, percentage_counts, marker='o', linestyle='-', color='b')
    ax.set_xlabel('Time (Sliding Window Center)')
    ax.set_ylabel('Percentage')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))  
    #plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('../Plots/' + 'SlidingWindow_Percentage.png')

    fig, ax = plt.subplots(figsize=(30, 18))
    ax.plot(window_centers, change_counts, marker='o', linestyle='-', color='b')
    ax.axhline(y=320, color='r', linestyle='--', label='Theoretical Limit')
    ax.set_xlabel('Time (Sliding Window Center)')
    ax.set_ylabel('Fraquency')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))  
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('../Plots/' + 'SlidingWindow_Fraquency.png')


