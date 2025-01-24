import KPIUtility
import matplotlib.pyplot as plt
import pandas as pd


def plot_WatchdogChange_frequency (BinCenterForMergeWindow_allCiC_RespectiveBins, WatchdogChangeCounts_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit, lower_limit, mean_value_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, WatchdogChangeCounts_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, WatchdogChangeCounts_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, WatchdogChangeCounts_iter)
        plt.scatter(BinCenterForMergeWindow_iter, WatchdogChangeCounts_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('WatchdogCode Change Counts', color='black')
    ax.set_ylim([0, 50])
    plt.savefig('../Plots/WatchdogCodeChangeCounts.png')


def plot_WatchdogChange_percentage(BinCenterForMergeWindow_allCiC_RespectiveBins, WatchdogPercentageCounts_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_percentage, lower_limit_percentage, mean_value_percentage_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, WatchdogPercentageCounts_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, WatchdogPercentageCounts_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, WatchdogPercentageCounts_iter)
        plt.scatter(BinCenterForMergeWindow_iter, WatchdogPercentageCounts_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_percentage_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_percentage, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_percentage, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Watchdog Code Abnormal Percentage', color='black')
    #ax.set_ylim([0, 50])
    plt.savefig('../Plots/WatchdogCodePercentage.png')






