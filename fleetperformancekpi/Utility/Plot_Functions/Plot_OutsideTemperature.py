import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KPIUtility


def plot_OutsideTemperature(BinCenterForMergeWindow_allCiC_RespectiveBins, AveragedOutsideTemp_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_averageOutT, lower_limit_averageOutT, mean_averageOutT_value_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Outside Temperature (Degree)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, AveragedOutsideTemp_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter)
        plt.scatter(BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter, alpha=0.6, s=70)

    #ax.set_ylim([-2, 2])
    #plt.yscale("log")

    plt.savefig('../Plots/AveragedOutsideTemp_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_averageOutT_value_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_averageOutT, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_averageOutT, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/AveragedOutsideTemp.png')











