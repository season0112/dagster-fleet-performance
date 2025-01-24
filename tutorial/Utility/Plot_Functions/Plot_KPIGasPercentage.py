import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import KPIUtility

def plot_GasPercentage(BinCenterForMergeWindow_allCiC_RespectiveBins, GasPercentage_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_GasPercentage, lower_limit_GasPercentage, mean_value_GasPercentage_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, GasPercentage_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, GasPercentage_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, GasPercentage_iter)
        plt.scatter(BinCenterForMergeWindow_iter, GasPercentage_iter, alpha=0.6, s=70) 

    #ax.set_ylim([0, 100])

    plt.savefig('../Plots/GasUseagePercentage_withoutMean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_GasPercentage_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_GasPercentage, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_GasPercentage, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Gas Useage Percentage (%)', color='black')

    plt.savefig('../Plots/GasUseagePercentage.png')


def plot_HeatPumpPercentage(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatPumpPercentage_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_HeatPumpPercentage, lower_limit_HeatPumpPercentage, mean_value_HeatPumpPercentage_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, HeatPumpPercentage_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatPumpPercentage_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, HeatPumpPercentage_iter)
        plt.scatter(BinCenterForMergeWindow_iter, HeatPumpPercentage_iter, alpha=0.6, s=70) 

    #ax.set_ylim([0, 100])

    plt.savefig('../Plots/HeatPumpPercentage_withoutMean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_HeatPumpPercentage_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_HeatPumpPercentage, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_HeatPumpPercentage, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('HeatPump Useage Percentage (%)', color='black')

    plt.savefig('../Plots/HeatPumpPercentage.png')






