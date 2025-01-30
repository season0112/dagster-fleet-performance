import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KPIUtility


def plot_InsulationIndicator(BinCenterForMergeWindow_allCiC_RespectiveBins, InsulationIndicator_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_InsulationIndicator, lower_limit_InsulationIndicator, mean_value_InsulationIndicator):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Insulation Indicator (wH/Degree)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, InsulationIndicator_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, InsulationIndicator_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, InsulationIndicator_iter)
        plt.scatter(BinCenterForMergeWindow_iter, InsulationIndicator_iter, alpha=0.6, s=70)

    #ax.set_ylim([-2, 2])
    #plt.yscale("log")

    plt.savefig('../Plots/InsulationIndicator_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_InsulationIndicator, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_InsulationIndicator, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_InsulationIndicator, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/InsulationIndicator.png')

    ax.set_ylim([-30, 200])
    plt.savefig('../Plots/InsulationIndicator_limitedY.png')


def plot_DissipationIndicator(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatDissipationIndicator_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_DissipationIndicator, lower_limit_DissipationIndicator, mean_value_DissipationIndicator):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Dissipation Indicator (wH/Degree)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, DissipationIndicator_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, HeatDissipationIndicator_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, DissipationIndicator_iter)
        plt.scatter(BinCenterForMergeWindow_iter, DissipationIndicator_iter, alpha=0.6, s=70)

    #ax.set_ylim([-2, 2])
    #plt.yscale("log")

    plt.savefig('../Plots/DissipationIndicator_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_DissipationIndicator, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_DissipationIndicator, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_DissipationIndicator, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/DissipationIndicator.png')

    ax.set_ylim([-10, 70])
    plt.savefig('../Plots/DissipationIndicator_limitedY.png')


def plot_SteadyStateRatio_Insulation(BinCenterForMergeWindow_allCiC_RespectiveBins, SteadyStateRatio_Insulation_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_SSRInsulation, lower_limit_SSRInsulation, mean_value_SSRInsulation):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('SteadyStateRatio (Insulation)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, SteadyStateRatio_Insulation_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, SteadyStateRatio_Insulation_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, SteadyStateRatio_Insulation_iter)
        plt.scatter(BinCenterForMergeWindow_iter, SteadyStateRatio_Insulation_iter, alpha=0.6, s=70)

    #ax.set_ylim([-2, 2])
    plt.savefig('../Plots/SteadyStateRatio_Insulation_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_SSRInsulation, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_SSRInsulation, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_SSRInsulation, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/SteadyStateRatio_Insulation.png')

def plot_SteadyStateRatio_Dissipation(BinCenterForMergeWindow_allCiC_RespectiveBins, SteadyStateRatio_HeatDissipation_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_SSRDissipation, lower_limit_SSRDissipation, mean_value_SSRDissipation):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('SteadyStateRatio (Dissipation)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, SteadyStateRatio_Dissipation_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, SteadyStateRatio_HeatDissipation_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, SteadyStateRatio_Dissipation_iter)
        plt.scatter(BinCenterForMergeWindow_iter, SteadyStateRatio_Dissipation_iter, alpha=0.6, s=70)

    #ax.set_ylim([-2, 2])
    plt.savefig('../Plots/SteadyStateRatio_Dissipation_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_SSRDissipation, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_SSRDissipation, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_SSRDissipation, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/SteadyStateRatio_Dissipation.png')

def plot_SteadyStateRatio_TrackingError(BinCenterForMergeWindow_allCiC_RespectiveBins, SteadyStateRatio_TrackingError_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_SSRTrackingError, lower_limit_SSRTrackingError, mean_value_SSRTrackingError):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('SteadyStateRatio (Tracking Error)', color='black')

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, SteadyStateRatio_TrackingError_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, SteadyStateRatio_TrackingError_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, SteadyStateRatio_TrackingError_iter)
        plt.scatter(BinCenterForMergeWindow_iter, SteadyStateRatio_TrackingError_iter, alpha=0.6, s=70)

    #ax.set_ylim([-2, 2])
    plt.savefig('../Plots/SteadyStateRatio_TrackingError_withoutmean.png')

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_SSRTrackingError, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_SSRTrackingError, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_SSRTrackingError, color='black', linestyle='-', linewidth=5)

    plt.savefig('../Plots/SteadyStateRatio_TrackingError.png')


