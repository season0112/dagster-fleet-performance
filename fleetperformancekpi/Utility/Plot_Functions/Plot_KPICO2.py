import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import KPIUtility
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from scipy.stats import norm


def plot_MeasuredCOP(BinCenterForMergeWindow_allCiC_RespectiveBins, MeasuredCOP_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_MeasuredCOP, lower_limit_MeasuredCOP, mean_value_all_MeasuredCOP):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, MeasuredCOP_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, MeasuredCOP_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, MeasuredCOP_iter)
        plt.scatter(BinCenterForMergeWindow_iter, MeasuredCOP_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_all_MeasuredCOP, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_MeasuredCOP, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_MeasuredCOP, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Measured COP', color='black')
    #ax.set_ylim([0, 2])
    plt.savefig('../Plots/MeasuredCOP.png')


def plot_ExpectedCOP(BinCenterForMergeWindow_allCiC_RespectiveBins, ExpectedCOP_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_ExpectedCOP, lower_limit_ExpectedCOP, mean_value_all_ExpectedCOP):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, ExpectedCOP_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, ExpectedCOP_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, ExpectedCOP_iter)
        plt.scatter(BinCenterForMergeWindow_iter, ExpectedCOP_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_all_ExpectedCOP, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_ExpectedCOP, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_ExpectedCOP, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('ExpectedCOP COP', color='black')
    #ax.set_ylim([0, 2])
    plt.savefig('../Plots/ExpectedCOP.png')


def plot_FleetCOPRatio(BinCenterForMergeWindow_allCiC_RespectiveBins, COPRatio_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit, lower_limit, mean_value_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    #ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, COPRatio_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, COPRatio_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, COPRatio_iter)
        plt.scatter(BinCenterForMergeWindow_iter, COPRatio_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('COP Ratio', color='black')
    ax.set_ylim([0, 2])
    plt.savefig('../Plots/FleetCOP.png')


def plot_FleetCOPRatio_Over_OutsideTemp(BinCenterForMergeWindow_allCiC_RespectiveBins, COPRatio_allCiC_RespectiveBins, AveragedOutsideTemp_allCiC_RespectiveBins, fleet_start_time, fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit, lower_limit, mean_value_all, upper_limit_averageOutT, lower_limit_averageOutT, mean_averageOutT_value_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    #ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, COPRatio_iter, AveragedOutsideTemp_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, COPRatio_allCiC_RespectiveBins, AveragedOutsideTemp_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, COPRatio_iter)
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter)
        plt.scatter(BinCenterForMergeWindow_iter, COPRatio_iter/AveragedOutsideTemp_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_value_all/mean_averageOutT_value_all, 'o-r', linewidth=10, markersize=25) 


    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('COP Ratio / Outside Temperature', color='black')
    ax.set_ylim([-40, 40])
    plt.savefig('../Plots/FleetCOP_Over_OutsideT.png')




def plot_FleetCOPRatioProjection(BinCenterForMergeWindow_allCiC_RespectiveBins, COPRatio_allCiC_RespectiveBins, BinCenterForMergeWindow_WholeTimeRange, mean_value_all, upper_limit, lower_limit, Nth_Projection):

    # Get Nth projection
    Nth_elements = []
    for idx, (BinCenterForMergeWindow_iter, COPRatio_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, COPRatio_allCiC_RespectiveBins)):  # loop for CiC list
        if BinCenterForMergeWindow_WholeTimeRange[Nth_Projection] in np.array(BinCenterForMergeWindow_iter):  # For each CiC, fill the prjection list if there is data 
            index = np.where(BinCenterForMergeWindow_iter == BinCenterForMergeWindow_WholeTimeRange[Nth_Projection])[0][0]
            Nth_elements.append(COPRatio_iter[index])
    Nth_elements = np.array(Nth_elements)

    #print("Nth_Projection:" + str(Nth_Projection))
    #print("Nth_elements:" + str(Nth_elements)) 
    #print("mean:" + str(np.mean(np.array(Nth_elements))))


    # Plot
    plt.figure(figsize=(30, 18))

    # plot histogram
    #plt.hist(Nth_elements, bins=50, color='blue', edgecolor='black', alpha=0.7, range=(0.2, 1.8))
    plt.hist(Nth_elements, bins=50, color='blue', edgecolor='black', alpha=0.7)

    # plot mean
    plt.axvline(mean_value_all[Nth_Projection], color='red', linestyle='--', linewidth=5, label=f'Mean = {mean_value_all[Nth_Projection]:.4f}')

    # plot uncertainty
    plt.axvline(lower_limit[Nth_Projection], color='purple', linestyle='-', linewidth=3, label=f'95% CI Lower = {lower_limit[Nth_Projection]:.4f}')
    plt.axvline(upper_limit[Nth_Projection], color='purple', linestyle='-', linewidth=3, label=f'95% CI Upper = {upper_limit[Nth_Projection]:.4f}')

    plt.title('Histogram of ' + str(Nth_Projection) + ' Element in COP Ratio', fontsize=40)
    plt.xlabel('COP Ratio')
    plt.ylabel('Frequency')
    plt.legend(fontsize=20)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('../Plots/FleetCOPProjection_' + str(Nth_Projection) + '_bin.png')
    



def plot_OutsideTemp(BinCenterForMergeWindow_allCiC_RespectiveBins, AveragedOutsideTemp_allCiC_RespectiveBins, fleet_start_time,fleet_end_time, freqvalue, BinCenterForMergeWindow_WholeTimeRange, upper_limit_averageOutT, lower_limit_averageOutT, mean_averageOutT_value_all):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

    # plot scatter plot
    for idx, (BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, AveragedOutsideTemp_allCiC_RespectiveBins)):
        KPIUtility.CheckBinXY(BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter)
        plt.scatter(BinCenterForMergeWindow_iter, AveragedOutsideTemp_iter, alpha=0.6, s=70) 

    # plot mean
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, mean_averageOutT_value_all, 'o-r', linewidth=10, markersize=25) 

    # plot upper lower limit
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, upper_limit_averageOutT, color='black', linestyle='-', linewidth=5)
    plt.plot(BinCenterForMergeWindow_WholeTimeRange, lower_limit_averageOutT, color='black', linestyle='-', linewidth=5)

    ax.set_xlim(fleet_start_time, fleet_end_time + pd.Timedelta(freqvalue))
    ax.set_ylabel('Outside Temperature', color='black')
    #ax.set_ylim([0, 100])    

    plt.savefig('../Plots/OutsideTemperature.png')

#### Obsolete

def plot_COPPerformance(df_load, BinCenterForMergeWindow, MergeWindow, MeasuredCOP, ExpectedCOP, CL):

    # Prepare: Check Bin dimentions
    KPIUtility.CheckBinXY(BinCenterForMergeWindow, ExpectedCOP)
    KPIUtility.CheckBinXY(BinCenterForMergeWindow, MeasuredCOP)

    # Plot COP
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.set_xlim([BinCenterForMergeWindow.min(), BinCenterForMergeWindow.max()])

    ax.plot(BinCenterForMergeWindow   , MeasuredCOP, 'rs', markersize=10, label='Measured COP')
    ax.scatter(BinCenterForMergeWindow, ExpectedCOP, color='green', s=50, label='COP(Poly Model)')

    ax.tick_params(axis='x', labelsize=30, size=30)
    ax.set_ylabel('COP', color='black') 
    ax.text(0.05, 0.95, "COPMergeWindow=" + str(MergeWindow) + "mins", transform=ax.transAxes, fontsize=30, verticalalignment='top', color='black')
    plt.legend(loc='best')
    ax.set_ylim([0, 14])
    plt.savefig('../Plots/COPPerformance_Poly.png')
    plt.close()

    # Calculate COP Ratio and Uncertainty: Fit and Get uncertainty band
    COPRatio, COPRatioFitted, lower_bound, upper_bound, mask_reasonalble = KPIUtility.CalculateCOPRatioAndUncertainty(ExpectedCOP, MeasuredCOP, BinCenterForMergeWindow, CL)

    #print(COPRatio)
    # Plot COP Ratio
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.set_xlabel('Month-Day', color='black')
    ax.set_ylabel('COP Ratio', color='black')

    ax.plot(BinCenterForMergeWindow[mask_reasonalble], COPRatio[mask_reasonalble], 'rs', markersize=10)
    ax.plot(BinCenterForMergeWindow[mask_reasonalble], COPRatioFitted            , 'b' , markersize=10)
    ax.fill_between(BinCenterForMergeWindow[mask_reasonalble], lower_bound, upper_bound, color='gray', alpha=0.3, label='Uncertainty Band')

    ax.set_xlim([BinCenterForMergeWindow.min(), BinCenterForMergeWindow.max()])
    ax.text(0.05, 0.95, "COPMergeWindow=" + str(MergeWindow) + "mins", transform=ax.transAxes, fontsize=30, verticalalignment='top', color='black')
    ax.set_ylim([0, 4])
    #plt.legend(loc='best')
    plt.savefig('../Plots/COPPerformanceRatio_Poly.png')
    plt.close()

def plot_COPPerformance_AllMethods(df_load, COPMergerWindow, COP, COP_MAX, ExpectedCOP_Linear, ExpectedCOP_Poly, ExpectedCOP_SVR, ExpectedCOP_MLP, ExpectedCOP_DTR, ExpectedCOP_RFR):

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    #ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), COP_MAX, 'rs', markersize=10, label='MAXCOP')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.set_ylim([0, 10])
    ax.set_ylabel('COP', color='black')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_Linear,  markersize=10, label='COP(Linear Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_Poly  ,  markersize=10, label='COP(Poly Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_SVR   ,  markersize=10, label='COP(SVR Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_MLP   ,  markersize=10, label='COP(MLP Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_DTR   ,  markersize=10, label='COP(DTR Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_RFR   ,  markersize=10, label='COP(RFR Model)')
    plt.legend(loc='best')
    plt.savefig('../Plots/COPPerformance_WithoutMeasuredCOP.png')
    ax.plot(pd.to_datetime(df_load['time_ts'][range(int(COPMergerWindow/2), len(df_load['time_ts']), COPMergerWindow)], utc=True), COP, 'rs', markersize=20, label='Measured COP')
    ax.set_ylim([-3, 20])
    plt.savefig('../Plots/COPPerformance_WithMeasuredCOP.png')
    plt.close()

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.set_ylim([0, 10])
    ax.set_ylabel('COP', color='black')
    ax.plot(pd.to_datetime(df_load['time_ts'][range(int(COPMergerWindow/2), len(df_load['time_ts']), COPMergerWindow)], utc=True), COP, 'rs', markersize=20, label='Measured COP')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_Linear,  color='blue', markersize=10, label='COP(Linear Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_Poly  ,  color='green', markersize=10, label='COP(Poly Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_SVR   ,  color='orange', markersize=10, label='COP(SVR Model)')
    ax.text(0.05, 0.95, "COPMergeWindow=" + str(COPMergerWindow*15/60) + "mins", transform=ax.transAxes, fontsize=30, verticalalignment='top', color='black')
    plt.legend(loc='best')
    ax.set_ylim([0, 10])
    plt.savefig('../Plots/COPPerformance_LinearPolySVR.png')
    plt.close()

    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    ax.set_ylim([0, 10])
    ax.set_ylabel('COP', color='black')
    ax.plot(pd.to_datetime(df_load['time_ts'][range(int(COPMergerWindow/2), len(df_load['time_ts']), COPMergerWindow)], utc=True), COP, 'rs', markersize=20, label='Measured COP')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_MLP   ,  color='blue', markersize=10, label='COP(MLP Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_DTR   ,  color='green', markersize=10, label='COP(DTR Model)')
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), ExpectedCOP_RFR   ,  color='orange', markersize=10, label='COP(RFR Model)')
    ax.text(0.05, 0.95, "COPMergeWindow=" + str(COPMergerWindow*15/60) + "mins", transform=ax.transAxes, fontsize=30, verticalalignment='top', color='black')
    plt.legend(loc='best')
    ax.set_ylim([0, 10])
    plt.savefig('../Plots/COPPerformance_MLPDTRRFR.png')
    plt.close()


