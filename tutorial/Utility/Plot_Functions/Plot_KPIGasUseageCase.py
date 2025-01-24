import KPIUtility
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .. import PythonPlotDefaultParameters
import matplotlib.dates as mdates

def plot_GasUseageCase(BinCenterForMergeWindow_WholeTimeRange, KPIlist):

    # initialize list
    num_elements = len(next(iter(KPIlist['boiler_usage_heat'].values()))['HighDemand'])  

    KPIlist['boiler_usage_heat'].values()

    sums_list_HighDemand       = [0] * num_elements
    sums_list_WaterTempTooHigh = [0] * num_elements
    sums_list_PreHeating       = [0] * num_elements
    sums_list_Anomaly          = [0] * num_elements
    sums_list_NoReason         = [0] * num_elements

    for values in KPIlist['boiler_usage_heat'].values():
        for i in range(num_elements):
            if i < len(values['HighDemand']):
                sums_list_HighDemand[i] += values['HighDemand'][i]
                sums_list_WaterTempTooHigh[i] += values['WaterTempTooHigh'][i]
                sums_list_PreHeating[i] += values['PreHeating'][i]
                sums_list_Anomaly[i] += values['Anomaly'][i]
                sums_list_NoReason[i] += values['NoReason'][i]

    data = np.vstack([
        sums_list_HighDemand,
        sums_list_WaterTempTooHigh,
        sums_list_PreHeating,
        sums_list_Anomaly,
        sums_list_NoReason
    ])

    # normalize
    data = (data / np.sum(data, axis=0)) * 100

    labels = BinCenterForMergeWindow_WholeTimeRange.dt.strftime('%Y-%m-%d')  
    codes = ['HighDemand', 'WaterTempTooHigh', 'PreHeating', 'Anomaly', 'NoReason']
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C299FF']  

    # plot
    fig, ax = plt.subplots(figsize=(30, 18))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    # ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=13))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    ax.stackplot(labels[0:len(sums_list_HighDemand)], data, labels=codes, colors=colors, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Gas Usage Percentage (%)')
    ax.legend(loc='upper left')
    #ax.tick_params(axis='x', labelsize=25)

    # plt.tight_layout()
    plt.savefig("../Plots/GasUseageCase.png")


def plot_PiePlotForWatchdogHP1(df_load):
    # For all watchdog Code
    watchdog_counts = df_load['hp1_watchdogCode'].value_counts()

    fig, ax = plt.subplots(figsize=(18, 12))  
    ax.pie(watchdog_counts, 
           labels=watchdog_counts.index, 
           autopct='%1.1f%%', 
           startangle=140,
           textprops={'fontsize': 25})  
    ax.axis('equal')  
    plt.title('hp1_watchdogCode Pie Plot', fontsize=30)  
    plt.savefig('../Plots/'+ 'watchdog.png')

    # For watchdog Code except "00"
    df_load = df_load[df_load['hp1_watchdogCode'] != 0]
    watchdog_counts_except00 = df_load['hp1_watchdogCode'].value_counts()

    fig, ax = plt.subplots(figsize=(18, 12))  
    ax.pie(watchdog_counts_except00, 
           labels=watchdog_counts_except00.index, 
           autopct='%1.1f%%', 
           startangle=140,
           textprops={'fontsize': 25})  
    ax.axis('equal')  
    plt.title('hp1_watchdogCode Pie Plot (Except Code "00")', fontsize=30)  
    plt.savefig('../Plots/'+ 'watchdog_except00.png')



#### Obsolete

def plot_CVRunningNumbers_Thresholds(ThresholdTime, df_load, minimumRecords_list):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(ThresholdTime, [KPIUtility.FindNonZeroPeriodNumbers(df_load, 'qc_cvPowerOutput', minimumRecords) for minimumRecords in minimumRecords_list], 'bs', markersize=20)
    ax.set_xlabel("Threshold (mins)")
    ax.set_ylabel("Number of CVRunning periods")
    #ax.set_yscale('log')
    plt.savefig('../Plots/CVRunningNumbers_Thresholds.png')
    plt.close()

def plot_SetpointTemperatureNumbers_Thresholds(ThresholdTime, df_load, minimumRecords_list):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(ThresholdTime, [KPIUtility.SetpointTemperatureChange(df_load, minimumRecords) for minimumRecords in minimumRecords_list], 'bs', markersize=20)
    ax.set_xlabel("Threshold (mins)")
    ax.set_ylabel("Number of SetpointTemperature periods")
    #ax.set_yscale('log')
    plt.savefig('../Plots/SetpointTemperatureNumbers_Thresholds.png')
    plt.close()

def plot_SetpointTemperature(df_load, SetpointTemperatureChangeNumber):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), df_load['thermostat_otFtRoomSetpoint'], 'bs', markersize=10, label='ChangeNumber: ' + str(SetpointTemperatureChangeNumber))
    ax.set_ylabel("thermostat_otFtRoomSetpoint")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    plt.legend(loc='best', fontsize=40)
    plt.savefig('../Plots/'+ 'SetpointTemperature' + '.png')
    plt.close()

def plot_cvPowerOutput(df_load, CVRunningNumbers):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(pd.to_datetime(df_load['time_ts'], utc=True), df_load['qc_cvPowerOutput'], 'bs', markersize=10, label='CV running number: ' + str(CVRunningNumbers))
    ax.set_ylabel("qc_cvPowerOutput")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))
    plt.legend(loc='best', fontsize=40)
    plt.savefig('../Plots/'+ 'qc_cvPowerOutput' + '.png')
    plt.close()


