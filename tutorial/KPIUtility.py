import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from scipy.stats import norm
import Filters
import os

def FailterOnPumpRunningOnly(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([2])].index, inplace=True)

def FailterOnPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([2, 3])].index, inplace=True)

def printwelcome():
    print("\n")
    print("#" * 69)
    print("#" + " " * 12 + "\033[1mWelcome to Fleet Performance KPIs Analysis!\033[0m" + " " * 12 + "#")
    print("#" * 69)
    print("\n")

def ending():
    print("\n")
    print("#" * 69)
    print("#" + " " * 12 + "\033[1mEnd of Fleet Performance KPIs Analysis. Good Bye!\033[0m" + " " * 6 + "#")
    print("#" * 69)
    print("\n")

def excutefilters(df_load, filter_list, general_filters):
    for filter_name in filter_list:
        filter_function = getattr(general_filters, filter_name, None)

        if callable(filter_function):
            number_before_filter = len(df_load)
            filter_function(df_load)
            print("\033[36m" + str(filter_name) + str(" Pass Ratio: \033[35m") + f"{len(df_load)/number_before_filter*100:.2f}" + "%")
        else:
            print(f"Filter '{filter_name}' doesn't exist.")

def applyfilters(df_load, FilterList, filters, filters_sector_name):
    RawRecordNumber = len(df_load)
    print("#" * 69)
    print("\033[1;31m" + filters_sector_name + ":" + "\033[0m")    

    excutefilters(df_load, FilterList, filters)

    print("\033[32mTotal Pass Ratio: \033[35m" + f"{len(df_load) / RawRecordNumber * 100:.2f} %" +  "\033[0m")
    print("#" * 69)     
    print("\n")


def check_quality(df):
    print("#" * 69)
    print("\033[1;31mStart Quality Check:\033[0m")
    print("\033[32mTotal Number of Analysed CiC Records: \033[35m" + str(len(df)) + "\033[0m")
    print("\033[32mTotal Number of Analysed Unique CiCs: \033[35m" + str(len(df['clientid'].unique())) + "\033[0m")
    nan_counts = df.isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]
    
    if not nan_columns.empty:
        error_message = "\033[32mData quality check: \033[35m" + "DataFrame contains NaN values.\033[0m\n"
        error_message += "\033[32mNaN counts by column:\033[0m\n"  
        for column, count in nan_columns.items():
            error_message += f" \033[36m- {column}\033[0m: \033[35m{count} NaN values\033[0m\n" 
        print(error_message)
    else:
        print("\033[32mData quality check passed: No NaN values found.\033[0m")  
    print('\n')
    print("#" * 69)

def CheckBinXY(x, y):
    if len(x) != len(y):
        print("x axis has length:" + str(len(x)))
        print("y axis has length:" + str(len(y)))
        raise ValueError(f"\033[31mBin Numbers Check: X and Y axis have different dimentions.\033[0m\n")

def custom_floor(ts, start_time, freq_minutes):
    delta = (ts - start_time).total_seconds() // (freq_minutes * 60)
    return start_time + pd.Timedelta(minutes=delta * freq_minutes)


def UnvertToKelvin(inputColumn):
    return inputColumn + 273.15

def SetpointTemperatureChange(df_load, minimumRecords=4):
    data = df_load['thermostat_otftroomsetpoint']
    change_points = data != data.shift(1)
    groups = change_points.cumsum()
    group_sizes = data.groupby(groups).size()
    result = (group_sizes >= minimumRecords).sum()
    return result

def supervisoryControlModeChange(df_load, minimumRecords=4): 
    df_load['shift'] = df_load['qc_supervisorycontrolmode'].shift(1, fill_value=df_load['qc_supervisorycontrolmode'].iloc[0]) # TODO: Check iloc or loc?
    df_load['group'] = (df_load['qc_supervisorycontrolmode'] != df_load['shift']).cumsum()
    consecutive_ones = df_load[df_load['qc_supervisorycontrolmode'] == 1].groupby('group').size()
    result = (consecutive_ones >= minimumRecords).sum()
    return result

def FindNonZeroPeriodNumbers(df_load, propertyName, minimumRecords=4):
    non_zero_indices = np.where(df_load[propertyName] != 0)[0]
    diff = np.diff(non_zero_indices)
    segment_starts = np.where(diff > 1)[0] + 1
    segments = np.split(non_zero_indices, segment_starts)
    count = sum(len(segment) >= minimumRecords for segment in segments)
    return count


def GetTemperatureOutside(df_load):
    df_load['hp1_temperatureoutside'] = pd.to_numeric(df_load['hp1_temperatureoutside'], errors='coerce')
    df_load['hp2_temperatureoutside'] = pd.to_numeric(df_load['hp2_temperatureoutside'], errors='coerce')

    if df_load['hp2_temperatureoutside'].isna().all():
        hp_temperatureOutside = df_load['hp1_temperatureoutside']
    else:
        hp_temperatureOutside = (df_load['hp1_temperatureoutside'] + df_load['hp2_temperatureoutside']) / 2
    
    return hp_temperatureOutside


def CalculateCOP(df_load):
    COP = ((df_load['hp1_thermalenergycounter'] + df_load['hp2_thermalenergycounter'].fillna(0)).diff() / (df_load['hp1_electricalenergycounter'] + df_load['hp2_electricalenergycounter'].fillna(0)).diff()).fillna(0)
    return COP


def MergeWindow(df_load, cic, MergeWindow, fleet_start_time, fleet_end_time, freqvalue):

    ## Check if there is any MergeWindow has no data 
    # prepare
    #print("df_load:" + str(df_load))
    #print("before:")
    #print(df_load[(df_load['clientid'] == 'CIC-166bbb7c-4246-5750-adc2-9dd051ecaabc') & (df_load['client_time'] > '2023-12-21 00:00:00') & (df_load['client_time'] < '2023-12-26 00:00:00')])
    df_load_original = df_load

    # Find merged window without record out of range
    #print("fleet_start_time:" + str(fleet_start_time))
    #print("fleet_end_time:" + str(fleet_end_time))
    all_hours               = pd.date_range(start = fleet_start_time, end=fleet_end_time, freq=freqvalue)

    '''
    grouping_Result2 = df_load.groupby(pd.Grouper(key='client_time', freq=freqvalue)).size()
    df_load['time_bin'] = pd.cut(df_load['client_time'], bins=all_hours, right=False)
    grouping_Result = df_load.groupby('time_bin').size()
    print("grouping_Result:" + str(grouping_Result))
    print("grouping_Result2:" + str(grouping_Result2))
    '''
    grouping_Result = pd.Series(0, index=all_hours)  
    for i in range(len(all_hours) - 1):
        mask = (df_load['client_time'] >= all_hours[i]) & (df_load['client_time'] < all_hours[i + 1])
        grouping_Result[all_hours[i]] = df_load.loc[mask].shape[0]

    timewindow_with_records = grouping_Result.index
    #print("all_hours:" + str(all_hours))
    #print("timewindow_with_records:" + str(timewindow_with_records))
    missingcount_outside_dataTaking = all_hours.difference(timewindow_with_records)
    #print("missingcount_outside_dataTaking: " + str(missingcount_outside_dataTaking))
    half_freq = pd.Timedelta(freqvalue) / 2
    missingcount_outside_dataTaking_center = missingcount_outside_dataTaking + half_freq
    #print("missingcount_outside_dataTaking_center: " + str(missingcount_outside_dataTaking_center))

    # Find merged window with "One or Zero" record in the range
    missingcount_within_dataTaking  = (grouping_Result == 0).sum()
    onecount_within_dataTaking      = (grouping_Result == 1).sum()
    missingcount_within_dataTaking_TimeIndex = grouping_Result[grouping_Result == 0].index
    onecount_within_dataTaking_TimeIndex     = grouping_Result[grouping_Result == 1].index
    missingcount_within_dataTaking_PositionIndex = grouping_Result.index.get_indexer(missingcount_within_dataTaking_TimeIndex) 
    onecount_within_dataTaking_PositionIndex     = grouping_Result.index.get_indexer(onecount_within_dataTaking_TimeIndex)
    totalPositionIndex = list(missingcount_within_dataTaking_PositionIndex) + list(onecount_within_dataTaking_PositionIndex)
    #print("totalPositionIndex: " + str(totalPositionIndex))   
 
    ''' 
    if len(totalPositionIndex) != 0 or len(missingcount_outside_dataTaking) != 0:
        print("\033[31mChecking Offline Warning: " + str(len(totalPositionIndex)) + " (0 or 1 record inside the range) and " + str(len(missingcount_outside_dataTaking)) + " (outside the range) " + "Merge Time Windows have no data!\033[0m")
        print('\n')
    '''

    # Drop the record (only for one record case)  #FIXME
    #print("missingcount_within_dataTaking_PositionIndex: " + str(missingcount_within_dataTaking_PositionIndex))
    #print("onecount_within_dataTaking_TimeIndex: " + str(onecount_within_dataTaking_TimeIndex))
    #df_load_filtered_records = df_load[df_load['client_time'].dt.floor(freqvalue).isin(onecount_within_dataTaking_TimeIndex)]
    df_load_min = df_load['client_time'].min().normalize()
    df_load_filtered_records = df_load[df_load['client_time'].apply(custom_floor, args=(df_load_min, MergeWindow)).isin(onecount_within_dataTaking_TimeIndex)].copy()
    #print("df_load_filtered_records: " + str(df_load_filtered_records))
    df_load = df_load[~df_load.index.isin(df_load_filtered_records.index)].copy()
    #print("df_load:" + str(df_load))

    # Get Index for MergeWindow Bin Edge (right bin edge in each merged time window?)
    #### FIXME below line!
    #MergeWindow_BinEdge = df_load.groupby(pd.Grouper(key='client_time', freq = freqvalue)).tail(1).index.tolist()  # If the time length can not be divided by the mergewindow, the last bin will still be keep.
    #MergeWindow_BinEdge = df_load.groupby('time_bin').apply(lambda x: x.index[-1]).dropna().tolist()
    df_load.loc[:, 'time_bin'] = pd.cut(df_load['client_time'], bins=all_hours, right=False)

    #print("Test: ")
    #print( df_load.groupby('time_bin', observed=False).apply(lambda x: x.index[-1] if not x.empty else None).dropna() )

    MergeWindow_BinEdge = ( df_load.groupby('time_bin', observed=False).apply(lambda x: x.index[-1] if not x.empty else None).dropna())
    #print(len(MergeWindow_BinEdge))
    if len(MergeWindow_BinEdge)>0:
        MergeWindow_BinEdge = MergeWindow_BinEdge.tolist()
    else:
        MergeWindow_BinEdge = []
    '''
    MergeWindow_BinEdge = (
        df_load.groupby('time_bin', observed=False)
        .apply(lambda x: x.index[-1] if not x.empty else None)  # check if sub group is empty
        .dropna()  # delete empty group
        .tolist()
    )
    '''

    '''
    # Add index=0 for the left bin edge of first bin, if there is only one record in first bin, then the first index is equal to last index in this bin, therefore no need to insert.
    if MergeWindow_BinEdge[0] != 0 :
        MergeWindow_BinEdge.insert(0, 0) 
    '''
    MergeWindow_BinEdge.insert(0, 0)

    # Get BinCenter (only in the original timestamp range) TODO！！！！
    # Remember "Timestamp.floor('7200min')" starts from 1970-01-01 00:00:00.
    #start_time = df_load_original['client_time'].min().floor(freqvalue)  # the begining of this day!!!
    #start_time = df_load_original['client_time'].min().normalize()
    start_time = fleet_start_time

    #end_time   = df_load_original['client_time'].max().floor(freqvalue)  # start from start_time, the left binedge of last bin
    #end_time_max = df_load_original['client_time'].max()
    #end_time = pd.date_range(start=start_time, end=end_time_max, freq=freqvalue).max()
    end_time = fleet_end_time

    start_time_half = start_time + pd.Timedelta(minutes=MergeWindow/2)
    end_time_half = end_time + pd.Timedelta(minutes=MergeWindow/2) 
    '''
    if end_time == df_load_original['client_time'].max():
        end_time_half = end_time - pd.Timedelta(minutes=MergeWindow/2) # just in case if the last timestamp is next day 00:00:00, but not today 23:59:59. Not necessary if the quary stop at today 23:59:59. 
    else:
        end_time_half = end_time + pd.Timedelta(minutes=MergeWindow/2)
    '''

    BinCenterForMergeWindow = pd.date_range(start=start_time_half, end=end_time_half, freq=freqvalue)
    BinCenterForMergeWindow = pd.Series(BinCenterForMergeWindow)
    #print("start_time_half:" + str(start_time_half))
    #print("end_time_half:" + str(end_time_half))
    #print("freqvalue:" + str(freqvalue))
    #print("BinCenterForMergeWindow: " + str(BinCenterForMergeWindow))

    # Drop the BinCenterForMergeWindow for MergeWindow with zero or just one record inside
    BinCenterForMergeWindow = BinCenterForMergeWindow.drop(  totalPositionIndex ).reset_index(drop=True)
    BinCenterForMergeWindow = BinCenterForMergeWindow[~BinCenterForMergeWindow.isin(missingcount_outside_dataTaking_center)].reset_index(drop=True)

    #print("MergeWindow_BinEdge: " + str(MergeWindow_BinEdge))
    '''
    for i in MergeWindow_BinEdge:
        print(df_load['client_time'][i])
    print("BinCenterForMergeWindow: " + str(BinCenterForMergeWindow))   
    '''
    #print("\n")
    return MergeWindow_BinEdge, BinCenterForMergeWindow, df_load  


def CalculateCOP_withWindow(df_load, MergeWindow_BinEdge):
    COP = []
    for i in MergeWindow_BinEdge[1:]:
        thermalEnergy_diff    = (df_load['hp1_thermalenergycounter'].loc[i]    + df_load['hp2_thermalenergycounter'].loc[i])    - (df_load['hp1_thermalenergycounter'].loc[i-1]    + df_load['hp2_thermalenergycounter'].loc[i-1])
        electricalEnergy_diff = (df_load['hp1_electricalenergycounter'].loc[i] + df_load['hp2_electricalenergycounter'].loc[i]) - (df_load['hp1_electricalenergycounter'].loc[i-1] + df_load['hp2_electricalenergycounter'].loc[i-1]) 
        if electricalEnergy_diff != 0 and thermalEnergy_diff != 0:
            COP_tem = thermalEnergy_diff/electricalEnergy_diff
        else:
            COP_tem = np.nan
        COP.append(COP_tem)
    return COP

def CalculateGasPercentage_withWindow(df_load, MergeWindow_BinEdge):
    GasPercentage = []
    HeatPumpPercentage = []
    for i in MergeWindow_BinEdge[1:]:
        thermalEnergy_diff = (df_load['hp1_thermalenergycounter'].loc[i] + df_load['hp2_thermalenergycounter'].loc[i]) - (df_load['hp1_thermalenergycounter'].loc[i-1] + df_load['hp2_thermalenergycounter'].loc[i-1])        
        cvEnergy_diff      = df_load['qc_cvenergycounter'].loc[i] - df_load['qc_cvenergycounter'].loc[i-1]

        # print("thermalEnergy_diff:" + str(thermalEnergy_diff))
        # print("cvEnergy_diff:" + str(cvEnergy_diff))

        if cvEnergy_diff >= 0 and thermalEnergy_diff >= 0:
            GasPercentage_tem = cvEnergy_diff / (cvEnergy_diff + thermalEnergy_diff)            # (* 100) turns to percentage % 
            HeatPumpPercentage_tem = thermalEnergy_diff / (cvEnergy_diff + thermalEnergy_diff)  # (* 100) turns to percentage %
        else:
            GasPercentage_tem = np.nan
            HeatPumpPercentage_tem = np.nan

        # print("cvEnergy_diff:" + str(cvEnergy_diff))
        # print("HeatPumpPercentage_tem:" + str(HeatPumpPercentage_tem))

        GasPercentage.append(GasPercentage_tem)
        HeatPumpPercentage.append(HeatPumpPercentage_tem)

    return GasPercentage, HeatPumpPercentage


def CalculateMAXCOP(df_load):
    COP_MAX = UnvertToKelvin(df_load['thermostat_otftroomtemperature']) / (UnvertToKelvin(df_load['thermostat_otftroomtemperature']) - UnvertToKelvin(GetTemperatureOutside(df_load)))
    return COP_MAX


def CalculatePowerDemand(df_load):
    power = df_load['qc_estimatedpowerdemand'] # kW
    df_load['energy'] = power  * 1000 * 15  # 15 to be changed
    powerDemand = df_load['energy'].cumsum()
    return powerDemand


def FillSystemFunctionStatus(df_load):
    df_load['functionStatus_boiler'] = 0
    df_load['functionStatus_HP'] = 0
    df_load['functionStatus_HP_and_boiler'] = 0

    df_load.loc[df_load['qc_supervisorycontrolmode'].isin( [2,3] ), 'functionStatus_HP'] = 1
    df_load.loc[df_load['qc_supervisorycontrolmode'].isin( [3,4] ), 'functionStatus_boiler'] = 1
    df_load.loc[df_load['qc_supervisorycontrolmode'].isin( [2,3,4] ), 'functionStatus_HP_and_boiler'] = 1

    df_load['functionStatus_boiler'] = df_load['functionStatus_boiler'].astype(int)
    df_load['functionStatus_HP'] = df_load['functionStatus_HP'].astype(int)
    df_load['functionStatus_HP_and_boiler'] = df_load['functionStatus_HP_and_boiler'].astype(int)


def FillWatchdogAbnormalStatus(df_load):
    df_load['WatchdogAbnormalStatus'] = 0
    df_load['SevereWatchdogAbnormalStatus'] = 0

    df_load.loc[
        (df_load['hp1_watchdogcode'] > 0) |
        ((df_load['hp2_watchdogcode'] > 0) & df_load['hp2_watchdogcode'].notna()) |
        (df_load['qc_systemwatchdogcode'] > 0),  
        'WatchdogAbnormalStatus'
    ] = 1

    df_load.loc[
        ((df_load['hp1_watchdogcode'] > 0) & ~df_load['hp1_watchdogcode'].isin([8, 9])) |
        ((df_load['hp2_watchdogcode'].notna()) & 
         (df_load['hp2_watchdogcode'] > 0) & 
         ~df_load['hp2_watchdogcode'].isin([8, 9])) |
        (df_load['qc_systemwatchdogcode'] > 0),  
        'SevereWatchdogAbnormalStatus'
    ] = 1

    df_load['WatchdogAbnormalStatus'] = df_load['WatchdogAbnormalStatus'].astype(int)
    df_load['SevereWatchdogAbnormalStatus'] = df_load['SevereWatchdogAbnormalStatus'].astype(int)


def ModelTraining(modelname):
    print("Now path:")
    print(os.getcwd()) 
    data = pd.read_csv("./tutorial/Utility/Heating_specs_Test_data.csv", sep=';').dropna()
    data['Compressor speed（Hz）'] = data['Compressor speed（Hz）'].str.replace('HZ', '').astype(float)
    X = np.array( data[['Outdoor temperature  (℃)', 'Outlet water temp（℃）', 'Compressor speed（Hz）', 'Inlet water temperature（℃）']] )
    y = np.array( data['COP （with water pump）'] )

    match modelname:
        case "Linear":
            #print("\033[31mLinear Model\033[0m")
            model = LinearRegression()
        case "Polynomial":
            #print("\033[31mPolynomial Model\033[0m")
            poly = PolynomialFeatures(degree=2)
            model = make_pipeline(poly, LinearRegression())
        case "SupportVectorRegression":
            #print("\033[31mSupport Vector Regression Model\033[0m")
            model = SVR(kernel='rbf')
        case "MLP":
            #print("\033[31mMLP Model\033[0m")
            model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000)
        case "DecisionTreeRegression":
            #print("\033[31mDTR Model\033[0m")
            model = DecisionTreeRegressor()
        case "RandomForestRegressor": 
            #print("\033[31mRandom Forest Model\033[0m")
            model = RandomForestRegressor(n_estimators=100)
 
    model.fit(X, y)

    '''
    # Print Model Result
    print("model score:", model.score(X, y))
    print('\n')
    # Get Linear model
    linear_model = model.named_steps['linearregression']
    # Get coefficients and intercept
    coefficients = linear_model.coef_  # 线性回归的系数
    intercept = linear_model.intercept_  # 线性回归的截距
    # Get features
    poly_features = model.named_steps['polynomialfeatures']
    X_df = pd.DataFrame(X, columns=[ 'OutdoorTemperature', 'OutletWaterTemp', 'CompressorSpeed', 'InletWaterTemperature']) 
    feature_names = poly_features.get_feature_names_out(input_features=X_df.columns)
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    print(f"Intercept: {intercept}")
    print(coef_df)
    '''

    return model


def ModelPredicting(df_load, model, MergeWindow_BinEdge, cic):

    # Preparing Input 
    if (df_load['hp2_outlettemperaturefiltered'].notnull().all() and
        df_load['hp2_compressorfrequency'].notnull().all() and
        df_load['hp2_inlettemperaturefiltered'].notnull().all()):
        new_data = pd.concat([
            GetTemperatureOutside(df_load),
            (df_load['hp1_outlettemperaturefiltered'] + df_load['hp2_outlettemperaturefiltered']) / 2,
            (df_load['hp1_compressorfrequency'] + df_load['hp2_compressorfrequency']) / 2,
            (df_load['hp1_inlettemperaturefiltered'] + df_load['hp2_inlettemperaturefiltered']) / 2
        ], axis=1, keys=['outdoor_temperature', 'hp1_outlettemperaturefiltered', 'hp1_compressorfrequency', 'hp1_inlettemperaturefiltered'])
    else:
        new_data = pd.concat([
            GetTemperatureOutside(df_load),
            df_load['hp1_outlettemperaturefiltered'],
            df_load['hp1_compressorfrequency'],
            df_load['hp1_inlettemperaturefiltered']
        ], axis=1, keys=['outdoor_temperature', 'hp1_outlettemperaturefiltered', 'hp1_compressorfrequency', 'hp1_inlettemperaturefiltered'])


    # Merge for Time Window
    labels = range(len(MergeWindow_BinEdge) - 1) 
    new_data['group'] = pd.cut(new_data.index, bins=MergeWindow_BinEdge, labels=labels, right=True, include_lowest=True) # each MergedWindow is one group

    #new_data['group'] = new_data['group'].fillna(0).astype(int)
    new_data['group'] = pd.to_numeric(new_data['group'], errors='coerce').fillna(0).astype(int)
    #new_data['group'] = new_data['group'].astype(int)


    # Use Mean value for each MergedWindow
    new_data = new_data.groupby('group').mean() 

    # Predict
    predicted_value = model.predict(new_data.values)

    if any(math.isnan(x) for x in predicted_value):
        print("\033[1;31mFind Nan Value in Expected COP.:\033[0m")

    return predicted_value, new_data['outdoor_temperature']

def GetAveragedOutsideTemperature(df_load, MergeWindow_BinEdge, cic):

    new_data = pd.concat([ GetTemperatureOutside(df_load) ], axis=1, keys=['outdoor_temperature'])

    labels = range(len(MergeWindow_BinEdge) - 1)
    new_data['group'] = pd.cut(new_data.index, bins=MergeWindow_BinEdge, labels=labels, right=True, include_lowest=True) # each MergedWindow is one group
    
    #new_data['group'] = new_data['group'].astype(int)
    new_data['group'] = pd.to_numeric(new_data['group'], errors='coerce').fillna(0).astype(int)

    new_data = new_data.groupby('group').mean()

    return new_data['outdoor_temperature'] 

def CalculateSlidingWindow_ChangeANDPercentage(BinCenterForMergeWindow, freqvalue, df_load, window_length, step_size, columnname): 

    window_centers    = []
    percentage_counts = []
    change_counts     = []

    start_time = BinCenterForMergeWindow.iloc[0] - pd.Timedelta(freqvalue)/2 
    end_time   = BinCenterForMergeWindow.iloc[-1] + pd.Timedelta(freqvalue)/2 
    current_time = start_time

    for bincenter in BinCenterForMergeWindow:
        window_start = bincenter - pd.Timedelta(freqvalue) / 2
        window_end   = bincenter + pd.Timedelta(freqvalue) / 2 

        window_data = df_load[(df_load['client_time'] >= window_start) & (df_load['client_time'] < window_end)]
        
        if len(window_data) > 0:
            StatusOn_percentage_counts = window_data[columnname].sum() / len(window_data) # (* 100) turns to percentage %  
            Status_change_count        = window_data[columnname].diff().abs().sum()   # Question: shoud I /2? Each change is 0->1 or 1->0, /2 is a cycle 0->1->0
        else:
            StatusOn_percentage_counts = 0  
            Status_change_count = 0
        
        window_center = window_start + (window_length / 2)
        window_centers.append(window_center)
        percentage_counts.append(StatusOn_percentage_counts)
        change_counts.append(Status_change_count) 
 
        current_time += step_size

    CheckBinXY(BinCenterForMergeWindow, change_counts)
    CheckBinXY(BinCenterForMergeWindow, percentage_counts)
    CheckBinXY(BinCenterForMergeWindow, window_centers)

    return percentage_counts, change_counts


def CalculateCOPRatioAndUncertainty(ExpectedCOP, MeasuredCOP, x_time_MergedWindow, CL):

    COPRatio = np.where(np.isnan(ExpectedCOP) | np.isnan(MeasuredCOP), np.nan, np.array(MeasuredCOP) / np.array(ExpectedCOP))
    Index_nonNaN = ~np.isnan(COPRatio)

    model = SVR(kernel='rbf')
    x_time_numeric = x_time_MergedWindow[Index_nonNaN].values.astype('datetime64[s]').astype('int').reshape(-1, 1)
    model.fit(x_time_numeric, COPRatio[Index_nonNaN])
    COPRatioFitted = model.predict(x_time_numeric)

    residuals = COPRatio[Index_nonNaN] - COPRatioFitted
    std_error = np.std(residuals) / np.sqrt( len(residuals) )
 
    CL_list     = [50,       80,    85,    90,    92,    95,    98,    99]
    ZScore_list = [0.674, 1.282, 1.440, 1.645, 1.751, 1.960, 2.326, 2.576]
    if CL in CL_list:
        index  = CL_list.index(CL)
        ZScore = ZScore_list[index]
    else:
        print("CL is not in the list.")        

    confidence_interval = ZScore * std_error
    lower_bound = COPRatioFitted + confidence_interval
    upper_bound = COPRatioFitted - confidence_interval
    return COPRatio, COPRatioFitted, lower_bound, upper_bound, Index_nonNaN


def CalculateBootstrapLimit(Nth_elements):
    B = 1000  # Bootstrap resample times
    bootstrap_means = np.empty(B)
    np.random.seed(42) 
    for i in range(B):
        sample = np.random.choice(Nth_elements, size=len(Nth_elements), replace=True)
        bootstrap_means[i] = np.mean(sample)  
    CI_lower_Bootstrap = np.percentile(bootstrap_means, 2.5)
    CI_upper_Bootstrap = np.percentile(bootstrap_means, 97.5)
    return CI_upper_Bootstrap, CI_lower_Bootstrap


def GetCiCNumbersInMergedWindows(BinCenterForMergeWindow_WholeTimeRange, BinCenterForMergeWindow_allCiC_RespectiveBins):

    totalCiCNumbersInMergedWindows = np.zeros( len(list(BinCenterForMergeWindow_WholeTimeRange)), dtype=np.float64)

    for idx, BinCenterForMergeWindow_iter in enumerate(BinCenterForMergeWindow_allCiC_RespectiveBins):
        print("BinCenterForMergeWindow_iter: " + str(BinCenterForMergeWindow_iter))
        indexes = np.array( BinCenterForMergeWindow_iter.apply(lambda x: BinCenterForMergeWindow_WholeTimeRange[BinCenterForMergeWindow_WholeTimeRange == x].index[0]) )
        totalCiCNumbersInMergedWindows[indexes] += 1

    return totalCiCNumbersInMergedWindows



def CalculateKPI_Mean_Uncertainty(BinCenterForMergeWindow_WholeTimeRange, BinCenterForMergeWindow_allCiC_RespectiveBins, KPI_allCiC_RespectiveBins):

    SEM_all = []
    mean_value_all = []
    CI_lower_Bootstrap_all = []
    CI_upper_Bootstrap_all = []

    for Nth_Projection in range(len(BinCenterForMergeWindow_WholeTimeRange)):
        Nth_elements = []
        for idx, (BinCenterForMergeWindow_iter, KPI_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, KPI_allCiC_RespectiveBins)):
            if BinCenterForMergeWindow_WholeTimeRange[Nth_Projection] in np.array(BinCenterForMergeWindow_iter):
                index = np.where(BinCenterForMergeWindow_iter == BinCenterForMergeWindow_WholeTimeRange[Nth_Projection])[0][0]
                '''
                print(len(KPI_iter))
                print("index:" + str(index))
                print("BinCenterForMergeWindow_iter" + str(BinCenterForMergeWindow_iter))
                print("BinCenterForMergeWindow_WholeTimeRange[Nth_Projection]" + str(BinCenterForMergeWindow_WholeTimeRange[Nth_Projection]))
                #if len(KPI_iter) == 0:
                '''
                if index >= len(KPI_iter): # FIXME
                    Nth_elements.append(0)
                else:
                    Nth_elements.append(KPI_iter[index])
        Nth_elements = np.array(Nth_elements)
            
        Nth_elements = Nth_elements[~np.isnan(Nth_elements)]
        #print("Nth_Projection:" + str(Nth_Projection))
        #print("Nth_elements:" + str(Nth_elements))
        mean_value = np.mean(Nth_elements)
        #print("mean_value:" + str(mean_value))
        std_dev = np.std(Nth_elements, ddof=1) 
        N = len(Nth_elements)
        SEM = std_dev / np.sqrt(N)

        mean_value_all.append(mean_value)
        SEM_all.append(SEM)

        CI_upper_Bootstrap, CI_lower_Bootstrap = CalculateBootstrapLimit(Nth_elements)
        CI_upper_Bootstrap_all.append(CI_upper_Bootstrap)
        CI_lower_Bootstrap_all.append(CI_lower_Bootstrap)


    SEM_all = np.array(SEM_all)
    mean_value_all = np.array(mean_value_all)

    #upper_limit = mean_value_all + SEM_all
    #lower_limit = mean_value_all - SEM_all

    z_score = norm.ppf(0.975)  # 95% Z score
    upper_limit = mean_value_all + z_score * SEM_all
    lower_limit = mean_value_all - z_score * SEM_all

    #upper_limit = CI_upper_Bootstrap_all
    #lower_limit = CI_lower_Bootstrap_all

    return mean_value_all, upper_limit, lower_limit


def SaveKPIResultsInCSVFile(BinCenterForMergeWindow_WholeTimeRange, BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIName, KPIlist):

    # Save Results
    print('\n')
    print("\033[1;31mSaving Results now." + "\033[0m")

    # Create a Restult CSV file
    csv_file_path = "../Results/KPIResult.csv"

    column_order = ['MeasuredCOP', 'ExpectedCOP', 'COPPerformance', 'OutsideTemperature', 'Heating_cycles_HP', 'Time_by_HP', 'Heating_cycles_boiler', 'Time_by_boiler', 'Heating_cycles_HP_and_boiler', 'Time_by_HP_and_boiler', 'Watchdog_code_changes', 'Time_with_watchdog', 'Severe_watchdog_code_changes', 'Time_with_severe_watchdog', 'Heat_by_boiler', 'Heat_by_HP', 'Tracking_error', 'Rise_time', 'SetPointTemperatureChange', 'House_Insulation_Indicator', 'Heat_distribution_system_capacity', 'SSR_House_insulation', 'SSR_Heat_distribution', 'SSR_Tracking_error', 'BoilerUsage_HighDemand', 'BoilerUsage_Limited_by_COP', 'BoilerUsage_WaterTempTooHigh', 'BoilerUsage_PreHeating', 'BoilerUsage_Anomaly', 'BoilerUsage_NoReason', 'BoilerUsage_TotalBoilerTime', 'BoilerUsage_Heat_HighDemand', 'BoilerUsage_Heat_Limited_by_COP', 'BoilerUsage_Heat_WaterTempTooHigh', 'BoilerUsage_Heat_PreHeating', 'BoilerUsage_Heat_Anomaly', 'BoilerUsage_Heat_NoReason', 'BoilerUsage_TotalBoilerHEAT']

    BinCenterForMergeWindow_WholeTimeRange_repeated = BinCenterForMergeWindow_WholeTimeRange.repeat(len(final_ciclist)).reset_index(drop=True)
    final_ciclist_tiled = pd.Series(final_ciclist * len(BinCenterForMergeWindow_WholeTimeRange))
    num_entries = len(BinCenterForMergeWindow_WholeTimeRange_repeated)

    # Create if not existed
    if not os.path.exists(csv_file_path):
        data = {
            "TimeStamp": BinCenterForMergeWindow_WholeTimeRange_repeated,
            "CICID": final_ciclist_tiled,
            **{col: np.full(num_entries, np.nan) for col in column_order}
            }
        KPIResult_empty = pd.DataFrame(data).set_index("TimeStamp")
        KPIResult_empty.to_csv(csv_file_path)
  
    # Write values into CSV file
    KPIResult = pd.read_csv(csv_file_path)
    KPIResult['TimeStamp'] = pd.to_datetime(KPIResult['TimeStamp'])

    # Create empty columns if not existed in current CSV file
    for col in column_order:
        if col not in KPIResult.columns:
            KPIResult[col] = np.nan


    if KPIName == "COPPerformance":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, MeasuredCOP_iter, ExpectedCOP_iter, COPRatio_iter, OutsideTemperature_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['MeasuredCOP'], KPIlist['ExpectedCOP'], KPIlist['COPPerformance_COPRatio'], KPIlist['COPPerformance_OutsideT'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp"       : timebincenter,
                    "CICID"           : cic_iter,
                    "MeasuredCOP" : MeasuredCOP_iter[index],
                    "ExpectedCOP" : ExpectedCOP_iter[index],
                    "COPPerformance" : COPRatio_iter[index],
                }
                KPIResult.loc[condition, ["MeasuredCOP", "ExpectedCOP", "COPPerformance"]] = [new_row['MeasuredCOP'], new_row['ExpectedCOP'], new_row['COPPerformance']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)

    elif KPIName == "OutsideTemperature":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, OutsideTemperature_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['OutsideTemperature'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_iter,
                    "OutsideTemperature"    : OutsideTemperature_iter[index],
                }
                KPIResult.loc[condition, ["OutsideTemperature"]] = [new_row['OutsideTemperature']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)

    elif KPIName == "HeatingCycle":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, HeatingCycleChangeCounts_boiler_iter, HeatingCyclePercentageCounts_boiler_iter, HeatingCycleChangeCounts_HP_iter, HeatingCyclePercentageCounts_HP_iter, HeatingCycleChangeCounts_HP_and_boiler_iter, HeatingCyclePercentageCounts_HP_and_boiler_iter) in enumerate(zip( BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['HeatingCycle_ChangeCounts_boiler'], KPIlist['HeatingCycle_PercentageCounts_boiler'], KPIlist['HeatingCycle_ChangeCounts_HP'], KPIlist['HeatingCycle_PercentageCounts_HP'], KPIlist['HeatingCycle_ChangeCounts_HP_and_boiler'], KPIlist['HeatingCycle_PercentageCounts_HP_and_boiler'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_iter,
                    "Heating_cycles_HP"    : HeatingCycleChangeCounts_HP_iter[index],
                    "Time_by_HP"           : HeatingCyclePercentageCounts_HP_iter[index],
                    "Heating_cycles_boiler": HeatingCycleChangeCounts_boiler_iter[index],
                    "Time_by_boiler"       : HeatingCyclePercentageCounts_boiler_iter[index],
                    "Heating_cycles_HP_and_boiler" : HeatingCycleChangeCounts_HP_and_boiler_iter[index],
                    "Time_by_HP_and_boiler"        : HeatingCyclePercentageCounts_HP_and_boiler_iter[index]
                } 
                KPIResult.loc[condition, ["Heating_cycles_HP"           , "Time_by_HP"]]            = [new_row['Heating_cycles_HP']           , new_row['Time_by_HP']]
                KPIResult.loc[condition, ["Heating_cycles_boiler"       , "Time_by_boiler"]]        = [new_row['Heating_cycles_boiler']       , new_row['Time_by_boiler']]
                KPIResult.loc[condition, ["Heating_cycles_HP_and_boiler", "Time_by_HP_and_boiler"]] = [new_row['Heating_cycles_HP_and_boiler'], new_row['Time_by_HP_and_boiler']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True) 

    elif KPIName == "WatchdogChange":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, WatchdogChangeCounts_iter, WatchdogPercentageCounts_iter, severeWatchdogChangeCounts_iter, severeWatchdogPercentageCounts_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['WatchdogChange_ChangeCounts'], KPIlist['WatchdogChange_PercentageCounts'], KPIlist['SevereWatchdogChange_ChangeCounts'], KPIlist['SevereWatchdogChange_PercentageCounts'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_iter,
                    "Watchdog_code_changes"  : WatchdogChangeCounts_iter[index],
                    "Time_with_watchdog"   : WatchdogPercentageCounts_iter[index],
                    "Severe_watchdog_code_changes" : severeWatchdogChangeCounts_iter[index],
                    "Time_with_severe_watchdog" : severeWatchdogPercentageCounts_iter[index]
                } 
                KPIResult.loc[condition, ["Watchdog_code_changes", "Time_with_watchdog", "Severe_watchdog_code_changes", "Time_with_severe_watchdog"]] = [new_row['Watchdog_code_changes'], new_row['Time_with_watchdog'], new_row['Severe_watchdog_code_changes'], new_row['Time_with_severe_watchdog']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)

    elif KPIName == "RiseTime":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, RisingTime_iter, SetpointT_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['RisingTime'], KPIlist['SetPointTemperatureChange'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_iter,
                    "Rise_time"     : RisingTime_iter[index],
                    "SetPointTemperatureChange" : SetpointT_iter[index]
                } 
                KPIResult.loc[condition, ["Rise_time", "SetPointTemperatureChange"]] = [new_row['Rise_time'], new_row['SetPointTemperatureChange']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)


    elif KPIName == "TwoIndicators":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, Tracking_error_iter, InsulationIndicator_iter, HeatDissipationIndicator_iter, SSR_Insulation_iter, SSR_HeatDissipation_iter, SSR_TrackingError_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['TrackingError'], KPIlist['InsulationIndicator'], KPIlist['HeatDissipationIndicator'], KPIlist['SSR_Insulation'], KPIlist['SSR_HeatDissipation'], KPIlist['SSR_TrackingError'] )):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_iter,
                    "Tracking_error" : Tracking_error_iter[index],
                    "House_Insulation_Indicator"     : InsulationIndicator_iter[index],
                    "Heat_distribution_system_capacity": HeatDissipationIndicator_iter[index],
                    "SSR_House_insulation"          : SSR_Insulation_iter[index],
                    "SSR_Heat_distribution"     : SSR_HeatDissipation_iter[index],
                    "SSR_Tracking_error"       : SSR_TrackingError_iter[index],
                }

                KPIResult.loc[condition, ["Tracking_error", "House_Insulation_Indicator", "Heat_distribution_system_capacity", "SSR_House_insulation", "SSR_Heat_distribution", "SSR_Tracking_error"]] = [new_row['Tracking_error'], new_row['House_Insulation_Indicator'], new_row['Heat_distribution_system_capacity'], new_row['SSR_House_insulation'], new_row['SSR_Heat_distribution'], new_row['SSR_Tracking_error']] 
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)


    elif KPIName == "HeatPercentage":
        new_rows = []  
        for idx, (BinCenterForMergeWindow_iter, cic_iter, GasPercentage_iter, HeatPumpPercentage_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['HeatPercentage_gas'], KPIlist['HeatPercentage_heatpump'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_iter,
                    "Heat_by_boiler": GasPercentage_iter[index],
                    "Heat_by_HP": HeatPumpPercentage_iter[index],
                }
                KPIResult.loc[condition, ['Heat_by_boiler', 'Heat_by_HP']] = [new_row['Heat_by_boiler'], new_row['Heat_by_HP']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows: 
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)      
                
    elif KPIName == "GasUseageCase":
        # print(KPIResult)
        # print(KPIlist['boiler_usage_time'])
        # print(KPIlist['boiler_usage_heat'])
        # print(BinCenterForMergeWindow_allCiC_RespectiveBins)
        new_rows = []
        
        for (cic_time, values_time), (cic_heat, values_heat), BinCenterForMergeWindow_iter in zip(
            KPIlist['boiler_usage_time'].items(), 
            KPIlist['boiler_usage_heat'].items(), 
            BinCenterForMergeWindow_allCiC_RespectiveBins):
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter): 
                condition = (KPIResult['TimeStamp'] == timebincenter) & (KPIResult['CICID'] == cic_time)     
                new_row = {
                    "TimeStamp": timebincenter,
                    "CICID": cic_time,
                    "BoilerUsage_HighDemand"      : values_time['HighDemand'][index],
                    "BoilerUsage_Limited_by_COP"  : values_time['Limited_by_COP'][index],
                    "BoilerUsage_WaterTempTooHigh": values_time['WaterTempTooHigh'][index],
                    "BoilerUsage_PreHeating"      : values_time['PreHeating'][index],
                    "BoilerUsage_Anomaly"         : values_time['Anomaly'][index],
                    "BoilerUsage_NoReason"        : values_time['NoReason'][index],
                    "BoilerUsage_TotalBoilerTime" : values_time['TotalBoilerTime'][index],

                    "BoilerUsage_Heat_HighDemand"       : values_heat['HighDemand'][index],
                    "BoilerUsage_Heat_Limited_by_COP"   : values_heat['Limited_by_COP'][index],
                    "BoilerUsage_Heat_WaterTempTooHigh" : values_heat['WaterTempTooHigh'][index],
                    "BoilerUsage_Heat_PreHeating"       : values_heat['PreHeating'][index],
                    "BoilerUsage_Heat_Anomaly"          : values_heat['Anomaly'][index],
                    "BoilerUsage_Heat_NoReason"         : values_heat['NoReason'][index],
                    "BoilerUsage_TotalBoilerHEAT"  : values_heat['TotalBoilerHEAT'][index]
                }    
                KPIResult.loc[condition, ['BoilerUsage_HighDemand', 'BoilerUsage_Limited_by_COP', 'BoilerUsage_WaterTempTooHigh', 'BoilerUsage_PreHeating', 'BoilerUsage_Anomaly', 'BoilerUsage_NoReason', 'BoilerUsage_TotalBoilerTime', 'BoilerUsage_Heat_HighDemand', 'BoilerUsage_Heat_Limited_by_COP', 'BoilerUsage_Heat_WaterTempTooHigh', 'BoilerUsage_Heat_PreHeating', 'BoilerUsage_Heat_Anomaly', 'BoilerUsage_Heat_NoReason', 'BoilerUsage_TotalBoilerHEAT']] = [ new_row['BoilerUsage_HighDemand'], new_row['BoilerUsage_Limited_by_COP'], new_row['BoilerUsage_WaterTempTooHigh'], new_row['BoilerUsage_PreHeating'], new_row['BoilerUsage_Anomaly'], new_row['BoilerUsage_NoReason'], new_row['BoilerUsage_TotalBoilerTime'], new_row['BoilerUsage_Heat_HighDemand'], new_row['BoilerUsage_Heat_Limited_by_COP'], new_row['BoilerUsage_Heat_WaterTempTooHigh'], new_row['BoilerUsage_Heat_PreHeating'], new_row['BoilerUsage_Heat_Anomaly'], new_row['BoilerUsage_Heat_NoReason'], new_row['BoilerUsage_TotalBoilerHEAT'] ]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)

    # Save to path                
    KPIResult.set_index("TimeStamp", inplace=True)
    KPIResult.sort_index(inplace=True)
    KPIResult.to_csv(csv_file_path, columns= (["CICID"] + column_order))    

    # End message
    print("\033[1;31mKPIResult: \033[35m" + str(np.array(KPIResult).shape) + "\033[0m")
    print("\033[1;31mSave to: \033[35m" + f"{csv_file_path}" + "\033[0m")


def calculate_list_ratios(nominator, *lists):
    """
    Calculate the ratio of each element in a list of lists compared to the sum of corresponding elements in other lists.

    Args:
        *lists: Multiple lists of lists with potentially different lengths.

    Returns:
        A list of lists, where each element is the ratio of the corresponding element in the input lists.
    """
    # Check if there are at least two lists provided
    if len(lists) < 2:
        raise ValueError("At least two lists must be provided.")

    # Ensure all provided arguments are lists
    for lst in lists:
        if not isinstance(lst, list):
            raise TypeError("All inputs must be lists of lists.")

    # Calculate the ratios
    result = [
        [
            elem / sum(values) if sum(values) != 0 else 0  
            for values in zip(*rows)
            for elem in [values[nominator]]
        ]
        for rows in zip(*lists)
    ]

    return result



def calculate_boiler_heat_code0(df_OneCiC_InOneMergeWindow, reason):

    time_differences = 0
    heat_differences = []


    for i in range(len(df_OneCiC_InOneMergeWindow) - 1):
        hp2_power = df_OneCiC_InOneMergeWindow['hp2_ratedpower'][i]
        
        if isinstance(hp2_power, (int, float)) and not isinstance(hp2_power, bool):
            hp2_power = hp2_power if not np.isnan(hp2_power) else 0
        else:
            hp2_power = 0  
        
        total_rated_power = df_OneCiC_InOneMergeWindow['hp1_ratedpower'][i] + hp2_power


        condition_high_demand = (
            df_OneCiC_InOneMergeWindow['hp1_watchdogcode'][i] == 0 and

            ((
            total_rated_power < df_OneCiC_InOneMergeWindow['qc_estimatedpowerdemand'][i] * 1.1 and
            total_rated_power > 2000) 
            # or total_rated_power == 0
            ) and

            df_OneCiC_InOneMergeWindow['hp1_limitedbycop'][i] == 0 and  

            (pd.isna(df_OneCiC_InOneMergeWindow['hp2_watchdogcode'][i]) or 
             df_OneCiC_InOneMergeWindow['hp2_watchdogcode'][i] == 0)
        )


        condition_high_LimitedByCop = (
            df_OneCiC_InOneMergeWindow['hp1_watchdogcode'][i] == 0 and
            (pd.isna(df_OneCiC_InOneMergeWindow['hp2_watchdogcode'][i]) or
             df_OneCiC_InOneMergeWindow['hp2_watchdogcode'][i] == 0) and

            df_OneCiC_InOneMergeWindow['hp1_limitedbycop'][i] == 1
        )

        condition_no_reason = (
            df_OneCiC_InOneMergeWindow['hp1_watchdogcode'][i] == 0 and
            (pd.isna(df_OneCiC_InOneMergeWindow['hp2_watchdogcode'][i]) or
             df_OneCiC_InOneMergeWindow['hp2_watchdogcode'][i] == 0) and

            (not condition_high_demand) and 
            (not condition_high_LimitedByCop)
        )

        #print("total_rated_power: " + str(total_rated_power))

        if reason == 'high_demand':
            is_valid = condition_high_demand
            #print("hp1_limitedbycop: " + str( df_OneCiC_InOneMergeWindow['hp1_limitedbycop'][i] ))
        elif reason == 'limited_by_COP':
            is_valid = condition_high_LimitedByCop
        elif reason == 'no_reason':
            is_valid = condition_no_reason
        else:
            raise ValueError(f"Invalid reason: {reason}")

        #print('reason:' + str(reason)) 
        #print('condition:' + str(condition))

        if is_valid:
            diff = (
                df_OneCiC_InOneMergeWindow.iloc[i + 1]['qc_cvenergycounter'] -
                df_OneCiC_InOneMergeWindow.iloc[i]['qc_cvenergycounter']
            )
            heat_differences.append(diff)
            time_differences = time_differences + 1

    return sum(heat_differences), time_differences


def calculate_boiler_heat_NonCode0(df_OneCiC_InOneMergeWindow, hp_watchdogcodelist, exclude):
    if exclude == False:
        indices = []
        for idx, row in df_OneCiC_InOneMergeWindow.iterrows():
            hp1_in_list = row['hp1_watchdogcode'] in hp_watchdogcodelist
            hp2_is_null = pd.isnull(row['hp2_watchdogcode'])
            hp2_in_list = row['hp2_watchdogcode'] in hp_watchdogcodelist if not hp2_is_null else False

            # Condition 1: hp2_watchdogcode is null and hp1_watchdogcode in hp_watchdogcodelist
            if hp2_is_null and hp1_in_list:
                indices.append(idx)
            # Condition 2: hp2_watchdogcode is not null and both hp1_watchdogcode and hp2_watchdogcode in hp_watchdogcodelist
            elif not hp2_is_null and hp1_in_list and hp2_in_list:
                indices.append(idx)
            # Condition 3: hp2_watchdogcode is not null, one of hp1_watchdogcode or hp2_watchdogcode in hp_watchdogcodelist, and the other is 0
            elif not hp2_is_null and (
                (hp1_in_list and row['hp2_watchdogcode'] == 0) or (hp2_in_list and row['hp1_watchdogcode'] == 0)
            ):
                indices.append(idx)
    
    elif exclude == True:
        indices = []
        for idx, row in df_OneCiC_InOneMergeWindow.iterrows():
            hp1_in_list = row['hp1_watchdogcode'] in hp_watchdogcodelist
            hp2_is_null = pd.isnull(row['hp2_watchdogcode'])
            hp2_in_list = row['hp2_watchdogcode'] in hp_watchdogcodelist if not hp2_is_null else False

            # Condition 1: hp2_watchdogcode is null and hp1_watchdogcode not in hp_watchdogcodelist
            if hp2_is_null and not hp1_in_list:
                indices.append(idx)
            # Condition 2: hp2_watchdogcode is not null and at least one of hp1_watchdogcode or hp2_watchdogcode in hp_watchdogcodelist
            elif not hp2_is_null and (hp1_in_list or hp2_in_list):
                indices.append(idx)
            # Condition 3: hp2_watchdogcode is not null, one of hp1_watchdogcode or hp2_watchdogcode in [8, 9], and the other in [10, 15, 21, 103]
            elif not hp2_is_null and (
                (row['hp1_watchdogcode'] in [8, 9] and row['hp2_watchdogcode'] in [10, 15, 21, 103]) or
                (row['hp2_watchdogcode'] in [8, 9] and row['hp1_watchdogcode'] in [10, 15, 21, 103])
            ):
                indices.append(idx)

    heat_differences = []
    time_differences = 0

    for idx in indices:
        if idx + 1 < len(df_OneCiC_InOneMergeWindow):
            diff = df_OneCiC_InOneMergeWindow.iloc[idx + 1]['qc_cvenergycounter'] - df_OneCiC_InOneMergeWindow.iloc[idx]['qc_cvenergycounter']
            heat_differences.append(diff)
            time_differences = time_differences + 1 
    return sum(heat_differences), time_differences

'''
def calculate_total_delivered_heat(df_load, start_index, end_index):
    HP1_delivered_heat = df_load['hp1_thermalenergycounter'].iloc[end_index] - df_load['hp1_thermalenergycounter'].iloc[start_index]

    if (np.isnan(df_load['hp2_thermalenergycounter'].iloc[end_index]) == False) and (np.isnan(df_load['hp2_thermalenergycounter'].iloc[start_index]) == False):
        HP2_delivered_heat = df_load['hp2_thermalenergycounter'].iloc[end_index] - df_load['hp2_thermalenergycounter'].iloc[start_index]
    else:
        HP2_delivered_heat = 0
    Boiler_delivered_heat = df_load['qc_cvenergycounter'].iloc[end_index] - df_load['qc_cvenergycounter'].iloc[start_index]    
    return HP1_delivered_heat + HP2_delivered_heat + Boiler_delivered_heat
'''


def calculate_total_delivered_heat(df_load, start_index, end_index):
    df_load['hp1_thermalenergycounter'] = pd.to_numeric(df_load['hp1_thermalenergycounter'], errors='coerce')
    df_load['hp2_thermalenergycounter'] = pd.to_numeric(df_load['hp2_thermalenergycounter'], errors='coerce')
    df_load['qc_cvenergycounter'] = pd.to_numeric(df_load['qc_cvenergycounter'], errors='coerce')
    
    HP1_delivered_heat = df_load['hp1_thermalenergycounter'].iloc[end_index] - df_load['hp1_thermalenergycounter'].iloc[start_index]
    
    hp2_start = df_load['hp2_thermalenergycounter'].iloc[start_index]
    hp2_end = df_load['hp2_thermalenergycounter'].iloc[end_index]
    
    if pd.notna(hp2_start) and pd.notna(hp2_end):
        HP2_delivered_heat = hp2_end - hp2_start
    else:
        HP2_delivered_heat = 0
    
    Boiler_delivered_heat = df_load['qc_cvenergycounter'].iloc[end_index] - df_load['qc_cvenergycounter'].iloc[start_index]
    
    return HP1_delivered_heat + HP2_delivered_heat + Boiler_delivered_heat


