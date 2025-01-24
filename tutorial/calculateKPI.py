import KPIUtility
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utility.Plot_Functions as Plot_function

def calculate_gas_useage_case_time_and_heat(MergeWindow_BinEdge, df_OneCiC):
    #### Gas boiler usage:
    # HighDemand: 1). hp1_watchdogcode=0, hp2_watchdogcode=null or hp1_watchdogcode=hp2_watchdogcode=0 2). ratedpower > estimated power 3).limited_by_COP=0
    # Limited_by_COP: 1). hp1_watchdogcode=0, hp2_watchdogcode=null or hp1_watchdogcode=hp2_watchdogcode=0 2). limited_by_COP=1
    # WaterTempTooHigh: 1). both in [8,9] 2). one is in [8,9], the other is 0
    # PreHeating: 1). both in [10, 15, 21, 103] 2). one in [10, 15, 21, 103], the other is 0
    # Anomaly: 1). either of two is not in [0, 8, 9, 10, 15, 21, 103] 2). One is [8,9], the other is [10, 15, 21, 103]
    # NoReason: 1). hp1_watchdogcode=0, hp2_watchdogcode=null or hp1_watchdogcode=hp2_watchdogcode=0 2). ratedpower < estimated power

    boiler_usage_thisCiC_RespectiveBins = {
                 "HighDemand": [],
                 "Limited_by_COP" : [],
                 "WaterTempTooHigh": [],
                 "PreHeating": [],
                 "Anomaly": [],
                 "NoReason": [],
                 "TotalBoilerTime": []
             }

    boiler_usage_heat_thisCiC_RespectiveBins = {
                 "HighDemand": [],
                 "Limited_by_COP" : [],
                 "WaterTempTooHigh": [],
                 "PreHeating": [],
                 "Anomaly": [],
                 "NoReason": [],
                 "TotalBoilerHEAT": []
             }

    known_states_list = [0, 8, 9, 10, 15, 21, 103] 

    for index_MergeWindow in range(len(MergeWindow_BinEdge)-1):

        # Get data in this Merge Window
        startindex = int(MergeWindow_BinEdge[index_MergeWindow])
        endindex = int(MergeWindow_BinEdge[index_MergeWindow+1])
        if startindex == 0:
            df_OneCiC_InOneMergeWindow = df_OneCiC.loc[ startindex : endindex ]      
        else:
            df_OneCiC_InOneMergeWindow = df_OneCiC.loc[ startindex+1 : endindex ]
        df_OneCiC_InOneMergeWindow = df_OneCiC_InOneMergeWindow.sort_values(by='time_ts').reset_index(drop=True) 

        # DEBUG ONLY, TODO
        # df_OneCiC_InOneMergeWindow = df_OneCiC_InOneMergeWindow[df_OneCiC_InOneMergeWindow["hp2_watchdogCode"].isna()]

        # Get CV heat and time in different causes
        absolute_heat_HighDemand , absolute_time_HighDemand      = KPIUtility.calculate_boiler_heat_code0(df_OneCiC_InOneMergeWindow, 'high_demand')
        absolute_heat_limited_by_COP , absolute_time_limited_by_COP      = KPIUtility.calculate_boiler_heat_code0(df_OneCiC_InOneMergeWindow, 'limited_by_COP')
        absolute_heat_WaterTempTooHigh, absolute_time_WaterTempTooHigh = KPIUtility.calculate_boiler_heat_NonCode0(df_OneCiC_InOneMergeWindow, [8,9], exclude=False) 
        absolute_heat_PreHeating, absolute_time_PreHeating       = KPIUtility.calculate_boiler_heat_NonCode0(df_OneCiC_InOneMergeWindow, [10, 15, 21, 103], exclude=False)
        absolute_heat_Anomaly, absolute_time_Anomaly          = KPIUtility.calculate_boiler_heat_NonCode0(df_OneCiC_InOneMergeWindow, [0, 8, 9, 10, 15, 21, 103], exclude=True)
        absolute_heat_NoReason , absolute_time_NoReason        = KPIUtility.calculate_boiler_heat_code0(df_OneCiC_InOneMergeWindow, 'no_reason')

        absolute_heat_total = absolute_heat_HighDemand + absolute_heat_limited_by_COP + absolute_heat_WaterTempTooHigh + absolute_heat_PreHeating + absolute_heat_Anomaly + absolute_heat_NoReason
        absolute_time_total = absolute_time_HighDemand + absolute_time_limited_by_COP + absolute_time_WaterTempTooHigh + absolute_time_PreHeating + absolute_time_Anomaly + absolute_time_NoReason

        '''
        print("total number:" + str(len(df_OneCiC_InOneMergeWindow)))
        print("absolute_time_total:" + str(absolute_time_total))
        print("Number of left records:" + str( len(df_OneCiC_InOneMergeWindow) - absolute_time_total) )
        '''

        if absolute_heat_total == 0:
            boiler_usage_heat_thisCiC_RespectiveBins['HighDemand'].append(0)
            boiler_usage_heat_thisCiC_RespectiveBins['Limited_by_COP'].append(0)
            boiler_usage_heat_thisCiC_RespectiveBins['WaterTempTooHigh'].append(0)
            boiler_usage_heat_thisCiC_RespectiveBins['PreHeating'].append(0)
            boiler_usage_heat_thisCiC_RespectiveBins['Anomaly'].append(0)
            boiler_usage_heat_thisCiC_RespectiveBins['NoReason'].append(0)
            boiler_usage_heat_thisCiC_RespectiveBins['TotalBoilerHEAT'].append(0)
        else:
            boiler_usage_heat_thisCiC_RespectiveBins['HighDemand'].append(absolute_heat_HighDemand / absolute_heat_total)
            boiler_usage_heat_thisCiC_RespectiveBins['Limited_by_COP'].append(absolute_heat_limited_by_COP / absolute_heat_total)
            boiler_usage_heat_thisCiC_RespectiveBins['WaterTempTooHigh'].append(absolute_heat_WaterTempTooHigh / absolute_heat_total)
            boiler_usage_heat_thisCiC_RespectiveBins['PreHeating'].append(absolute_heat_PreHeating / absolute_heat_total)
            boiler_usage_heat_thisCiC_RespectiveBins['Anomaly'].append(absolute_heat_Anomaly / absolute_heat_total)
            boiler_usage_heat_thisCiC_RespectiveBins['NoReason'].append(absolute_heat_NoReason / absolute_heat_total)
            boiler_usage_heat_thisCiC_RespectiveBins['TotalBoilerHEAT'].append(absolute_heat_total) 

        if absolute_time_total == 0:
            boiler_usage_thisCiC_RespectiveBins['HighDemand'].append(0)
            boiler_usage_thisCiC_RespectiveBins['Limited_by_COP'].append(0)
            boiler_usage_thisCiC_RespectiveBins['WaterTempTooHigh'].append(0)
            boiler_usage_thisCiC_RespectiveBins['PreHeating'].append(0)
            boiler_usage_thisCiC_RespectiveBins['Anomaly'].append(0)
            boiler_usage_thisCiC_RespectiveBins['NoReason'].append(0)
            boiler_usage_thisCiC_RespectiveBins['TotalBoilerTime'].append(0)
        else:
            boiler_usage_thisCiC_RespectiveBins['HighDemand'].append(absolute_time_HighDemand / absolute_time_total)
            boiler_usage_thisCiC_RespectiveBins['Limited_by_COP'].append(absolute_time_limited_by_COP / absolute_time_total)
            boiler_usage_thisCiC_RespectiveBins['WaterTempTooHigh'].append(absolute_time_WaterTempTooHigh /absolute_time_total)
            boiler_usage_thisCiC_RespectiveBins['PreHeating'].append(absolute_time_PreHeating / absolute_time_total)
            boiler_usage_thisCiC_RespectiveBins['Anomaly'].append(absolute_time_Anomaly / absolute_time_total)
            boiler_usage_thisCiC_RespectiveBins['NoReason'].append(absolute_time_NoReason / absolute_time_total)
            boiler_usage_thisCiC_RespectiveBins['TotalBoilerTime'].append(absolute_time_total)
        '''
        print("absolute_time_HighDemand:" + str(absolute_time_HighDemand))
        print("absolute_time_WaterTempTooHigh:" + str(absolute_time_WaterTempTooHigh))
        print("absolute_time_PreHeating:" + str(absolute_time_PreHeating))
        print("absolute_time_Anomaly:" + str(absolute_time_Anomaly))
        print("absolute_time_NoReason:" + str(absolute_time_NoReason))
        print(df_OneCiC_InOneMergeWindow['hp1_watchdogCode'].value_counts())
        print('\n')
        '''
    return boiler_usage_thisCiC_RespectiveBins, boiler_usage_heat_thisCiC_RespectiveBins

def DetermingSteadyStates(df_OneCiC_InOneMergeWindow, window_size, BinCenterForMergeWindow, index_MergeWindow, KPIIndicator_ThisCiC_RespectiveBins, NumberofKPIIndicator_ThisCiC_RespectiveBins, SteadyStateRatio_ThisCiC_RespectiveBins, KPIName): 

    # Start with the begining within this merge window, eg: 1 day
    start_idx = 0
    n = len(df_OneCiC_InOneMergeWindow)   
    totalLengthInSteadyStateinThisMergeWindow = 0 

    while start_idx < n:
        # Start with the minimum steady state window 
        end_idx = start_idx + window_size - 1
        if end_idx >= n:
            break

        # flag to indicate if a steady state is found starting this start index
        found_valid_sequence = False

        # Find longest steady state
        while end_idx < n:
            df_sub = df_OneCiC_InOneMergeWindow.iloc[start_idx:end_idx + 1]
            
            # Get statistics 
            if KPIName == 'InsulationIndicator':
                OutsideTemperature = df_sub['hp1_temperatureOutside']
                thermostat         = df_sub['thermostat_otFtRoomTemperature']
                OutsideTemperature_max, OutsideTemperature_mean, OutsideTemperature_min = OutsideTemperature.max(), OutsideTemperature.mean(), OutsideTemperature.min()
                thermostat_max, thermostat_mean, thermostat_min                        = thermostat.max(), thermostat.mean(), thermostat.min()
                # Try to match steady state condition
                if (OutsideTemperature_max - OutsideTemperature_mean < 0.5 and
                    OutsideTemperature_mean - OutsideTemperature_min < 0.5 and
                    thermostat_max - thermostat_mean < 0.5 and
                    thermostat_mean - thermostat_min < 0.5 and 
                    df_sub['qc_supervisoryControlMode'].isin([2,3,4]).all() and
                    df_sub['thermostat_otFtChEnabled'].all()):
                    # if a steady state found starting this start index, reverse the flag, and extend the searching range
                    found_valid_sequence = True
                    end_idx += 1
                else:
                    # If no steady state found from this start index, or the longest steady state found, then exit
                    break

            elif KPIName == 'Dissipation':
                WaterTemperature = df_sub['qc_supplyTemperatureFiltered']
                thermostat = df_sub['thermostat_otFtRoomTemperature']
                WaterTemperature_max, WaterTemperature_mean, WaterTemperature_min = WaterTemperature.max(), WaterTemperature.mean(), WaterTemperature.min()
                thermostat_max, thermostat_mean, thermostat_min = thermostat.max(), thermostat.mean(), thermostat.min()
                # Try to match steady state condition
                if (WaterTemperature_max - WaterTemperature_mean < 0.5 and
                    WaterTemperature_mean - WaterTemperature_min < 0.5 and
                    thermostat_max - thermostat_mean < 0.5 and
                    thermostat_mean - thermostat_min < 0.5 and
                    df_sub['qc_supervisoryControlMode'].isin([2,3,4]).all() and 
                    df_sub['thermostat_otFtChEnabled'].all()):
                    # if a steady state found starting this start index, reverse the flag, and extend the searching range
                    found_valid_sequence = True
                    end_idx += 1
                else:
                    # If no steady state found from this start index, or the longest steady state found, then exit
                    break

            elif KPIName == 'TrackingError':
                SetpointTemperature = df_sub['thermostat_otFtRoomSetpoint']
                thermostat          = df_sub['thermostat_otFtRoomTemperature']
                SetpointTemperature_max, SetpointTemperature_mean, SetpointTemperature_min = SetpointTemperature.max(), SetpointTemperature.mean(), SetpointTemperature.min()          
                thermostat_max, thermostat_mean, thermostat_min = thermostat.max(), thermostat.mean(), thermostat.min()
                # Try to match steady state condition
                if (SetpointTemperature_max - SetpointTemperature_mean == 0 and
                    SetpointTemperature_mean - SetpointTemperature_min == 0 and
                    thermostat_max - thermostat_mean < 0.2 and
                    thermostat_mean - thermostat_min < 0.2 and
                    df_sub['qc_supervisoryControlMode'].isin([2,3,4]).all() and
                    df_sub['thermostat_otFtChEnabled'].all() ):
                    # if a steady state found starting this start index, reverse the flag, and extend the searching range
                    found_valid_sequence = True
                    end_idx += 1
                else:
                    # If no steady state found from this start index, or the longest steady state found, then exit
                    break

        # Save the result
        if found_valid_sequence == True:

            # Calculate the time spent in steady state
            totalLengthInSteadyStateinThisMergeWindow = totalLengthInSteadyStateinThisMergeWindow + ((end_idx-1) - start_idx)

            if KPIName == 'InsulationIndicator': 
                deliveredHeat = KPIUtility.calculate_total_delivered_heat(df_OneCiC_InOneMergeWindow, start_idx, end_idx-1)
                mean_RoomTemperature = df_OneCiC_InOneMergeWindow.iloc[start_idx:(end_idx-1)]['thermostat_otFtRoomTemperature'].mean()
                mean_OutsideTemperature = df_OneCiC_InOneMergeWindow.iloc[start_idx:(end_idx-1)]['hp1_temperatureOutside'].mean()
                KPIIndicator = deliveredHeat / (mean_RoomTemperature - mean_OutsideTemperature)
            elif KPIName == 'Dissipation':
                deliveredHeat = KPIUtility.calculate_total_delivered_heat(df_OneCiC_InOneMergeWindow, start_idx, end_idx-1)
                mean_RoomTemperature = df_OneCiC_InOneMergeWindow.iloc[start_idx:(end_idx-1)]['thermostat_otFtRoomTemperature'].mean()
                mean_WaterTemperature = df_OneCiC_InOneMergeWindow.iloc[start_idx:(end_idx-1)]['qc_supplyTemperatureFiltered'].mean()
                KPIIndicator = deliveredHeat / (mean_WaterTemperature - mean_RoomTemperature)
            elif KPIName == 'TrackingError':
                KPIIndicator = 0
                data_for_tracking_error = df_OneCiC_InOneMergeWindow.iloc[start_idx:(end_idx-1)]              
                errors = data_for_tracking_error['thermostat_otFtRoomTemperature'] - data_for_tracking_error['thermostat_otFtRoomSetpoint']
                rms = np.sqrt(np.mean(errors**2))
                KPIIndicator = rms

            # if there are multiple steady state in one merged time window, calculate mean of all KPIIndicator in each merged time bin
            NumberofKPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow] = NumberofKPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow] + 1
            if math.isnan(KPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow]):
                KPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow] = KPIIndicator
            else:
                KPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow] = (KPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow] + KPIIndicator) / NumberofKPIIndicator_ThisCiC_RespectiveBins[index_MergeWindow] 

        # Start seaching for next steady state
        start_idx = end_idx

    SteadyStateRatio_ThisCiC_RespectiveBins[index_MergeWindow] = totalLengthInSteadyStateinThisMergeWindow / len(df_OneCiC_InOneMergeWindow) 

    return KPIIndicator_ThisCiC_RespectiveBins, SteadyStateRatio_ThisCiC_RespectiveBins


def calculate_rise_time_and_tracking_error(BinCenterForMergeWindow, MergeWindow_BinEdge, df_OneCiC, cic):

    ## further preparation 
    TrackingError_ThisCiC_RespectiveBins                    = [np.nan] * len(BinCenterForMergeWindow)
    NumberofTrackingError_ThisCiC_RespectiveBins            = [0]      * len(BinCenterForMergeWindow)
    RisingTime_ThisCiC_RespectiveBins                       = [np.nan] * len(BinCenterForMergeWindow)
    NumberofRisingTime_ThisCiC_RespectiveBins               = [0]      * len(BinCenterForMergeWindow)
    SetPointTemperatureChange_ThisCiC_RespectiveBins         = [np.nan] * len(BinCenterForMergeWindow)
    NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins = [0]      * len(BinCenterForMergeWindow)

    ## quality check flag to indicate if the setpoint temperature is changed, therefore the rising time can be calculated.
    # check every 15-second CiC record if: 
    # 1). thermostat_otFtRoomSetpoint is changed in next record, and thermostat_otFtRoomSetpoint in next record is at least 1 degree higher than current thermostat_otFtRoomSetpoint. 
    # 2). thermostat_otFtRoomSetpoint in next record is at least 0.5 degree higher than thermostat_otFtRoomTemperature in this record 
    # 3). thermostat_otFtRoomSetpoint is not changing further afterwards for One hour
    # Then store the True/False flag in 'SetPointChangedFlag' column
    df_OneCiC.loc[:, 'SetPointChangedFlag'] = (
        (df_OneCiC['thermostat_otFtRoomSetpoint'].shift(-1) >= df_OneCiC['thermostat_otFtRoomSetpoint'] + 1) &
        (df_OneCiC['thermostat_otFtRoomSetpoint'].shift(-1) >= df_OneCiC['thermostat_otFtRoomTemperature'] + 0.5) &
        (df_OneCiC['thermostat_otFtRoomSetpoint'].rolling(239).apply(lambda x: len(set(x)) == 1, raw=True).shift(-240).fillna(0).astype(bool))
    ).fillna(False)


    ## loop for 15-second CiC record with SetPointChangedFlag==True
    for idx in df_OneCiC.index[df_OneCiC['SetPointChangedFlag']]:

        # find this values belongs to which merged time window.
        LeftEdgeIndex = -999
        for i in range(len(MergeWindow_BinEdge) - 1):
            if MergeWindow_BinEdge[i] <= idx < MergeWindow_BinEdge[i + 1]:
                LeftEdgeIndex = i
                break

        # define rising time and setpoint temperature change
        # logic to determine rising time: wait until the thermostat_otFtRoomTemperature reaches thermostat_otFtRoomSetpoint - 0.5 degree. 
        index_ReachedSteadyState = idx
        while (
            abs(df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState] - df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[idx + 1]) > 0.5
            and index_ReachedSteadyState < len(df_OneCiC) - 1  
        ):                    
            index_ReachedSteadyState = index_ReachedSteadyState + 1

        if index_ReachedSteadyState < len(df_OneCiC) - 1 and LeftEdgeIndex >= 0:
            risingtime                = df_OneCiC['time_ts'].iloc[index_ReachedSteadyState] - df_OneCiC['time_ts'].iloc[idx]
            thermostat_asking_heat_ratio = df_OneCiC['thermostat_otFtChEnabled'].iloc[idx:index_ReachedSteadyState].sum() / len(df_OneCiC.iloc[idx:index_ReachedSteadyState])
            setpointTemperatureChange = df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[idx+1] - df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[idx] 
            #print("Rising time until reach the steady state:" + str(risingtime) )
            #print("thermostat_asking_heat_ratio:" + str(thermostat_asking_heat_ratio))
            #print("SetPoint Temperature changes:" + str(setpointTemperatureChange))
            # Convert Rising Time to mins.
            risingtime = risingtime.total_seconds() / 60 * thermostat_asking_heat_ratio 
        else:
            continue

        # if there are multiple heating cycles in one merged time window, calculate mean of all rising time and setpoint temperature change in each merged time bin
        NumberofRisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] = NumberofRisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] + 1
        NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] = NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] + 1
        if math.isnan(RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex]):
            RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] = risingtime
            SetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] = setpointTemperatureChange
        else:
            RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] = (RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] + risingtime) / NumberofRisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] 
            SetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] = (SetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] + setpointTemperatureChange) / NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex]

        # Tracking Error
        # Determine the Length of SteadyState for Tracking Error (by checking the set point temperature). Edge case: Last SteadyState will decided by the length of df_OneCiC. 
        StableSetPointTemperature = df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[index_ReachedSteadyState]
        index_EndOfStableSetPointTemperature = next( (i for i, SearchingSetPointTemperature in enumerate(df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[index_ReachedSteadyState:], start=index_ReachedSteadyState) if SearchingSetPointTemperature != StableSetPointTemperature), len(df_OneCiC)-1)
        # print("StableSetPointTemperature last: " + str((df_OneCiC['time_ts'].iloc[index_EndOfStableSetPointTemperature] - df_OneCiC['time_ts'].iloc[index_ReachedSteadyState])) )
            
        # Check if the pump operation is in SteadyState
        # SteadyState: 1). RoomTemperature is fluctuating around the mean within 1 degrees; 2). This SteadyState last at least 1 hour 
        SteadyState_LastPeriod = index_EndOfStableSetPointTemperature - index_ReachedSteadyState
        # print("SteadyState_LastPeriod:" + str(SteadyState_LastPeriod))
        flag = ( 
            (df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod].max() - df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod].mean() < 1) and 
            (df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod].mean() - df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod].min() < 1) and
            len(set(df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod])) == 1 and 
            #SteadyState_LastPeriod > 240
            df_OneCiC['time_ts'].iloc[index_EndOfStableSetPointTemperature] - df_OneCiC['time_ts'].iloc[index_ReachedSteadyState] > pd.Timedelta(minutes=30)
        )

        # calculate each tracking error after heating and reaching steady state 
        if (flag):
            #averagedRoomTemperature  = df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod].mean() 
            #fixedSetPointTemperature = df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[index_ReachedSteadyState]
            #trackingError = fixedSetPointTemperature - averagedRoomTemperature 
  
            data_for_tracking_error = df_OneCiC.iloc[index_ReachedSteadyState:index_ReachedSteadyState+SteadyState_LastPeriod]
            data_for_tracking_error = data_for_tracking_error[data_for_tracking_error['thermostat_otFtChEnabled'] == True]
            errors = data_for_tracking_error['thermostat_otFtRoomTemperature'] - data_for_tracking_error['thermostat_otFtRoomSetpoint']
            rms = np.sqrt(np.mean(errors**2))
            trackingError = rms

            #Plot_function.Plot_KPITrackingError.plot_OneCiCOneDayTemperature(df_OneCiC, idx, index_ReachedSteadyState, index_EndOfStableSetPointTemperature, cic)
        else:
            continue


        #### if there are multiple heating cycles in one merged time window, calculate mean of all trackingErrors in each merged time bin 
        NumberofTrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex] = NumberofTrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex] + 1
        if math.isnan(TrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex]):
            TrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex] = trackingError
        else:
            TrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex] = (TrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex] + trackingError) / NumberofTrackingError_ThisCiC_RespectiveBins[LeftEdgeIndex] 


    return TrackingError_ThisCiC_RespectiveBins, RisingTime_ThisCiC_RespectiveBins, SetPointTemperatureChange_ThisCiC_RespectiveBins 


def calculate_rise_time(BinCenterForMergeWindow, MergeWindow_BinEdge, df_OneCiC, cic):

    # further preparation
    RisingTime_ThisCiC_RespectiveBins                       = [np.nan] * len(BinCenterForMergeWindow)
    NumberofRisingTime_ThisCiC_RespectiveBins               = [0]      * len(BinCenterForMergeWindow)
    SetPointTemperatureChange_ThisCiC_RespectiveBins         = [np.nan] * len(BinCenterForMergeWindow)
    NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins = [0]      * len(BinCenterForMergeWindow)

    ## quality check flag to indicate if the setpoint temperature is changed, therefore the rising time can be calculated.
    # check every 15-second CiC record if: 
    # 1). thermostat_otFtRoomSetpoint is changed in next record, and thermostat_otFtRoomSetpoint in next record is at least 1 degree higher than current thermostat_otFtRoomSetpoint. 
    # 2). thermostat_otFtRoomSetpoint in next record is at least 0.5 degree higher than thermostat_otFtRoomTemperature in this record 
    # 3). thermostat_otFtRoomSetpoint is not changing further afterwards for One hour
    # Then store the True/False flag in 'SetPointChangedFlag' column
    df_OneCiC.loc[:, 'SetPointChangedFlag'] = (
        (df_OneCiC['thermostat_otFtRoomSetpoint'].shift(-1) >= df_OneCiC['thermostat_otFtRoomSetpoint'] + 1) &
        (df_OneCiC['thermostat_otFtRoomSetpoint'].shift(-1) >= df_OneCiC['thermostat_otFtRoomTemperature'] + 0.5) &
        (df_OneCiC['thermostat_otFtRoomSetpoint'].rolling(239).apply(lambda x: len(set(x)) == 1, raw=True).shift(-240).fillna(0).astype(bool))
    ).fillna(False)

    # loop for 15-second CiC record with SetPointChangedFlag==True (从每一个SetPointChangedFlag等于True开始)
    for idx in df_OneCiC.index[df_OneCiC['SetPointChangedFlag']]:

        # find this values belongs to which merged time window.
        LeftEdgeIndex = -999
        for i in range(len(MergeWindow_BinEdge) - 1):
            if MergeWindow_BinEdge[i] <= idx < MergeWindow_BinEdge[i + 1]:
                LeftEdgeIndex = i
                break

        # define rising time and setpoint temperature change
        # logic to determine rising time: wait until the thermostat_otFtRoomTemperature reaches thermostat_otFtRoomSetpoint - 0.5 degree. 
        index_ReachedSteadyState = idx
        while (
            abs(df_OneCiC['thermostat_otFtRoomTemperature'].iloc[index_ReachedSteadyState] - df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[idx + 1]) > 0.5
            and index_ReachedSteadyState < len(df_OneCiC) - 1  
        ):                    
            index_ReachedSteadyState = index_ReachedSteadyState + 1

        if index_ReachedSteadyState < len(df_OneCiC) - 1 and LeftEdgeIndex >= 0:
            risingtime                = df_OneCiC['time_ts'].iloc[index_ReachedSteadyState] - df_OneCiC['time_ts'].iloc[idx]
            thermostat_asking_heat_ratio = df_OneCiC['thermostat_otFtChEnabled'].iloc[idx:index_ReachedSteadyState].sum() / len(df_OneCiC.iloc[idx:index_ReachedSteadyState])
            setpointTemperatureChange = df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[idx+1] - df_OneCiC['thermostat_otFtRoomSetpoint'].iloc[idx] 
            #print("Rising time until reach the steady state:" + str(risingtime) )
            #print("thermostat_asking_heat_ratio:" + str(thermostat_asking_heat_ratio))
            #print("SetPoint Temperature changes:" + str(setpointTemperatureChange))
            # Convert Rising Time to mins.
            risingtime = risingtime.total_seconds() / 60 * thermostat_asking_heat_ratio 
        else:
            continue

        # if there are multiple heating cycles in one merged time window, calculate mean of all rising time and setpoint temperature change in each merged time bin
        NumberofRisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] = NumberofRisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] + 1
        NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] = NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] + 1
        if math.isnan(RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex]):
            RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] = risingtime
            SetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] = setpointTemperatureChange
        else:
            RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] = (RisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] + risingtime) / NumberofRisingTime_ThisCiC_RespectiveBins[LeftEdgeIndex] 
            SetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] = (SetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex] + setpointTemperatureChange) / NumberofSetPointTemperatureChange_ThisCiC_RespectiveBins[LeftEdgeIndex]

    return RisingTime_ThisCiC_RespectiveBins, SetPointTemperatureChange_ThisCiC_RespectiveBins


