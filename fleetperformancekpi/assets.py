import pandas as pd
from dagster import op, Out, job, AssetIn, asset, resource, EnvVar, get_dagster_logger, AssetExecutionContext, Config
import argparse
import os
import sys
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import KPIUtility
from . import Filters
from . import calculateKPI
from . import loadRawCSVData
import Utility.Plot_Functions as Plot_function

logger = get_dagster_logger()

class KpiConfig(Config):
    kpi: str  

@asset
def printVariable() -> None:
    snowflake_user = os.environ.get("SNOWFLAKE_USER")
    snowflake_password = os.environ.get("SNOWFLAKE_PASSWORD")
    print(f"SNOWFLAKE_USER: {snowflake_user}")
    print(f"SNOWFLAKE_PASSWORD: {snowflake_password}")


@asset(required_resource_keys={"snowflake_client"})
def raw_cic_dataset(context) -> pd.DataFrame:

    query = """
        SELECT
            HP1_WATCHDOGCODE,  
            HP2_WATCHDOGCODE,  
            HP1_THERMALENERGYCOUNTER,  
            HP2_THERMALENERGYCOUNTER,  
            HP1_ELECTRICALENERGYCOUNTER,  
            HP2_ELECTRICALENERGYCOUNTER,  
            HP1_INLETTEMPERATUREFILTERED,  
            HP1_OUTLETTEMPERATUREFILTERED,  
            HP1_COMPRESSORFREQUENCY,  
            HP2_INLETTEMPERATUREFILTERED,  
            HP2_OUTLETTEMPERATUREFILTERED,  
            HP2_COMPRESSORFREQUENCY,  

            HP1_TEMPERATUREOUTSIDE,  
            HP2_TEMPERATUREOUTSIDE,  
            HP1_LIMITEDBYCOP,  
            HP2_LIMITEDBYCOP,  
            HP1_RATEDPOWER,  
            HP2_RATEDPOWER,  

            QC_SUPERVISORYCONTROLMODE,  
            QC_SUPPLYTEMPERATUREFILTERED,  
            QC_CVENERGYCOUNTER,  
            QC_ESTIMATEDPOWERDEMAND,  
            QC_SYSTEMWATCHDOGCODE,

            THERMOSTAT_OTFTROOMTEMPERATURE,  
            THERMOSTAT_OTFTROOMSETPOINT,  
            THERMOSTAT_OTFTCHENABLED,  

            CLIENTID,
            CLIENT_TIME
        FROM
            FIREHOSE_CIC.CIC.STATS_DYNAMIC
        WHERE
            CLIENTID = 'cic-319d1ceb-4ee9-51dd-a13a-18eb047dd625' -- Mark's CiC
            -- CLIENTID LIKE 'cic-5795534%'

            AND CLIENT_TIME BETWEEN DATEADD(DAY, -10, CURRENT_TIMESTAMP()) AND CURRENT_TIMESTAMP() -- last 10 days, for demo production
            -- AND CLIENT_TIME BETWEEN DATEADD(HOUR, -1, DATE_TRUNC('HOUR', CURRENT_TIMESTAMP())) AND DATE_TRUNC('HOUR', CURRENT_TIMESTAMP())   -- for example: trigger on 10:08, CLIENT_TIME will be 09:00 to 10:00
            -- AND CLIENT_TIME BETWEEN DATEADD(DAY, -1, CURRENT_TIMESTAMP()) AND CURRENT_TIMESTAMP() -- last 24 hours

        ORDER BY
            CLIENT_TIME
    """

    # Connect to Snowflake and Run query
    context.log.info("Querying Snowflake...")
    conn = context.resources.snowflake_client
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # return to DataFrame
    df = pd.DataFrame(result, columns=columns)
    df.columns = df.columns.astype(str)  # turn columns to str

    return df


@asset
# def apply_filters(context, raw_cic_dataset, kpi: str) -> pd.DataFrame:
def apply_filters(context, raw_cic_dataset) -> pd.DataFrame:

    # kpi = config.kpi  
    # kpi = "HeatPercentage"
    # kpi = "COPPerformance"
    # kpi = "HeatingCycle"
    # kpi = "WatchdogChange"
    # kpi = "RiseTime"
    # kpi = "TwoIndicators"
    # kpi = "GasUseageCase"
    kpi = "OutsideTemperature"

    logger.info(f"Applying filters for KPI: {kpi}")
    generalfilterList = ['FilterOnTestRigs', 'FilterOnInactiveCiC', 'FilterOnDropZero', 'FilterOnDropABit', 'FilterOnIncreaseCounter', 'FilterOnAllNull']
    KPIUtility.applyfilters(raw_cic_dataset, generalfilterList, Filters.GeneralFilters, filters_sector_name='General Filters')

    if kpi == "COPPerformance":
        COPfilterList = ['FillZeroForHP2', 'FilterOnCOPTrainDataAvialable', 'FilterOnZeroCounter', 'FilterOnOnlyPumpRunning']
        KPIUtility.applyfilters(raw_cic_dataset, COPfilterList, Filters.COPFilters, filters_sector_name='COP Filters')
    elif kpi == "HeatPercentage":
        HeatPercentagefilterList = ['FilterOnCVEnergyAvialable', 'FillZeroThermalForHP2', 'FilterOnPumpOrBoilerRunning'] 
        KPIUtility.applyfilters(raw_cic_dataset, HeatPercentagefilterList, Filters.HeatPercentageFilters, filters_sector_name='HeatPercentage Filters')
    elif kpi == "HeatingCycle":
        HeatingCycleFilterList = ['FilterOnSupervisoryControlModeNotNull']
        KPIUtility.applyfilters(raw_cic_dataset, HeatingCycleFilterList, Filters.HeatingCycleFilters, filters_sector_name='HeatingCycle Filters')
    elif kpi == "WatchdogChange":
        WatchdogChangeFilterList = ['FilterOnHP1Watchdog', 'FilterOnSystemWatchdog']
        KPIUtility.applyfilters(raw_cic_dataset, WatchdogChangeFilterList, Filters.WatchdogChangeFilters, filters_sector_name='WatchdogChange Filters')
    elif kpi == "RiseTime":
        TrackingErrorFilterList = ['FilterOnSetPointTempAvialable', 'FilterOnRoomTempAvialable']
        KPIUtility.applyfilters(raw_cic_dataset, TrackingErrorFilterList, Filters.TrackingError, filters_sector_name='TrackingError Filters')
    elif kpi == "TwoIndicators":
        TwoIndicatorsFilterList = ['FilterOnRoomTempAvialable', 'FilterOnWaterTempAvialable', 'FilterOnOutsideTempAvialable', 'FilterOnHPHeatAvialable', 'FilterOnBoilerHeatAvialable', 'FilterOnSetPointTempAvialable']
        KPIUtility.applyfilters(raw_cic_dataset, TwoIndicatorsFilterList, Filters.TwoIndicatorsFilters, filters_sector_name='Two Indicators Filters')
    elif kpi == "GasUseageCase":
        GasUseageCaseFilterList = ['FilterOnHP1WatchdogNull', 'FilterOnBoilerRunning', 'FilterOnHP1RatedPowerNull', 'FilterOnEstimatedPowerNull']
        KPIUtility.applyfilters(raw_cic_dataset, GasUseageCaseFilterList, Filters.GasUseageCaseFilters, filters_sector_name='GasUseageCase Filters')
    elif kpi == "OutsideTemperature":
        OutsideTemperatureList = ['FilterOnHP1OutsideTemperature']
        KPIUtility.applyfilters(raw_cic_dataset, OutsideTemperatureList, Filters.OutsideTemperatureFilters, filters_sector_name='Outside Temperature Filters')

    context.log.info(f"Applied filters for KPI: {kpi}")
    return raw_cic_dataset

@asset
def calculate_kpi(context, apply_filters) -> pd.DataFrame:
    '''
    asset to calculate KPIs
    '''

    #kpi = config.kpi  # 从配置中获取 kpi
    # kpi = "HeatPercentage"
    # kpi = "COPPerformance"
    # kpi = "HeatingCycle"
    # kpi = "WatchdogChange"
    # kpi = "RiseTime"
    # kpi = "TwoIndicators"
    # kpi = "GasUseageCase"
    kpi = 'OutsideTemperature'

    logger.info(f"Calculating KPI: {kpi}")

    ## Preparation
    apply_filters['client_time'] = pd.to_datetime(apply_filters['client_time'], utc=True)
    BinCenterForMergeWindow_allCiC_RespectiveBins= []

    if kpi == "COPPerformance":
        MergeWindow = 1440 # 60: hourly, 1440: daily, 5 days:7200, 43200:monthly (30 days) 
        MeasuredCOP_allCiC_RespectiveBins = [] # length= CiC numbers
        ExpectedCOP_allCiC_RespectiveBins = [] 
        COPRatio_allCiC_RespectiveBins    = []
        AveragedOutsideTemp_allCiC_RespectiveBins = []

    elif kpi == "HeatPercentage":
        MergeWindow = 1440
        GasPercentage_allCiC_RespectiveBins = []
        HeatPumpPercentage_allCiC_RespectiveBins = []

    elif kpi == "HeatingCycle":
        MergeWindow = 1440 
        HeatingCycleChangeCounts_boiler_allCiC_RespectiveBins = []
        HeatingCyclePercentageCounts_boiler_allCiC_RespectiveBins = []
        HeatingCycleChangeCounts_HP_allCiC_RespectiveBins = []
        HeatingCyclePercentageCounts_HP_allCiC_RespectiveBins = []
        HeatingCycleChangeCounts_HP_and_boiler_allCiC_RespectiveBins = []
        HeatingCyclePercentageCounts_HP_and_boiler_allCiC_RespectiveBins = []
        KPIUtility.FillSystemFunctionStatus(apply_filters)

    elif kpi == "WatchdogChange":
        MergeWindow = 1440
        WatchdogChangeCounts_allCiC_RespectiveBins     = []
        WatchdogPercentageCounts_allCiC_RespectiveBins = []
        SevereWatchdogChangeCounts_allCiC_RespectiveBins = []
        SevereWatchdogPercentageCounts_allCiC_RespectiveBins = []
        KPIUtility.FillWatchdogAbnormalStatus(apply_filters)

    elif kpi == "RiseTime":
        MergeWindow = 1440   
        RisingTime_allCiC_RespectiveBins                = []
        SetPointTemperatureChange_allCiC_RespectiveBins = []

    elif kpi == "TwoIndicators":
        MergeWindow = 1440
        TrackingError_allCiC_RespectiveBins                    = []
        InsulationIndicator_allCiC_RespectiveBins              = []
        HeatDissipationIndicator_allCiC_RespectiveBins         = []
        SteadyStateRatio_Insulation_allCiC_RespectiveBins      = []
        SteadyStateRatio_HeatDissipation_allCiC_RespectiveBins = []
        SteadyStateRatio_TrackingError_allCiC_RespectiveBins   = []

    elif kpi == "GasUseageCase":
        MergeWindow = 1440
        boiler_usage_allCiC_RespectiveBins = {}
        boiler_usage_heat_allCiC_RespectiveBins = {}

    elif kpi == "OutsideTemperature":
        MergeWindow = 1440
        AveragedOutsideTemp_allCiC_RespectiveBins = []

    freqvalue = str(MergeWindow) + 'min'
    fleet_start_time = apply_filters['client_time'].min().normalize()
    fleet_end_time_max = apply_filters['client_time'].max()
    fleet_end_time = pd.date_range(start = fleet_start_time, end = fleet_end_time_max, freq=freqvalue).max()

    BinCenterForMergeWindow_WholeTimeRange = pd.Series(pd.date_range(start = fleet_start_time + pd.Timedelta(minutes=MergeWindow/2), end=fleet_end_time + pd.Timedelta(minutes=MergeWindow/2), freq=freqvalue))
    final_ciclist = []
    KPIlist = {}

    ## Initial result PandasDataFrame 
    column_order = ["DATE_TIME", "CICID", "MeasuredCOP", "ExpectedCOP", "COPPerformance", "OutsideTemperature", "Heating_cycles_HP", "Time_by_HP", "Heating_cycles_boiler", "Time_by_boiler", "Heating_cycles_HP_and_boiler", "Time_by_HP_and_boiler", "Watchdog_code_changes", "Time_with_watchdog", "Severe_watchdog_code_changes", "Time_with_severe_watchdog", "Heat_by_boiler", "Heat_by_HP", "Tracking_error", "Rise_time", "SetPointTemperatureChange", "House_Insulation_Indicator", "Heat_distribution_system_capacity", "SSR_House_insulation", "SSR_Heat_distribution", "SSR_Tracking_error", "BoilerUsage_HighDemand", "BoilerUsage_Limited_by_COP", "BoilerUsage_WaterTempTooHigh", "BoilerUsage_PreHeating", "BoilerUsage_Anomaly", "BoilerUsage_NoReason", "BoilerUsage_TotalBoilerTime", "BoilerUsage_Heat_HighDemand", "BoilerUsage_Heat_Limited_by_COP", "BoilerUsage_Heat_WaterTempTooHigh", "BoilerUsage_Heat_PreHeating", "BoilerUsage_Heat_Anomaly", "BoilerUsage_Heat_NoReason", "BoilerUsage_TotalBoilerHEAT"]
    KPIResult = pd.DataFrame(columns=column_order)
    KPIResult["DATE_TIME"] = pd.to_datetime(KPIResult["DATE_TIME"])

    ## Loop for CiC
    raw_ciclist = apply_filters['clientid'].unique()
    total_cic = len(raw_ciclist)

    for cic_idx, cic in enumerate(raw_ciclist):
        df_OneCiC = apply_filters[apply_filters['clientid'] == cic].copy()
        df_OneCiC = df_OneCiC.reset_index(drop=True)

        MergeWindow_BinEdge, BinCenterForMergeWindow, df_OneCiC = KPIUtility.MergeWindow(df_OneCiC, cic, MergeWindow, fleet_start_time, fleet_end_time, freqvalue)

        if len(df_OneCiC) <= 1 or len(BinCenterForMergeWindow)<1:
            continue
        BinCenterForMergeWindow_allCiC_RespectiveBins.append(BinCenterForMergeWindow)

        if kpi == "COPPerformance":
            MeasuredCOP = KPIUtility.CalculateCOP_withWindow(df_OneCiC, MergeWindow_BinEdge)
            model       = KPIUtility.ModelTraining('Polynomial') 
            ExpectedCOP, AveragedOutsideTemp = KPIUtility.ModelPredicting(df_OneCiC, model, MergeWindow_BinEdge, cic) 
            COPRatio    = np.where(np.isnan(ExpectedCOP) | np.isnan(MeasuredCOP), np.nan, np.array(MeasuredCOP) / np.array(ExpectedCOP))   
            KPIUtility.CheckBinXY(BinCenterForMergeWindow, COPRatio)
            MeasuredCOP_allCiC_RespectiveBins.append(MeasuredCOP)
            ExpectedCOP_allCiC_RespectiveBins.append(ExpectedCOP)
            COPRatio_allCiC_RespectiveBins.append(COPRatio) 
            AveragedOutsideTemp_allCiC_RespectiveBins.append(AveragedOutsideTemp) 
            KPIlist['MeasuredCOP'] = MeasuredCOP_allCiC_RespectiveBins
            KPIlist['ExpectedCOP'] = ExpectedCOP_allCiC_RespectiveBins
            KPIlist['COPPerformance_COPRatio'] = COPRatio_allCiC_RespectiveBins
            KPIlist['COPPerformance_OutsideT'] = AveragedOutsideTemp_allCiC_RespectiveBins

        elif kpi == "HeatPercentage":
            GasPercentage_tem, HeatPumpPercentage_tem = KPIUtility.CalculateGasPercentage_withWindow(df_OneCiC, MergeWindow_BinEdge)
            GasPercentage_allCiC_RespectiveBins.append(GasPercentage_tem)
            HeatPumpPercentage_allCiC_RespectiveBins.append(HeatPumpPercentage_tem)
            KPIlist['HeatPercentage_gas']      = GasPercentage_allCiC_RespectiveBins
            KPIlist['HeatPercentage_heatpump'] = HeatPumpPercentage_allCiC_RespectiveBins

        elif kpi == "HeatingCycle":
            window_length     = pd.Timedelta(days=1)     # Sliding Window Lendth
            step_size         = pd.Timedelta(days=1)     # Sliding Speed
            percentage_counts_boiler, change_counts_boiler = KPIUtility.CalculateSlidingWindow_ChangeANDPercentage(BinCenterForMergeWindow, freqvalue, df_OneCiC, window_length, step_size, columnname='functionStatus_boiler')
            percentage_counts_HP, change_counts_HP = KPIUtility.CalculateSlidingWindow_ChangeANDPercentage(BinCenterForMergeWindow, freqvalue, df_OneCiC, window_length, step_size, columnname='functionStatus_HP')
            percentage_counts_HP_and_boiler, change_counts_HP_and_boiler = KPIUtility.CalculateSlidingWindow_ChangeANDPercentage(BinCenterForMergeWindow, freqvalue, df_OneCiC, window_length, step_size, columnname='functionStatus_HP_and_boiler')
            HeatingCycleChangeCounts_boiler_allCiC_RespectiveBins.append(change_counts_boiler)
            HeatingCyclePercentageCounts_boiler_allCiC_RespectiveBins.append(percentage_counts_boiler)
            HeatingCycleChangeCounts_HP_allCiC_RespectiveBins.append(change_counts_HP)
            HeatingCyclePercentageCounts_HP_allCiC_RespectiveBins.append(percentage_counts_HP)
            HeatingCycleChangeCounts_HP_and_boiler_allCiC_RespectiveBins.append(change_counts_HP_and_boiler)
            HeatingCyclePercentageCounts_HP_and_boiler_allCiC_RespectiveBins.append(percentage_counts_HP_and_boiler)
            KPIlist['HeatingCycle_ChangeCounts_boiler']     = HeatingCycleChangeCounts_boiler_allCiC_RespectiveBins
            KPIlist['HeatingCycle_PercentageCounts_boiler'] = HeatingCyclePercentageCounts_boiler_allCiC_RespectiveBins
            KPIlist['HeatingCycle_ChangeCounts_HP']         = HeatingCycleChangeCounts_HP_allCiC_RespectiveBins
            KPIlist['HeatingCycle_PercentageCounts_HP']     = HeatingCyclePercentageCounts_HP_allCiC_RespectiveBins
            KPIlist['HeatingCycle_ChangeCounts_HP_and_boiler']     = HeatingCycleChangeCounts_HP_and_boiler_allCiC_RespectiveBins
            KPIlist['HeatingCycle_PercentageCounts_HP_and_boiler'] = HeatingCyclePercentageCounts_HP_and_boiler_allCiC_RespectiveBins

        elif kpi == "WatchdogChange":
            window_length     = pd.Timedelta(days=1)     # Sliding Window Lendth
            step_size         = pd.Timedelta(days=1)     # Sliding Speed
            percentage_counts, change_counts = KPIUtility.CalculateSlidingWindow_ChangeANDPercentage(BinCenterForMergeWindow, freqvalue, df_OneCiC, window_length, step_size, columnname='WatchdogAbnormalStatus')
            WatchdogChangeCounts_allCiC_RespectiveBins.append(change_counts)
            WatchdogPercentageCounts_allCiC_RespectiveBins.append(percentage_counts) 
            percentage_counts_severe, change_counts_severe = KPIUtility.CalculateSlidingWindow_ChangeANDPercentage(BinCenterForMergeWindow, freqvalue, df_OneCiC, window_length, step_size, columnname='SevereWatchdogAbnormalStatus')
            SevereWatchdogChangeCounts_allCiC_RespectiveBins.append(change_counts_severe)
            SevereWatchdogPercentageCounts_allCiC_RespectiveBins.append(percentage_counts_severe)
            KPIlist['WatchdogChange_ChangeCounts']     = WatchdogChangeCounts_allCiC_RespectiveBins
            KPIlist['WatchdogChange_PercentageCounts'] = WatchdogPercentageCounts_allCiC_RespectiveBins
            KPIlist['SevereWatchdogChange_ChangeCounts'] = SevereWatchdogChangeCounts_allCiC_RespectiveBins
            KPIlist['SevereWatchdogChange_PercentageCounts'] = SevereWatchdogPercentageCounts_allCiC_RespectiveBins

        elif kpi == "RiseTime":
            RisingTime_ThisCiC_RespectiveBins, SetPointTemperatureChange_ThisCiC_RespectiveBins = calculateKPI.calculate_rise_time(BinCenterForMergeWindow, MergeWindow_BinEdge, df_OneCiC, cic)
            RisingTime_allCiC_RespectiveBins   .append(RisingTime_ThisCiC_RespectiveBins)
            SetPointTemperatureChange_allCiC_RespectiveBins.append(SetPointTemperatureChange_ThisCiC_RespectiveBins)
            KPIlist['RisingTime']    = RisingTime_allCiC_RespectiveBins
            KPIlist['SetPointTemperatureChange'] = SetPointTemperatureChange_allCiC_RespectiveBins

        elif kpi == "TwoIndicators":
            # further preparation
            InsulationIndicator_ThisCiC_RespectiveBins                 = [np.nan] * len(BinCenterForMergeWindow)
            NumberofInsulationIndicator_ThisCiC_RespectiveBins         = [0]      * len(BinCenterForMergeWindow)
            SteadyStateRatio_Insulation_ThisCiC_RespectiveBins         = [0]      * len(BinCenterForMergeWindow)
            DissipationIndicator_ThisCiC_RespectiveBins                = [np.nan] * len(BinCenterForMergeWindow)
            NumberofDissipationIndicator_ThisCiC_RespectiveBins        = [0]      * len(BinCenterForMergeWindow)
            SteadyStateRatio_HeatDissipation_ThisCiC_RespectiveBins    = [0]      * len(BinCenterForMergeWindow) 
            TrackingError_ThisCiC_RespectiveBins                       = [np.nan] * len(BinCenterForMergeWindow)
            NumberofTrackingError_ThisCiC_RespectiveBins               = [0]      * len(BinCenterForMergeWindow)
            SteadyStateRatio_TrackingError_ThisCiC_RespectiveBins      = [0]      * len(BinCenterForMergeWindow)
            # loop for every merge window, eg: 1 day
            for index_MergeWindow in range(len(MergeWindow_BinEdge)-1):
                # Get data in this Merge Window
                startindex = int(MergeWindow_BinEdge[index_MergeWindow])
                endindex = int(MergeWindow_BinEdge[index_MergeWindow+1])
                if startindex == 0:
                    df_OneCiC_InOneMergeWindow = df_OneCiC.loc[ startindex : endindex ]
                else:
                    df_OneCiC_InOneMergeWindow = df_OneCiC.loc[ startindex+1 : endindex ]
                # preparation 
                df_OneCiC_InOneMergeWindow = df_OneCiC_InOneMergeWindow.reset_index(drop=True)
                n = len(df_OneCiC_InOneMergeWindow)
                window_size = 120 # minimum steady state window, size of 120 is 30 mins 
                InsulationIndicator_ThisCiC_RespectiveBins, SteadyStateRatio_Insulation_ThisCiC_RespectiveBins = calculateKPI.DetermingSteadyStates(df_OneCiC_InOneMergeWindow, window_size, BinCenterForMergeWindow, index_MergeWindow, InsulationIndicator_ThisCiC_RespectiveBins, NumberofInsulationIndicator_ThisCiC_RespectiveBins, SteadyStateRatio_Insulation_ThisCiC_RespectiveBins, 'InsulationIndicator') 
                DissipationIndicator_ThisCiC_RespectiveBins, SteadyStateRatio_HeatDissipation_ThisCiC_RespectiveBins = calculateKPI.DetermingSteadyStates(df_OneCiC_InOneMergeWindow, window_size, BinCenterForMergeWindow, index_MergeWindow, DissipationIndicator_ThisCiC_RespectiveBins, NumberofDissipationIndicator_ThisCiC_RespectiveBins, SteadyStateRatio_HeatDissipation_ThisCiC_RespectiveBins, 'Dissipation')
                TrackingError_ThisCiC_RespectiveBins, SteadyStateRatio_TrackingError_ThisCiC_RespectiveBins = calculateKPI.DetermingSteadyStates(df_OneCiC_InOneMergeWindow, window_size, BinCenterForMergeWindow, index_MergeWindow, TrackingError_ThisCiC_RespectiveBins, NumberofTrackingError_ThisCiC_RespectiveBins, SteadyStateRatio_TrackingError_ThisCiC_RespectiveBins, 'TrackingError')  
 
            # add this CiC result to the All CiC Result list
            InsulationIndicator_allCiC_RespectiveBins             .append(InsulationIndicator_ThisCiC_RespectiveBins)
            SteadyStateRatio_Insulation_allCiC_RespectiveBins     .append(SteadyStateRatio_Insulation_ThisCiC_RespectiveBins)
            HeatDissipationIndicator_allCiC_RespectiveBins        .append(DissipationIndicator_ThisCiC_RespectiveBins)
            SteadyStateRatio_HeatDissipation_allCiC_RespectiveBins.append(SteadyStateRatio_HeatDissipation_ThisCiC_RespectiveBins)
            TrackingError_allCiC_RespectiveBins                   .append(TrackingError_ThisCiC_RespectiveBins) 
            SteadyStateRatio_TrackingError_allCiC_RespectiveBins  .append(SteadyStateRatio_TrackingError_ThisCiC_RespectiveBins)
            KPIlist['TrackingError'] = TrackingError_allCiC_RespectiveBins
            KPIlist['InsulationIndicator']         = InsulationIndicator_allCiC_RespectiveBins 
            KPIlist['HeatDissipationIndicator']    = HeatDissipationIndicator_allCiC_RespectiveBins
            KPIlist['SSR_Insulation']              = SteadyStateRatio_Insulation_allCiC_RespectiveBins
            KPIlist['SSR_HeatDissipation']         = SteadyStateRatio_HeatDissipation_allCiC_RespectiveBins
            KPIlist['SSR_TrackingError']           = SteadyStateRatio_TrackingError_allCiC_RespectiveBins

        elif kpi == "GasUseageCase":
            boiler_usage_ThisCiC_RespectiveBins, boiler_usage_heat_ThisCiC_RespectiveBins = calculateKPI.calculate_gas_useage_case_time_and_heat(MergeWindow_BinEdge, df_OneCiC)
            boiler_usage_allCiC_RespectiveBins[cic] = boiler_usage_ThisCiC_RespectiveBins
            boiler_usage_heat_allCiC_RespectiveBins[cic] = boiler_usage_heat_ThisCiC_RespectiveBins
            KPIlist['boiler_usage_time'] = boiler_usage_allCiC_RespectiveBins
            KPIlist['boiler_usage_heat'] = boiler_usage_heat_allCiC_RespectiveBins

        elif kpi == "OutsideTemperature":
            AveragedOutsideTemp = KPIUtility.GetAveragedOutsideTemperature(df_OneCiC, MergeWindow_BinEdge, cic)
            AveragedOutsideTemp_allCiC_RespectiveBins.append(AveragedOutsideTemp)
            KPIlist['OutsideTemperature'] = AveragedOutsideTemp_allCiC_RespectiveBins

        final_ciclist.append(cic)


    ## Save the result
    if kpi == "COPPerformance":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, MeasuredCOP_iter, ExpectedCOP_iter, COPRatio_iter, OutsideTemperature_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['MeasuredCOP'], KPIlist['ExpectedCOP'], KPIlist['COPPerformance_COPRatio'], KPIlist['COPPerformance_OutsideT'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME"       : timebincenter,
                    "CICID"           : cic_iter,
                    "MeasuredCOP" : MeasuredCOP_iter[index],
                    "ExpectedCOP" : ExpectedCOP_iter[index],
                    "COPPerformance" : COPRatio_iter[index],
                }
                KPIResult.loc[condition, ["MeasuredCOP", "ExpectedCOP", "COPPerformance"]] = [new_row['MeasuredCOP'], new_row['ExpectedCOP'], new_row['COPPerformance']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)

    elif kpi == "HeatPercentage":
        new_rows = []  
        for idx, (BinCenterForMergeWindow_iter, cic_iter, GasPercentage_iter, HeatPumpPercentage_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['HeatPercentage_gas'], KPIlist['HeatPercentage_heatpump'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME": timebincenter,
                    "CICID": cic_iter,
                    "Heat_by_boiler": GasPercentage_iter[index],
                    "Heat_by_HP": HeatPumpPercentage_iter[index],
                }
                KPIResult.loc[condition, ['Heat_by_boiler', 'Heat_by_HP']] = [new_row['Heat_by_boiler'], new_row['Heat_by_HP']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows: 
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)      

    elif kpi == "HeatingCycle":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, HeatingCycleChangeCounts_boiler_iter, HeatingCyclePercentageCounts_boiler_iter, HeatingCycleChangeCounts_HP_iter, HeatingCyclePercentageCounts_HP_iter, HeatingCycleChangeCounts_HP_and_boiler_iter, HeatingCyclePercentageCounts_HP_and_boiler_iter) in enumerate(zip( BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['HeatingCycle_ChangeCounts_boiler'], KPIlist['HeatingCycle_PercentageCounts_boiler'], KPIlist['HeatingCycle_ChangeCounts_HP'], KPIlist['HeatingCycle_PercentageCounts_HP'], KPIlist['HeatingCycle_ChangeCounts_HP_and_boiler'], KPIlist['HeatingCycle_PercentageCounts_HP_and_boiler'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME": timebincenter,
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

    elif kpi == "WatchdogChange":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, WatchdogChangeCounts_iter, WatchdogPercentageCounts_iter, severeWatchdogChangeCounts_iter, severeWatchdogPercentageCounts_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['WatchdogChange_ChangeCounts'], KPIlist['WatchdogChange_PercentageCounts'], KPIlist['SevereWatchdogChange_ChangeCounts'], KPIlist['SevereWatchdogChange_PercentageCounts'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME": timebincenter,
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

    elif kpi == "RiseTime":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, RisingTime_iter, SetpointT_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['RisingTime'], KPIlist['SetPointTemperatureChange'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME": timebincenter,
                    "CICID": cic_iter,
                    "Rise_time"     : RisingTime_iter[index],
                    "SetPointTemperatureChange" : SetpointT_iter[index]
                } 
                KPIResult.loc[condition, ["Rise_time", "SetPointTemperatureChange"]] = [new_row['Rise_time'], new_row['SetPointTemperatureChange']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)

    elif kpi == "TwoIndicators":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, Tracking_error_iter, InsulationIndicator_iter, HeatDissipationIndicator_iter, SSR_Insulation_iter, SSR_HeatDissipation_iter, SSR_TrackingError_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['TrackingError'], KPIlist['InsulationIndicator'], KPIlist['HeatDissipationIndicator'], KPIlist['SSR_Insulation'], KPIlist['SSR_HeatDissipation'], KPIlist['SSR_TrackingError'] )):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME": timebincenter,
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

                
    elif kpi == "GasUseageCase":
        new_rows = []
        for (cic_time, values_time), (cic_heat, values_heat), BinCenterForMergeWindow_iter in zip(
            KPIlist['boiler_usage_time'].items(), 
            KPIlist['boiler_usage_heat'].items(), 
            BinCenterForMergeWindow_allCiC_RespectiveBins):
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter): 
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_time)     
                new_row = {
                    "DATE_TIME": timebincenter,
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

    elif kpi == "OutsideTemperature":
        new_rows = []
        for idx, (BinCenterForMergeWindow_iter, cic_iter, OutsideTemperature_iter) in enumerate(zip(BinCenterForMergeWindow_allCiC_RespectiveBins, final_ciclist, KPIlist['OutsideTemperature'])):  # loop for CiC list
            for index, timebincenter in enumerate(BinCenterForMergeWindow_iter):
                condition = (KPIResult['DATE_TIME'] == timebincenter) & (KPIResult['CICID'] == cic_iter)
                new_row = {
                    "DATE_TIME": timebincenter,
                    "CICID": cic_iter,
                    "OutsideTemperature"    : OutsideTemperature_iter[index],
                }
                KPIResult.loc[condition, ["OutsideTemperature"]] = [new_row['OutsideTemperature']]
                new_rows.extend([new_row] if not condition.any() else [])
        if new_rows:
            KPIResult = pd.concat([KPIResult, pd.DataFrame(new_rows)], ignore_index=True)


        
    return KPIResult


@asset
def KPI_results(calculate_kpi) -> pd.DataFrame:
    df = pd.DataFrame(calculate_kpi)
    return df

