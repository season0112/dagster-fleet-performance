import KPIUtility

def FilterOnOutsideTemperatureBelow2Degree(df_load):
    df_load.drop(df_load[df_load['hp1_temperatureoutside'] <= 2.0].index, inplace=True)

    df_load.drop(df_load[(~df_load['hp2_temperatureoutside'].isna()) & 
                         (df_load['hp2_temperatureoutside'] <= 2.0)].index, inplace=True)

def FilterOnHybrid(df_load):
    df_load.drop(df_load[df_load['system_ccNumberOfHeatPumps'] > 1.0].index, inplace=True)

def FillZeroForHP2(df_load):
    hp2_CiCproproty = ['hp2_electricalenergycounter', 'hp2_thermalenergycounter']
    for CiCproperty in hp2_CiCproproty:
        df_load[CiCproperty] = df_load[CiCproperty].fillna(0)
    df_load = df_load.dropna()

def FilterOnCOPTrainDataAvialable(df_load):
    df_load.dropna(how='any', subset=["hp1_temperatureoutside", "hp1_outlettemperaturefiltered", "hp1_compressorfrequency", "hp1_inlettemperaturefiltered"], inplace=True)

def FilterOnCOPTestDataRange(df_load):
    #outdoor_temperature_range           = [-15, 12]
    hp1_temperatureoutside_range        = [-15, 12]
    hp1_outlettemperaturefiltered_range = [35 , 55]
    hp1_compressorfrequency_range       = [43 , 90]
    hp1_inlettemperaturefiltered_range  = [30 , 54.5]

    KPIUtility.GetTemperatureOutside(df_load)

    in_range = (
        #(KPIUtility.GetTemperatureOutside(df_load).between(outdoor_temperature_range[0], outdoor_temperature_range[1])) &
        (df_load['hp1_temperatureoutside'].between(hp1_temperatureoutside_range[0], hp1_temperatureoutside_range[1])) &
        (df_load['hp1_outlettemperaturefiltered'].between(hp1_outlettemperaturefiltered_range[0], hp1_outlettemperaturefiltered_range[1])) &
        (df_load['hp1_compressorfrequency'].between(hp1_compressorfrequency_range[0], hp1_compressorfrequency_range[1])) &
        (df_load['hp1_inlettemperaturefiltered'].between(hp1_inlettemperaturefiltered_range[0], hp1_inlettemperaturefiltered_range[1]))
    )
    df_load[in_range]

    indices_to_keep = df_load[in_range].index
    df_load.drop(df_load.index.difference(indices_to_keep), inplace=True)
 
def FilterOnZeroCounter(df_load):
    df_load.drop(df_load[(df_load['hp1_thermalenergycounter'] == 0) | (df_load['hp1_electricalenergycounter'] == 0)].index, inplace=True)  

def FilterOnPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([2, 3])].index, inplace=True)

def FilterOnOnlyPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([2])].index, inplace=True)


