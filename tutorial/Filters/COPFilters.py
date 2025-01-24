import KPIUtility

def FilterOnOutsideTemperatureBelow2Degree(df_load):
    df_load.drop(df_load[df_load['hp1_temperatureOutside'] <= 2.0].index, inplace=True)

    df_load.drop(df_load[(~df_load['hp2_temperatureOutside'].isna()) & 
                         (df_load['hp2_temperatureOutside'] <= 2.0)].index, inplace=True)

def FilterOnHybrid(df_load):
    df_load.drop(df_load[df_load['system_ccNumberOfHeatPumps'] > 1.0].index, inplace=True)

def FillZeroForHP2(df_load):
    hp2_CiCproproty = ['hp2_electricalEnergyCounter', 'hp2_thermalEnergyCounter']
    for CiCproperty in hp2_CiCproproty:
        df_load[CiCproperty] = df_load[CiCproperty].fillna(0)
    df_load = df_load.dropna()

def FilterOnCOPTrainDataAvialable(df_load):
    df_load.dropna(how='any', subset=["hp1_temperatureOutside", "hp1_outletTemperatureFiltered", "hp1_compressorFrequency", "hp1_inletTemperatureFiltered"], inplace=True)

def FilterOnCOPTestDataRange(df_load):
    #outdoor_temperature_range           = [-15, 12]
    hp1_temperatureOutside_range        = [-15, 12]
    hp1_outletTemperatureFiltered_range = [35 , 55]
    hp1_compressorFrequency_range       = [43 , 90]
    hp1_inletTemperatureFiltered_range  = [30 , 54.5]

    KPIUtility.GetTemperatureOutside(df_load)

    in_range = (
        #(KPIUtility.GetTemperatureOutside(df_load).between(outdoor_temperature_range[0], outdoor_temperature_range[1])) &
        (df_load['hp1_temperatureOutside'].between(hp1_temperatureOutside_range[0], hp1_temperatureOutside_range[1])) &
        (df_load['hp1_outletTemperatureFiltered'].between(hp1_outletTemperatureFiltered_range[0], hp1_outletTemperatureFiltered_range[1])) &
        (df_load['hp1_compressorFrequency'].between(hp1_compressorFrequency_range[0], hp1_compressorFrequency_range[1])) &
        (df_load['hp1_inletTemperatureFiltered'].between(hp1_inletTemperatureFiltered_range[0], hp1_inletTemperatureFiltered_range[1]))
    )
    df_load[in_range]

    indices_to_keep = df_load[in_range].index
    df_load.drop(df_load.index.difference(indices_to_keep), inplace=True)
 
def FilterOnZeroCounter(df_load):
    df_load.drop(df_load[(df_load['hp1_thermalEnergyCounter'] == 0) | (df_load['hp1_electricalEnergyCounter'] == 0)].index, inplace=True)  

def FilterOnPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([2, 3])].index, inplace=True)

def FilterOnOnlyPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([2])].index, inplace=True)


