


def FilterOnBothPumpAndBoilerRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([3])].index, inplace=True)

def FilterOnBoilerRunningOnly(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([4])].index, inplace=True)

def FilterOnCVEnergyAvialable(df_load):
    #df_load.dropna(how='any', subset=["qc_cvEnergyCounter"], inplace=True)
    #df_load.dropna(how='any', subset=["QC_CVENERGYCOUNTER"], inplace=True)
    df_load.dropna(how='any', subset=["qc_cvenergycounter"], inplace=True) 

def FillZeroThermalForHP2(df_load):
    #hp2_CiCproproty = ['hp2_thermalEnergyCounter']
    #hp2_CiCproproty = ['HP2_THERMALENERGYCOUNTER']
    hp2_CiCproproty = ['hp2_thermalenergycounter']

    for CiCproperty in hp2_CiCproproty:
        df_load[CiCproperty] = df_load[CiCproperty].fillna(0)
    df_load = df_load.dropna()



