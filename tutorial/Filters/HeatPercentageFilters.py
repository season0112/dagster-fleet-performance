


def FilterOnBothPumpAndBoilerRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([3])].index, inplace=True)

def FilterOnBoilerRunningOnly(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([4])].index, inplace=True)

def FilterOnCVEnergyAvialable(df_load):
    df_load.dropna(how='any', subset=["qc_cvenergycounter"], inplace=True) 

def FillZeroThermalForHP2(df_load):
    hp2_CiCproproty = ['hp2_thermalenergycounter']

    for CiCproperty in hp2_CiCproproty:
        df_load[CiCproperty] = df_load[CiCproperty].fillna(0)
    df_load = df_load.dropna()



