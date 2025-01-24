
def FilterOnBoilerRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([3, 4])].index, inplace=True)

def FilterOnHP1WatchdogNullAnd0(df_load):
    df_load.dropna(subset=['hp1_watchdogCode'], inplace=True)
    df_load.drop(df_load[df_load['hp1_watchdogCode'] == 0].index, inplace=True) 

def FilterOnHP1WatchdogNull(df_load):
    df_load.dropna(subset=['hp1_watchdogCode'], inplace=True)

def FilterOnHP1RatedPowerNull(df_load):
    df_load.dropna(subset=['hp1_ratedPower'], inplace=True)

def FilterOnEstimatedPowerNull(df_load):
    df_load.dropna(subset=['qc_estimatedPowerDemand'], inplace=True)


