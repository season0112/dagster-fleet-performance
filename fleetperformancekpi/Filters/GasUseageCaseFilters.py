
def FilterOnBoilerRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([3, 4])].index, inplace=True)

def FilterOnHP1WatchdogNullAnd0(df_load):
    df_load.dropna(subset=['hp1_watchdogcode'], inplace=True)
    df_load.drop(df_load[df_load['hp1_watchdogcode'] == 0].index, inplace=True) 

def FilterOnHP1WatchdogNull(df_load):
    df_load.dropna(subset=['hp1_watchdogcode'], inplace=True)

def FilterOnHP1RatedPowerNull(df_load):
    df_load.dropna(subset=['hp1_ratedpower'], inplace=True)

def FilterOnEstimatedPowerNull(df_load):
    df_load.dropna(subset=['qc_estimatedpowerdemand'], inplace=True)


