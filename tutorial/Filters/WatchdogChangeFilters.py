import KPIUtility

def FilterOnHP1Watchdog(df_load):
    df_load.dropna(subset=['hp1_watchdogcode'], inplace=True)

def FilterOnSystemWatchdog(df_load):
    df_load.dropna(subset=['qc_systemwatchdogcode'], inplace=True)

def FilterOnHP1WatchdogNullAnd0(df_load):
    df_load.dropna(subset=['hp1_watchdogcode'], inplace=True)
    df_load.drop(df_load[df_load['hp1_watchdogcode'] == 0].index, inplace=True)



 


