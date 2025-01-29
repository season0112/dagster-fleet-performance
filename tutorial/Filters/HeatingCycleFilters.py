import KPIUtility

def FilterOnPumpRunningOnly(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([2])].index, inplace=True)

def FilterOnPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisorycontrolmode'].isin([2, 3])].index, inplace=True)

def FilterOnSupervisoryControlModeNotNull(df_load):
    df_load.dropna(how='all', subset=["qc_supervisorycontrolmode"], inplace=True)


