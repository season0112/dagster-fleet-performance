import KPIUtility

def FilterOnSetPointTempAvialable(df_load):
    df_load.dropna(subset=['thermostat_otFtRoomSetpoint'], inplace=True)

def FilterOnRoomTempAvialable(df_load):
    df_load.dropna(subset=['thermostat_otFtRoomTemperature'], inplace=True)


