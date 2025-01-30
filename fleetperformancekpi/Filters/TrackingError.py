import KPIUtility

def FilterOnSetPointTempAvialable(df_load):
    df_load.dropna(subset=['thermostat_otftroomsetpoint'], inplace=True)

def FilterOnRoomTempAvialable(df_load):
    df_load.dropna(subset=['thermostat_otftroomtemperature'], inplace=True)


