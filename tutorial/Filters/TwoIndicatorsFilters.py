
def FilterOnRoomTempAvialable(df_load):
    df_load.dropna(subset=['thermostat_otftroomtemperature'], inplace=True)

def FilterOnWaterTempAvialable(df_load):
    df_load.dropna(subset=['qc_supplytemperaturefiltered'], inplace=True)

def FilterOnOutsideTempAvialable(df_load):
    df_load.dropna(subset=['hp1_temperatureoutside'], inplace=True)

def FilterOnHPHeatAvialable(df_load):
    df_load.dropna(subset=['hp1_thermalenergycounter'], inplace=True)

def FilterOnBoilerHeatAvialable(df_load):
    df_load.dropna(subset=['qc_cvenergycounter'], inplace=True)



