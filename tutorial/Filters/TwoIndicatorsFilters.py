
def FilterOnRoomTempAvialable(df_load):
    df_load.dropna(subset=['thermostat_otFtRoomTemperature'], inplace=True)

def FilterOnWaterTempAvialable(df_load):
    df_load.dropna(subset=['qc_supplyTemperatureFiltered'], inplace=True)

def FilterOnOutsideTempAvialable(df_load):
    df_load.dropna(subset=['hp1_temperatureOutside'], inplace=True)

def FilterOnHPHeatAvialable(df_load):
    df_load.dropna(subset=['hp1_thermalEnergyCounter'], inplace=True)

def FilterOnBoilerHeatAvialable(df_load):
    df_load.dropna(subset=['qc_cvEnergyCounter'], inplace=True)



