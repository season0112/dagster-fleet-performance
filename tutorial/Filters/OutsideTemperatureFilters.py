def FilterOnHP1OutsideTemperature(df_load):
    df_load.dropna(subset=['hp1_temperatureoutside'], inplace=True)


