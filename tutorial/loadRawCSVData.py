import pandas as pd

def LoadRawData(kpiname):

    match kpiname:
        case "COPPerformance":
            datafilename = "../Data/KPICOP/KPICOP_CiC1_01_12_2023_to_30_12_2023.csv"  
            # datafilename = "../Data/KPICOP/KPICOP_CiC1_25_10_2024_to_30_11_2024.csv" 
        case "OutsideTemperature":
            # KPIOutsideTemperature (Raw data Shared with KPICOP, spliting and make new reduced csv file just for faster process) 
            # This outside temperature KPI is more accurate than the temperature for COP fit, since the later one filter out the records outside test data range.
            # datafilename = "../Data/KPICOP/KPICOP_CiC1_25_10_2024_to_30_11_2024.csv"
            datafilename = "../Data/KPICOP/KPICOP_CiC1_01_12_2023_to_30_12_2023.csv"
        case "HeatingCycle":
            datafilename = "../Data/KPIHeatingCycle/KPIHeatingCycle_CiC1_01_12_2023_to_30_12_2023.csv"
            # datafilename = "../Data/KPIHeatingCycle/KPIHeatingCycle_CiC1_01_11_2023_to_30_01_2024.csv" 
            # datafilename = "../Data/KPIHeatingCycle/KPIHeatingCycle_CiC1_25_10_2024_to_30_11_2024.csv" 
        case "WatchdogChange":
            # datafilename = "../Data/KPIWatchdogChange/KPIWatchdogChange_CiC1_01_11_2023_to_30_01_2024.csv"  
            # datafilename = "../Data/KPIWatchdogChange/KPIWatchdogChange_CiC1_25_10_2024_to_30_11_2024.csv" 
            # datafilename = "../Data/KPIWatchdogChange/KPIWatchdogChange_CiC1_01_12_2023_to_02_12_2023.csv" 
            datafilename = "../Data/KPIWatchdogChange/KPIWatchdogChange_CiC1_01_12_2023_to_30_12_2023.csv"
        case "HeatPercentage":
            datafilename = "../Data/KPIHeatPercentage/KPI_HeatPercentage_CiC1_01_12_2023_to_30_12_2023.csv"
            # datafilename = "../Data/KPIHeatPercentage/KPI_HeatPercentage_CiC1_25_10_2024_to_30_11_2024.csv"
        case "RiseTime" | "TwoIndicators": 
            # datafilename = "../Data/KPITrackingErrorAndTwoIndicators/KPITrackingErrorAndTwoIndicators_CiC1_25_10_2024_to_26_10_2024.csv"
            datafilename = "../Data/KPITrackingErrorAndTwoIndicators/KPITrackingErrorAndTwoIndicators_CiC1_01_12_2023_to_30_12_2023.csv"
            # datafilename = "../Data/KPITrackingErrorAndTwoIndicators/KPITrackingErrorAndTwoIndicators_CiC1_25_10_2024_to_30_11_2024.csv"
        case "GasUseageCase":
            # datafilename = "../Data/KPIGasUseageCase/KPIGasUseageCase_CiC1_01_11_2023_to_30_01_2024.csv"
            # datafilename = "../Data/KPIGasUseageCase/KPIGasUseageCase_CiC1_01_12_2023_to_05_12_2023.csv"
            # datafilename = "../Data/KPIGasUseageCase/KPIGasUseageCase_CiC1_10_12_2023_to_20_12_2023.csv"
            datafilename = "../Data/KPIGasUseageCase/KPIGasUseageCase_CiC1_01_12_2023_to_30_12_2023.csv"
            # datafilename = "../Data/KPIGasUseageCase/KPIGasUseageCase_CiC1_25_10_2024_to_30_11_2024.csv"

    # EmployeeHeatPumpData
    #datafilename = "../Data/EmployeeHeatPump/EmployeeHeatPump.csv"

    print("#" * 69)
    print("\033[31mCalculating KPI: \033[35m" + str(kpiname) + "\033[0m")
    print("#" * 69)

    print("#" * 69)
    print("\033[1;31mStart Loading Raw Data:\033[0m")
    df_load = pd.read_csv(datafilename)
    print("\033[32mTotal Number of CiC Records: \033[35m" + str(len(df_load)) + "\033[0m")
    print("\033[32mTotal Number of Unique CiCs: \033[35m" + str(len(df_load['clientid'].unique())) + "\033[0m")
    print("\033[32mStart of collecting data: \033[35m" + str(min(df_load['time_ts'])) + "\033[0m")
    print("\033[32mEnd of collecting data: \033[35m" + str(max(df_load['time_ts'])) + "\033[0m")
    print("#" * 69)
    print('\n')

    return df_load

