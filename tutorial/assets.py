import pandas as pd
from dagster import op, Out, job, AssetIn, asset, resource, EnvVar, get_dagster_logger, AssetExecutionContext, Config
import argparse
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import KPIUtility
from . import Filters
from . import calculateKPI
from . import loadRawCSVData
import Utility.Plot_Functions as Plot_function

logger = get_dagster_logger()

class KpiConfig(Config):
    kpi: str  # 定义 kpi 参数

@asset
def printVariable() -> None:
    snowflake_user = os.environ.get("SNOWFLAKE_USER")
    snowflake_password = os.environ.get("SNOWFLAKE_PASSWORD")
    print(f"SNOWFLAKE_USER: {snowflake_user}")
    print(f"SNOWFLAKE_PASSWORD: {snowflake_password}")


@asset(required_resource_keys={"snowflake_client"})
def raw_cic_dataset(context) -> pd.DataFrame:
    query = """
        SELECT
            HP1_THERMALENERGYCOUNTER,
            HP2_THERMALENERGYCOUNTER,
            QC_CVENERGYCOUNTER,
            QC_SUPERVISORYCONTROLMODE,
            CLIENT_TIME,
            CLIENTID
        FROM
            FIREHOSE_CIC.CIC.STATS_DYNAMIC
        WHERE
            -- CLIENTID LIKE 'cic-5795534%'
            -- CLIENTID = 'cic-5795534f-50c3-5cac-be43-80d8d52310b3'   -- start from 2023-06-05 06:53:21
            CLIENTID = 'cic-319d1ceb-4ee9-51dd-a13a-18eb047dd625' -- Mark's CiC  

            -- AND CLIENT_TIME BETWEEN DATEADD(HOUR, -1, DATE_TRUNC('HOUR', CURRENT_TIMESTAMP())) AND DATE_TRUNC('HOUR', CURRENT_TIMESTAMP())   -- for example: trigger on 10:08, CLIENT_TIME will be 09:00 to 10:00
            -- AND CLIENT_TIME BETWEEN DATEADD(DAY, -1, CURRENT_TIMESTAMP()) AND CURRENT_TIMESTAMP() -- last 24 hours
            AND CLIENT_TIME BETWEEN DATEADD(DAY, -10, CURRENT_TIMESTAMP()) AND CURRENT_TIMESTAMP()

            AND QC_SUPERVISORYCONTROLMODE in (2, 3, 4)
        ORDER BY
            CLIENT_TIME 

    """

    # 获取 Snowflake 连接并执行查询
    context.log.info("Querying Snowflake...")
    conn = context.resources.snowflake_client
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    # 将查询结果转换为 DataFrame
    df = pd.DataFrame(result, columns=columns)
    df.columns = df.columns.astype(str)  # 强制将列名转换为字符串

    return df


@asset
# def apply_filters(context, raw_cic_dataset, kpi: str) -> pd.DataFrame:
def apply_filters(context, raw_cic_dataset) -> pd.DataFrame:
    #kpi = config.kpi  # 从配置中获取 kpi
    kpi = "HeatPercentage"

    print("raw_cic_dataset is:")
    print(str(raw_cic_dataset))
    print(raw_cic_dataset.columns)

    """应用过滤器"""
    logger.info(f"Applying filters for KPI: {kpi}")
    # generalfilterList = ['FilterOnTestRigs', 'FilterOnInactiveCiC', 'FilterOnDropZero', 'FilterOnDropABit', 'FilterOnIncreaseCounter', 'FilterOnAllNull']
    # KPIUtility.applyfilters(raw_cic_dataset, generalfilterList, Filters.GeneralFilters, filters_sector_name='General Filters')

    if kpi == "COPPerformance":
        COPfilterList = ['FillZeroForHP2', 'FilterOnCOPTrainDataAvialable', 'FilterOnZeroCounter', 'FilterOnOnlyPumpRunning']
        KPIUtility.applyfilters(raw_cic_dataset, COPfilterList, Filters.COPFilters, filters_sector_name='COP Filters')
    elif kpi == "HeatPercentage":
        HeatPercentagefilterList = ['FilterOnCVEnergyAvialable', 'FillZeroThermalForHP2'] # FilterOnBothPumpAndBoilerRunning
        KPIUtility.applyfilters(raw_cic_dataset, HeatPercentagefilterList, Filters.HeatPercentageFilters, filters_sector_name='HeatPercentage Filters')
    # 其他 KPI 的过滤器逻辑可以继续添加

    context.log.info(f"Applied filters for KPI: {kpi}")
    return raw_cic_dataset


@asset
#def calculate_kpi(context, apply_filters, kpi: str) -> dict:
def calculate_kpi(context, apply_filters) -> pd.DataFrame:

    #kpi = config.kpi  # 从配置中获取 kpi
    kpi = "HeatPercentage"

    """计算 KPI"""
    logger.info(f"Calculating KPI: {kpi}")
    # 这里可以调用你的 KPI 计算逻辑
    if kpi == "COPPerformance":
        # 示例：计算 COPPerformance
        MergeWindow = 1440
        MeasuredCOP_allCiC_RespectiveBins = []
        kpi_results = {}
        # 其他计算逻辑
    elif kpi == "HeatPercentage":
        # MergeWindow = 1440  # 合并窗口大小为 1440 分钟（1 天）
        MergeWindow = 1440
        #MergeWindow_BinEdge, BinCenterForMergeWindow, df_OneCiC = KPIUtility.MergeWindow(apply_filters, "HeatPercentage", MergeWindow, apply_filters['time_ts'].min().normalize(), apply_filters['time_ts'].max(), str(MergeWindow) + 'min')
        MergeWindow_BinEdge, BinCenterForMergeWindow, df_OneCiC = KPIUtility.MergeWindow(apply_filters, "HeatPercentage", MergeWindow, apply_filters['client_time'].min().normalize(), apply_filters['client_time'].max(), str(MergeWindow) + 'min')
        GasPercentage_allCiC_RespectiveBins = []
        HeatPumpPercentage_allCiC_RespectiveBins = []
        for cic in apply_filters['clientid'].unique():
            df_OneCiC = apply_filters[apply_filters['clientid'] == cic].copy()
            print("df_OneCiC:" + str(df_OneCiC))
            GasPercentage_tem, HeatPumpPercentage_tem = KPIUtility.CalculateGasPercentage_withWindow(df_OneCiC, MergeWindow_BinEdge)
            GasPercentage_allCiC_RespectiveBins.append(GasPercentage_tem)
            HeatPumpPercentage_allCiC_RespectiveBins.append(HeatPumpPercentage_tem)
        # 返回结果
        '''
        kpi_results = {
            "GasPercentage": GasPercentage_allCiC_RespectiveBins,
            "HeatPumpPercentage": HeatPumpPercentage_allCiC_RespectiveBins,
            "BinCenterForMergeWindow": BinCenterForMergeWindow
        }
        '''
        print("here:")

        print("GasPercentage_allCiC_RespectiveBins:" + str(GasPercentage_allCiC_RespectiveBins))
        print("HeatPumpPercentage_allCiC_RespectiveBins:" + str(HeatPumpPercentage_allCiC_RespectiveBins))
        print("BinCenterForMergeWindow:" + str(BinCenterForMergeWindow) )

        GasPercentage_allCiC_RespectiveBins = GasPercentage_allCiC_RespectiveBins[0]
        HeatPumpPercentage_allCiC_RespectiveBins = HeatPumpPercentage_allCiC_RespectiveBins[0]

        kpi_results = pd.DataFrame({
            "GasPercentage" : GasPercentage_allCiC_RespectiveBins,
            "HeatPumpPercentage" : HeatPumpPercentage_allCiC_RespectiveBins,
            "BinCenterForMergeWindow" : BinCenterForMergeWindow
        })

        '''
        kpi_results = {
            "GasPercentage": pd.Series(GasPercentage_allCiC_RespectiveBins),
            "HeatPumpPercentage": pd.Series(HeatPumpPercentage_allCiC_RespectiveBins),
            "BinCenterForMergeWindow": pd.Series(BinCenterForMergeWindow),
        }

        kpi_results = pd.DataFrame(kpi_results)
        '''

    # 其他 KPI 的计算逻辑...


    return kpi_results



@asset
def KPI_results(calculate_kpi) -> pd.DataFrame:
    df = pd.DataFrame(calculate_kpi)
    return df

