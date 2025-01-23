from dagster_snowflake_pandas import SnowflakePandasIOManager
from dagster import EnvVar

# 定义 SnowflakePandasIOManager 资源
snowflake_io_manager = SnowflakePandasIOManager(
    account="iejbhuk-quatt",
    user=EnvVar("SNOWFLAKE_USER"),
    password=EnvVar("SNOWFLAKE_PASSWORD"),
    warehouse="COMPUTE_WH",
    # database="DEV_SICHEN",
    # schema="IRIS",
    database="DWH",
    schema="DEV_SICHEN",
    role="DATA_ANALYST",
)

