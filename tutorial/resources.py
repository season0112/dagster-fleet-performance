from dagster_snowflake_pandas import SnowflakePandasIOManager
import snowflake.connector
from dagster import resource, EnvVar
import os

# 用于管理 Snowflake 资产存储的 IO Manager
snowflake_io_manager = SnowflakePandasIOManager(
    account="iejbhuk-quatt",
    user=os.environ.get("SNOWFLAKE_USER"),
    password=os.environ.get("SNOWFLAKE_PASSWORD"),
    warehouse="COMPUTE_WH",
    database="DWH",
    schema="DEV_SICHEN",
    role="DATA_ANALYST",
)

@resource
def snowflake_client(context):
    # 创建一个与 Snowflake 的连接
    conn = snowflake.connector.connect(
        account="iejbhuk-quatt",
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        warehouse="COMPUTE_WH",
        database="DWH",
        schema="DEV_SICHEN",
        role="DATA_ANALYST",
    )
    
    # 返回用于执行查询的 Snowflake 连接
    try:
        yield conn
    finally:
        conn.close()


