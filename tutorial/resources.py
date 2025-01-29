from dagster_snowflake_pandas import SnowflakePandasIOManager
import snowflake.connector
from dagster import resource, EnvVar
import os


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
    conn = snowflake.connector.connect(
        account="iejbhuk-quatt",
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        warehouse="COMPUTE_WH",
        database="DWH",
        schema="DEV_SICHEN",
        role="DATA_ANALYST",
    )
    
    try:
        yield conn
    finally:
        conn.close()


