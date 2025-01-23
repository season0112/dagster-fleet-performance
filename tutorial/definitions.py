from dagster import Definitions
from .assets import iris_dataset, iris_cleaned, iris_harvest_data
from .resources import snowflake_io_manager

# 定义 Dagster 的 Definitions
defs = Definitions(
    assets=[iris_dataset, iris_harvest_data, iris_cleaned],
    resources={"io_manager": snowflake_io_manager},
)

