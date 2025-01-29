from dagster import (
    AssetSelection,
    Definitions,
    ScheduleDefinition,
    load_assets_from_modules,
    EnvVar,
    define_asset_job,
    Config,
)

from . import assets
from .resources import snowflake_io_manager, snowflake_client

all_assets = load_assets_from_modules([assets])

kpi_config = {
    "ops": {
        "load_raw_data": {"config": {"kpi": "COPPerformance"}},  # default kpi
        "apply_filters": {"config": {"kpi": "COPPerformance"}},
        "calculate_kpi": {"config": {"kpi": "COPPerformance"}},
        "save_results": {"config": {"kpi": "COPPerformance"}},
        "plot_results": {"config": {"kpi": "COPPerformance"}},
    }
}


fleetperformance_job = define_asset_job(
    name="fleetperformance_job",
    selection=AssetSelection.all(),  
    config=kpi_config,  
)

fleetperformance_schedule = ScheduleDefinition(
    name="fleetperformance_schedule",
    target=AssetSelection.all(),
    cron_schedule="8 * * * *",  # every hour + 8 mins
    # cron_schedule="0 12 * * *", # every day at 12:00  
)


defs = Definitions(
    assets=all_assets,
    schedules=[fleetperformance_schedule],
    resources={
        "io_manager": snowflake_io_manager, 
        "snowflake_client": snowflake_client
    },
)




