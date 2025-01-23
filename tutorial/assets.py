import pandas as pd
from dagster import asset, AssetSpec

# 定义 AssetSpec
iris_harvest_data = AssetSpec(key="iris_harvest_data")

# 定义资产
@asset
def iris_dataset() -> pd.DataFrame:
    return pd.read_csv(
        "https://docs.dagster.io/assets/iris.csv",
        names=[
            "sepal_length_cm",
            "sepal_width_cm",
            "petal_length_cm",
            "petal_width_cm",
            "species",
        ],
    )


@asset
def iris_cleaned(iris_dataset: pd.DataFrame) -> pd.DataFrame:
    return iris_dataset.dropna().drop_duplicates()

