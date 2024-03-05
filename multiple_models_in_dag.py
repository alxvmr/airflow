import io
import json
import logging
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import NoReturn, Literal, Dict, Any

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner" : "Maria Alexeeva",
    "email" : "alexeevamro@gmail.com",
    "email_on_failure" : True,
    "email_on_retry" : False,
    "retry" : 3,
    "retry_delay" : timedelta(minutes=1)
}

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = "test-bucket-alxvmr"
DATA_PATH = "datasets/california_housing"
FEATURES = ["MedInc", "HouseAge", "AveRooms", 
            "AveBedrms", "Population", "AveOccup", 
            "Latitude", "Longitude"]
TARGET = "MedHouseVal"


models = dict(zip(["rf", "lr", "hgb"], [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()]))

dag = DAG(
    dag_id = "multiple_models_in_dag",
    schedule_interval = "0 1 * * *",
    start_date = days_ago(2),
    catchup = False,
    tags = ["mlops"],
    default_args = DEFAULT_ARGS
)

def init() -> Dict[str, Any]:
    metrics = {}
    metrics["start_timestamp"] = datetime.now().strftime("%Y_%n_%d_%H")
    return metrics

def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "init")
        
        # датасет уже лежит на s3
        return metrics

def prepare_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "get_data")

        return metrics

def train_model(name_model) -> Dict[str, Any]:
        m_name = name_model
        metrics = {}
        metrics["name_model"] = name_model

        s3_hook = S3Hook("s3_connection")
        data = {}
        for name in ["X_train", "X_test", "y_train", "y_test"]:
            file = s3_hook.download_file(key=f"datasets/{name}.pkl", bucket_name=BUCKET)
            data[name] = pd.read_pickle(file)

        model = models[m_name]
        metrics["train_start"] = datetime.now().strftime("%Y_%n_%d_%H")
        model.fit(data["X_train"], data["y_train"])
        prediction = model.predict(data["X_test"])
        metrics["train_end"] = datetime.now().strftime("%Y_%n_%d_%H")

        metrics["r2_score"] = r2_score(data["y_test"], prediction)
        metrics["rmse"] = mean_squared_error(data["y_test"], prediction)
        metrics["mse"] = median_absolute_error(data["y_test"], prediction)

        return metrics

def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids = "prepare_data")
        metrics["models_metrics"] = []
        
        for n in models.keys():
            metrics["models_metrics"].append(ti.xcom_pull(task_ids = f"train_model_{n}"))

        metrics["end_timestamp"] = datetime.now().strftime("%Y_%n_%d_%H")

        # сохраняем на s3
        s3_hook = S3Hook("s3_connection")
        date = datetime.now().strftime("%Y_%n_%d_%H")
        session = s3_hook.get_session("ru-central1")
        resource = session.resource("s3")
        json_byte_object = json.dumps(metrics)
        resource.Object(BUCKET, f"results/multiple_models_in_dag_{date}.json").put(Body=json_byte_object)


with dag:
    task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)
    task_get_data = PythonOperator(task_id="get_data", python_callable=get_data_from_postgres, dag=dag, provide_context=True)
    task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag, provide_context=True)

    list_tasks_train_model = []
    for n in models.keys():
        list_tasks_train_model.append(PythonOperator(task_id=f"train_model_{n}", python_callable=train_model, dag=dag, op_kwargs={"name_model": n}))
    task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag, provide_context=True)

    for task_train_model in list_tasks_train_model:
        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results