from abstract import DataRepository, Model
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

class S3YandexDataRepository(DataRepository):
    def __init__(self, hook_name):
        self.hook = S3Hook(hook_name)