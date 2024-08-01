# Date: 2024/8/1 10:39
# Author: cls1277
# Email: cls1277@163.com

import json

class Config:
    data = None
    @staticmethod
    def load_config(file_path="./config.json"):
        try:
            with open(file_path, 'r') as file:
                Config.data = json.load(file)
            print("Configuration loaded successfully.")
        except FileNotFoundError:
            print(f"Config file {file_path} does not exist.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    @staticmethod
    def get_value(key, default=None):
        keys = key.split('.')
        value = Config.data
        try:
            for k in keys:
                value = value[k]
            return value
        except (TypeError, KeyError):
            return default