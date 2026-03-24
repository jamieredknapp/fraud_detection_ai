import os
import sys
from pyspark.sql.functions import sum as spark_sum, count

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

spark = SparkSession.builder \
    .master("spark://172.23.125.26:7077") \
    .getOrCreate()
input("Press Enter to exit...")