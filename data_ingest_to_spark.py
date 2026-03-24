df = spark.read.csv(
    "/home/ubuntu/data/fraudTrain.csv",
    header=True,
    inferSchema=True
)