from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, udf
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml import PipelineModel

def main():
    """
    Main function to run the Spark Structured Streaming application.
    """
    spark = SparkSession.builder \
        .appName("RealTimeFraudDetection") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    print("--- Starting Real-Time Fraud Detection Stream ---")

    # --- 1. Load the pre-trained ML model ---
    model_path = "medium-stories/data-enhancement/spark_ml_model"
    print(f"Loading pre-trained model from {model_path}...")
    model = PipelineModel.load(model_path)
    print("Model loaded successfully.")

    # --- 2. Define the schema for incoming Kafka data ---
    # This must match the structure of the JSON data produced by producer.py
    kafka_schema = StructType([
        StructField("type", StringType(), True),
        StructField("amount", DoubleType(), True),
        StructField("oldbalanceOrg", DoubleType(), True),
        StructField("newbalanceOrig", DoubleType(), True),
    ])

    # --- 3. Read from the 'raw_transactions' Kafka topic ---
    kafka_bootstrap_servers = 'kafka-cluster.local:9092'
    raw_transactions_topic = 'raw_transactions'

    print(f"Subscribing to Kafka topic '{raw_transactions_topic}'...")
    kafka_stream_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("subscribe", raw_transactions_topic) \
        .option("startingOffsets", "latest") \
        .load()

    # --- 4. Parse the JSON data and apply the ML model ---
    parsed_stream_df = kafka_stream_df \
        .select(from_json(col("value").cast("string"), kafka_schema).alias("data")) \
        .select("data.*")

    print("Applying ML model to the stream...")
    predictions_df = model.transform(parsed_stream_df)

    # --- 5. Format the output for the 'enhanced_transactions' topic ---
    extract_prob_udf = udf(lambda v: float(v[1]), DoubleType())

    output_df = predictions_df.withColumn("fraud_probability", extract_prob_udf(col("probability")))

    final_df = output_df.select(
        to_json(
            struct(
                col("type"),
                col("amount"),
                col("oldbalanceOrg"),
                col("newbalanceOrig"),
                col("prediction").alias("fraud_prediction"),
                col("fraud_probability")
            )
        ).alias("value")
    )

    # --- 6. Write the enhanced data to the 'enhanced_transactions' Kafka topic ---
    enhanced_transactions_topic = 'enhanced_transactions'
    checkpoint_location = '/tmp/spark_checkpoints/fraud_detection'
    
    print(f"Writing enhanced stream to Kafka topic '{enhanced_transactions_topic}'...")
    query = final_df \
        .writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
        .option("topic", enhanced_transactions_topic) \
        .option("checkpointLocation", checkpoint_location) \
        .start()

    print("Stream is now running. Waiting for termination...")
    print("Check the output of the consumer script to see the enhanced data.")
    query.awaitTermination()

if __name__ == "__main__":
    main() 