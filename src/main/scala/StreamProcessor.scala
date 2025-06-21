import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector

object StreamProcessor {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = createSparkSession

    println("--- Starting Real-Time Fraud Detection Stream ---")

    val model: PipelineModel = loadModel

    val kafkaSchema: StructType = createSchema

    val kafkaBootstrapServers = "kafka-cluster.local:9092"
    val rawTransactionsTopic = "raw_transactions"

    val parsedStreamDF = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("subscribe", rawTransactionsTopic)
      .option("startingOffsets", "latest")
      .load()
      .select(from_json(col("value").cast("string"), kafkaSchema).alias("data"))
      .select("data.*")

    // --- Perform same feature engineering as in training ---
    println("Applying feature engineering to the stream...")
    val engineeredStreamDF = parsedStreamDF
      .withColumn("fromTo",
        when(col("nameOrig").contains("C") && col("nameDest").contains("C"), "CC")
          .when(col("nameOrig").contains("C") && col("nameDest").contains("M"), "CM")
          .when(col("nameOrig").contains("M") && col("nameDest").contains("C"), "MC")
          .when(col("nameOrig").contains("M") && col("nameDest").contains("M"), "MM")
          .otherwise(lit(null))
      )
      .filter(col("fromTo").isNotNull)
      .withColumn("HourOfDay", col("step") % 24)

    println("Applying ML model to the stream...")
    val predictionsDF = model.transform(engineeredStreamDF)

    // --- Format the output for the 'enhanced_transactions' topic ---
    val extractProbability = udf((v: Vector) => v(1))

    val outputDF = predictionsDF
      .withColumn("fraud_probability", extractProbability(col("probability")))
      .select(
        to_json(
          struct(
            col("type"),
            col("amount"),
            col("prediction").alias("fraud_prediction"),
            col("fraud_probability")
          )
        ).alias("value")
      )

    val enhancedTransactionsTopic = "enhanced_transactions"
    val checkpointLocation = "/tmp/spark_checkpoints/fraud_detection_scala"
    
    outputDF
      .writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("topic", enhancedTransactionsTopic)
      .option("checkpointLocation", checkpointLocation)
      .start()
      .awaitTermination()
  }

  private def createSchema: StructType = {
    StructType(Array(
      StructField("step", IntegerType, nullable = true),
      StructField("type", StringType, nullable = true),
      StructField("amount", DoubleType, nullable = true),
      StructField("nameOrig", StringType, nullable = true),
      StructField("oldbalanceOrg", DoubleType, nullable = true),
      StructField("newbalanceOrig", DoubleType, nullable = true),
      StructField("nameDest", StringType, nullable = true),
      StructField("oldbalanceDest", DoubleType, nullable = true),
      StructField("newbalanceDest", DoubleType, nullable = true),
      StructField("isFlaggedFraud", IntegerType, nullable = true)
    ))
  }

  private def loadModel: PipelineModel = {
    val modelPath = "/home/damian/business/medium-stories/data-enhancement/data-enhancement-model/spark_ml_model"
    val model = PipelineModel.load(modelPath)
    println("Model loaded successfully.")
    model
  }

  private def createSparkSession = {
    val spark = SparkSession.builder
      .appName("RealTimeFraudDetection")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark
  }
}