from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    """
    Main function to run the model training pipeline for fraud detection.
    """
    spark = SparkSession.builder \
        .appName("PaySimFraudDetectionTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    data_path = 'medium-stories/data-enhancement/PS_20174392719_1491204439457_log.csv'
    print(f"Loading data from {data_path}...")
    
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # Select the features that will be used in the model
    # These were chosen based on the logic in the original and Scala versions
    df_labeled = df.withColumn('label', col('isFraud').cast('double')).drop('isFraud')

    print("Sampling the dataset for faster training...")
    # For a real-world scenario, you might not sample or would use techniques
    # to handle the imbalanced nature of the dataset.
    df_sampled = df_labeled.sample(withReplacement=False, fraction=0.1, seed=42)

    print("Defining the machine learning pipeline...")
    # Stage 1: Convert the 'type' string column to a numerical index
    type_indexer = StringIndexer(inputCol="type", outputCol="type_indexed", handleInvalid="keep")
    
    # Stage 2: Assemble feature columns into a single vector
    assembler_inputs = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'type_indexed', 'newbalanceDest', 'oldbalanceDest']
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Stage 3: Define the RandomForestClassifier model
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

    # Chain the stages into a pipeline
    pipeline = Pipeline(stages=[type_indexer, assembler, rf])
    
    print("Splitting data into training and testing sets...")
    (training_data, test_data) = df_sampled.randomSplit([0.8, 0.2], seed=42)

    print("Training the RandomForest model... This may take a few minutes.")
    model = pipeline.fit(training_data)
    print("Model training complete.")

    # Overwrite the model in the specified path
    model_path = "medium-stories/data-enhancement/spark_ml_model"
    print(f"Saving new fraud detection model to {model_path}...")
    model.write().overwrite().save(model_path)
    print("Model saved successfully.")

    print("Evaluating model performance on the test set...")
    predictions = model.transform(test_data)
    
    # Use F1-Score as it's a good metric for imbalanced datasets
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)
    
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)

    print(f"  Test Set F1-Score: {f1_score:.4f}")
    print(f"  Test Set Accuracy: {accuracy:.4f}")

    print("Extracting a sample of data for the Kafka producer...")
    # The columns selected here should match what the producer and stream processor expect
    producer_sample_df = df_labeled.select('type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud')
    
    # Take 100 records and write them to a JSON file for the producer to use
    kafka_sample_data = producer_sample_df.limit(100).toJSON().collect()
    
    output_path = "medium-stories/data-enhancement/kafka_sample_data.json"
    print(f"Overwriting sample data at {output_path}...")
    with open(output_path, 'w') as f:
        for row in kafka_sample_data:
            f.write(row + '\n')
            
    print(f"100 sample records saved to {output_path}.")
    print("\n--- Model training script finished successfully! ---")

    spark.stop()

if __name__ == "__main__":
    main() 