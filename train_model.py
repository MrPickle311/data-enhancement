import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
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

    # --- 1. Data Loading and Feature Explanation ---
    print("--- Starting Fraud Detection Model Training ---")
    print("\nFeature Explanation:")
    print("""
    We are building a model to detect fraudulent financial transactions in real-time.
    Based on common fraud patterns, we've selected the following features from the PaySim dataset:
    
    - type: The type of transaction (e.g., 'CASH_OUT', 'TRANSFER'). Fraud often occurs in specific transaction types.
    - amount: The transaction amount. Unusually large amounts can be a red flag.
    - oldbalanceOrg: The balance of the sender's account before the transaction.
    - newbalanceOrig: The balance of the sender's account after the transaction. Drastic changes or discrepancies can indicate fraud.
    - isFraud: Our target variable. '1' for a fraudulent transaction, '0' otherwise.
    """)

    data_path = 'PS_20174392719_1491204439457_log.csv'
    print(f"Loading data from {data_path}...")
    
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    # --- 2. Feature Selection & Preprocessing ---
    feature_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']
    
    df_selected = df.select(*feature_columns)

    # Rename the label column to 'label' for consistency
    df_labeled = df_selected.withColumn('label', col('isFraud').cast('double')).drop('isFraud')

    # The dataset is large. Let's sample it to speed up training for this demonstration.
    # NOTE: The dataset is also highly imbalanced. For a production system, techniques like
    # SMOTE (Synthetic Minority Over-sampling Technique) or using weights would be essential.
    # For this example, we proceed with a simple random sample.
    print("Sampling the dataset for faster training...")
    df_sampled = df_labeled.sample(withReplacement=False, fraction=0.1, seed=42)

    # --- 3. ML Pipeline Definition ---
    print("Defining the machine learning pipeline...")
    # Stage 1: StringIndexer for the 'type' column
    type_indexer = StringIndexer(inputCol="type", outputCol="type_indexed", handleInvalid="keep")

    # Stage 2: VectorAssembler to combine all features into a single vector
    assembler_inputs = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'type_indexed']
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Stage 3: RandomForestClassifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)

    # Combine all stages into a Pipeline
    pipeline = Pipeline(stages=[type_indexer, assembler, rf])
    
    # --- 4. Train/Test Split & Model Training ---
    print("Splitting data into training and testing sets...")
    (training_data, test_data) = df_sampled.randomSplit([0.8, 0.2], seed=42)

    print("Training the RandomForest model... This may take a few minutes.")
    model = pipeline.fit(training_data)
    print("Model training complete.")

    # --- 5. Model Saving ---
    model_path = "medium-stories/data-enhancement/spark_ml_model"
    print(f"Saving new fraud detection model to {model_path}...")
    model.write().overwrite().save(model_path)
    print("Model saved successfully.")

    # --- 6. Model Evaluation ---
    print("Evaluating model performance on the test set...")
    predictions = model.transform(test_data)
    
    # F1-Score is a better metric for imbalanced datasets like this one.
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)
    
    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)

    print(f"  Test Set F1-Score: {f1_score:.4f}")
    print(f"  Test Set Accuracy: {accuracy:.4f}")


    # --- 7. Sample Data Extraction for Kafka Producer ---
    print("Extracting a sample of data for the Kafka producer...")
    # Select columns that the streaming app will expect. We don't need the label.
    producer_sample_df = df_labeled.select('type', 'amount', 'oldbalanceOrg', 'newbalanceOrig')
    
    # Take 100 records and convert to a list of JSON strings
    kafka_sample_data = producer_sample_df.limit(100).toJSON().collect()
    
    output_path = "medium-stories/data-enhancement/kafka_sample_data.json"
    print(f"Overwriting sample data at {output_path} with new transaction data.")
    with open(output_path, 'w') as f:
        for row in kafka_sample_data:
            f.write(row + '\n')
            
    print(f"100 sample records for producer saved to {output_path}.")
    print("\n--- Model training script finished successfully! ---")

    spark.stop()

if __name__ == "__main__":
    main() 