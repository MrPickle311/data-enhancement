from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, MinMaxScaler
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

    data_path = 'PS_20174392719_1491204439457_log.csv'
    print(f"Loading data from {data_path}...")

    raw_df = spark.read.csv(data_path, header=True, inferSchema=True)

    # --- Feature Engineering ---
    # Create new features. These transformations return new DataFrames.
    df_features = raw_df.withColumn("fromTo",
                                    when(col("nameOrig").contains("C") & col("nameDest").contains("C"), "CC")
                                    .when(col("nameOrig").contains("C") & col("nameDest").contains("M"), "CM")
                                    .when(col("nameOrig").contains("M") & col("nameDest").contains("C"), "MC")
                                    .when(col("nameOrig").contains("M") & col("nameDest").contains("M"), "MM")
                                    .otherwise(None)
                                    ) \
        .withColumn("HourOfDay", col("step") % 24)

    # Add the label column for training, using the feature-engineered DataFrame
    df_labeled = df_features.withColumn('label', col('isFraud').cast('double')).drop('isFraud')

    # --- Undersampling to handle class imbalance ---
    # Separate the majority and minority classes
    majority_df = df_labeled.filter(col("label") == 0)
    minority_df = df_labeled.filter(col("label") == 1)

    # Get the counts of each class
    minority_count = minority_df.count()
    majority_count = majority_df.count()

    # Calculate the desired ratio. Here, we'll aim for a 1:1 ratio for simplicity and effectiveness.
    # For every fraud case, we will keep one non-fraud case.
    sampling_ratio = minority_count / majority_count

    print(f"Before undersampling: Majority count={majority_count}, Minority count={minority_count}")

    # Undersample the majority class
    sampled_majority_df = majority_df.sample(withReplacement=False, fraction=sampling_ratio, seed=42)

    # Combine the undersampled majority class with the original minority class
    df_balanced = sampled_majority_df.unionAll(minority_df)

    print(
        f"After undersampling: New majority count={sampled_majority_df.count()}, Minority count={minority_df.count()}")

    print("Defining the machine learning pipeline...")
    # Stages for one-hot encoding categorical variables
    categorical_cols = ['type', 'fromTo']
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
        for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]

    # Stage to assemble all feature columns into a single vector
    numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'oldbalanceDest',
                      'HourOfDay']
    ohe_cols = [f"{c}_ohe" for c in categorical_cols]
    assembler_inputs = numerical_cols + ohe_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Stage for the RandomForestClassifier model, now using the scaled features
    rf = RandomForestClassifier(numTrees=100)

    # Chain all stages into a pipeline, including the new scaler
    pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

    print("Splitting data into training and testing sets...")
    # We now use the balanced dataframe for training. We still need a separate, original test set for a fair evaluation.
    (training_data, test_data) = df_balanced.randomSplit([0.8, 0.2], seed=42)

    print("Training the RandomForest model... This may take a few minutes.")
    model = pipeline.fit(training_data)
    print("Model training complete.")

    # Overwrite the model in the specified path
    model_path = "data-enhancement-model/spark_ml_model"
    print(f"Saving new fraud detection model to {model_path}...")
    model.write().overwrite().save(model_path)
    print("Model saved successfully.")

    print("Evaluating model performance on the test set...")
    predictions = model.transform(test_data)

    # Use F1-Score as it's a good metric for imbalanced datasets
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)

    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                           metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)

    print(f"  Test Set F1-Score: {f1_score:.4f}")
    print(f"  Test Set Accuracy: {accuracy:.4f}")

    print("Extracting a sample of FRAUDULENT data for the Kafka producer...")
    # The columns selected here must match the schema expected by the StreamProcessor
    # We select from the original minority_df which contains only records where isFraud == 1 (label == 1.0)
    producer_sample_df = minority_df.select('step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig',
                                            'nameDest',
                                            'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud')

    # Take 100 records and write them to a JSON file for the producer to use
    kafka_sample_data = producer_sample_df.limit(100).toJSON().collect()

    output_path = "data-enhancement-model/kafka_sample_data.json"
    print(f"Overwriting sample data at {output_path}...")
    with open(output_path, 'w') as f:
        for row in kafka_sample_data:
            f.write(row + '\n')

    print(f"100 sample records saved to {output_path}.")
    print("\n--- Model training script finished successfully! ---")

    spark.stop()


if __name__ == "__main__":
    main()
