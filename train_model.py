from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from typing import Tuple


def create_spark_session() -> SparkSession:
    """Creates and configures a Spark Session."""
    print("Creating Spark session...")
    spark = SparkSession.builder \
        .appName("PaySimFraudDetectionTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_data(spark: SparkSession, data_path: str) -> DataFrame:
    """Loads the transaction data from a CSV file."""
    print(f"Loading data from {data_path}...")
    return spark.read.csv(data_path, header=True, inferSchema=True)


def engineer_features(df: DataFrame) -> DataFrame:
    """Engineers new features and adds the 'label' column."""
    print("Engineering features...")
    df_features = df.withColumn("fromTo",
                                when(col("nameOrig").contains("C") & col("nameDest").contains("C"), "CC")
                                .when(col("nameOrig").contains("C") & col("nameDest").contains("M"), "CM")
                                .when(col("nameOrig").contains("M") & col("nameDest").contains("C"), "MC")
                                .when(col("nameOrig").contains("M") & col("nameDest").contains("M"), "MM")
                                .otherwise(None)
                                ) \
        .withColumn("HourOfDay", col("step") % 24)

    df_labeled = df_features.withColumn('label', col('isFraud').cast('double')).drop('isFraud')
    return df_labeled


def balance_data(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """Performs undersampling on the majority class to balance the dataset."""
    print("Balancing data by undersampling...")
    majority_df = df.filter(col("label") == 0)
    minority_df = df.filter(col("label") == 1)

    minority_count = minority_df.count()
    majority_count = majority_df.count()

    if majority_count == 0 or minority_count == 0:
        print("Cannot balance data, one class is empty.")
        return df, minority_df
    sampling_ratio = minority_count / majority_count
    print(f"Before undersampling: Majority count={majority_count}, Minority count={minority_count}")

    sampled_majority_df = majority_df.sample(withReplacement=False, fraction=sampling_ratio, seed=42)
    df_balanced = sampled_majority_df.unionAll(minority_df)

    print(f"After undersampling: New majority count={sampled_majority_df.count()}, Minority count={minority_df.count()}")
    return df_balanced, minority_df


def build_pipeline() -> Pipeline:
    """Defines and builds the machine learning pipeline."""
    print("Defining the machine learning pipeline...")
    categorical_cols = ['type', 'fromTo']
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep") for c in categorical_cols
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_ohe") for c in categorical_cols
    ]

    numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'newbalanceDest', 'oldbalanceDest', 'HourOfDay']
    ohe_cols = [f"{c}_ohe" for c in categorical_cols]
    assembler_inputs = numerical_cols + ohe_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="skip")

    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")
    rf = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures", numTrees=100)

    return Pipeline(stages=indexers + encoders + [assembler, scaler, rf])


def train_and_evaluate(pipeline: Pipeline, df_balanced: DataFrame) -> PipelineModel:
    """Splits data, trains the model, and evaluates its performance."""
    print("Splitting data into training and testing sets...")
    (training_data, test_data) = df_balanced.randomSplit([0.8, 0.2], seed=42)

    print("Training the RandomForest model... This may take a few minutes.")
    model = pipeline.fit(training_data)
    print("Model training complete.")

    print("Evaluating model performance on the test set...")
    predictions = model.transform(test_data)
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)

    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_accuracy.evaluate(predictions)

    print(f"  Test Set F1-Score: {f1_score:.4f}")
    print(f"  Test Set Accuracy: {accuracy:.4f}")

    return model


def save_outputs(model: PipelineModel, df_fraud: DataFrame, model_path: str, sample_data_path: str):
    """Saves the trained model and sample data for the producer."""
    print(f"Saving new fraud detection model to {model_path}...")
    model.write().overwrite().save(model_path)
    print("Model saved successfully.")

    print(f"Extracting a sample of FRAUDULENT data to {sample_data_path}...")
    producer_sample_df = df_fraud.select('step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud')
    kafka_sample_data = producer_sample_df.limit(100).toJSON().collect()

    with open(sample_data_path, 'w') as f:
        for row in kafka_sample_data:
            f.write(row + '\n')

    print(f"{len(kafka_sample_data)} sample records saved.")


def main():
    """Main function to run the model training pipeline."""
    spark = create_spark_session()

    raw_df = load_data(spark, 'PS_20174392719_1491204439457_log.csv')
    df_labeled = engineer_features(raw_df)
    df_balanced, minority_df = balance_data(df_labeled)
    pipeline = build_pipeline()
    model = train_and_evaluate(pipeline, df_balanced)
    save_outputs(model, minority_df,
                 model_path="data-enhancement-model/spark_ml_model",
                 sample_data_path="data-enhancement-model/kafka_sample_data.json")

    print("\n--- Model training script finished successfully! ---")
    spark.stop()


if __name__ == "__main__":
    main()
