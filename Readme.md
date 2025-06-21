# Real-Time Fraud Detection with Kafka, Spark, and Scala

This project demonstrates a real-time fraud detection pipeline using a hybrid architecture that leverages the strengths of both Python and Scala. It uses Apache Kafka for message brokering and Apache Spark for stream processing and machine learning.

The core of the stream processing is built in Scala for performance and robustness, while the machine learning model training, data generation, and the data producer are implemented in Python for its rich data science ecosystem and ease of use.

## Project Architecture

The data flows through the system as follows:

1.  **`producer.py` (Python)**: Reads sample transaction data (specifically, fraudulent transactions generated during model training) from a JSON file and streams it to a Kafka topic named `raw_transactions`.
2.  **`StreamProcessor.scala` (Scala)**: A Spark Streaming application that consumes data from the `raw_transactions` topic. It performs the same feature engineering as the training script, applies the pre-trained ML model to predict fraud, and writes the enriched data (including the fraud prediction and probability) to an `enhanced_transactions` Kafka topic.
3.  **`train_model.py` (Python)**: A PySpark script that performs the following tasks:
    *   Loads the raw transaction dataset from a CSV file.
    *   Engineers features required for the model.
    *   Trains a Random Forest Classifier to distinguish fraudulent transactions.
    *   Saves the trained Spark ML Pipeline to disk.
    *   Saves a sample of fraudulent transactions to `kafka_sample_data.json` for the producer to use.

## Core Technologies

*   **Programming Languages**: Scala, Python
*   **Big Data Frameworks**: Apache Spark (Spark SQL, Spark Streaming, Spark MLlib)
*   **Messaging System**: Apache Kafka
*   **Build Tool**: Apache Maven (for the Scala application)

## How to Run the Project

Follow these steps to get the complete pipeline running.

### Prerequisites

*   **Java**: Version 8 or 11.
*   **Python**: Version 3.7+.
*   **Apache Maven**: For building the Scala project.
*   **Apache Spark**: Must be installed and available on your system `PATH`.
*   **Apache Kafka**: A running Kafka cluster. The application is configured to connect to `kafka-cluster.local:9092`. You will also need to create the topics `raw_transactions` and `enhanced_transactions`.

### Step 1: Set Up the Python Environment

First, set up a virtual environment and install the required Python packages.

```bash
cd medium-stories/data-enhancement

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Train the ML Model

Run the training script. This will train the model, save it to the `data-enhancement-model/` directory, and create the `kafka_sample_data.json` file needed by the producer.

```bash
python train_model.py
```

### Step 3: Build the Scala Stream Processor

Use Maven to compile the Scala code and package it into a single "fat" JAR that includes all dependencies.

```bash
mvn clean package
```

This command will produce a JAR file in the `target/` directory, for example: `target/data-enhancement-1.0-SNAPSHOT.jar`.

### Step 4: Start the Scala Stream Processor

Use `spark-submit` to run the Scala application. This application will wait for data to arrive on the `raw_transactions` Kafka topic.

Make sure to provide the necessary Spark-Kafka packages that match your Spark and Scala versions.

```bash
spark-submit \
  --master local \
  --deploy-mode client \
  --class StreamProcessor \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1 \
  target/data-enhancement-1.0-SNAPSHOT.jar
```
**Note:** The `--packages` argument downloads the required Kafka connector. You might need to adjust the version (`3.2.1`) to match your Spark installation.

### Step 5: Start the Python Producer

Finally, in a new terminal (with the Python virtual environment activated), start the producer to begin sending data to the stream processor.

```bash
source venv/bin/activate
python producer.py
```

You should now see the `StreamProcessor` application processing the data in its terminal window as the producer sends it. You can optionally set up a Kafka consumer to listen to the `enhanced_transactions` topic to see the final output. 