import json
import time
from kafka import KafkaProducer
import random

def create_producer(bootstrap_servers='kafka-cluster.local:9092'):
    """
    Creates and returns a Kafka producer.
    """
    print(f"Connecting to Kafka at {bootstrap_servers}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("Successfully connected to Kafka.")
        return producer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        exit()

def load_records(data_file):
    """
    Reads transaction records from a JSON file.
    """
    print(f"Reading sample data from {data_file}...")
    try:
        with open(data_file, 'r') as f:
            # For very large files, this would be memory-intensive.
            # Reading line-by-line would be better in a production scenario.
            records = f.readlines()
            print(f"Successfully loaded {len(records)} records.")
            return records
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file}")
        print("Please run the training script first to generate the sample data.")
        exit()

def stream_records(producer, topic_name, records):
    """
    Continuously sends records to a Kafka topic.
    """
    print(f"Starting to stream to topic '{topic_name}'...")
    print("Press Ctrl+C to stop the stream.")

    try:
        while True:
            for record_str in records:
                try:
                    record_json = json.loads(record_str)
                    producer.send(topic_name, value=record_json)
                    print(f"Sent Transaction: Type={record_json.get('type', 'N/A')}, Amount={record_json.get('amount', 0)}")
                    # Wait for 1-3 seconds to simulate a real-time stream
                    time.sleep(random.uniform(1, 3))
                except json.JSONDecodeError:
                    # This can happen if a line in the file is not a valid JSON.
                    print(f"Could not decode JSON, skipping record: {record_str}")
            print("Completed a full loop over the data. Restarting...")
            time.sleep(5) # Wait before restarting the loop
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")

def main():
    """
    Main function to set up and run the Kafka producer.
    """
    KAFKA_BOOTSTRAP_SERVERS = 'kafka-cluster.local:9092'
    KAFKA_TOPIC = 'raw_transactions'
    DATA_FILE = 'data-enhancement-model/kafka_sample_data.json'

    producer = create_producer(KAFKA_BOOTSTRAP_SERVERS)
    records = load_records(DATA_FILE)

    try:
        stream_records(producer, KAFKA_TOPIC, records)
    finally:
        producer.flush()
        producer.close()
        print("Kafka producer flushed and closed.")

if __name__ == "__main__":
    main() 