import json
import time
from kafka import KafkaProducer

def create_producer(bootstrap_servers='localhost:9092'):
    """
    Creates and returns a Kafka producer.
    
    Args:
        bootstrap_servers (str): The Kafka bootstrap servers.
    
    Returns:
        KafkaProducer: A configured Kafka producer instance.
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
        # In a real application, you might want to handle this more gracefully.
        # For this example, we will exit if we can't connect.
        exit()

def stream_data(producer, topic_name='raw_transactions', data_file='medium-stories/data-enhancement/kafka_sample_data.json'):
    """
    Reads data from a file and sends it to a Kafka topic.
    
    Args:
        producer (KafkaProducer): The Kafka producer to use.
        topic_name (str): The name of the Kafka topic.
        data_file (str): The path to the file containing sample data.
    """
    print(f"Reading sample data from {data_file}...")
    with open(data_file, 'r') as f:
        # readlines() is used for simplicity. For very large files,
        # consider reading line by line to avoid memory issues.
        records = f.readlines()

    print(f"Starting to stream {len(records)} records to topic '{topic_name}'...")
    print("Press Ctrl+C to stop the stream.")

    try:
        while True:
            for record_str in records:
                record_json = json.loads(record_str)
                producer.send(topic_name, value=record_json)
                print(f"Sent Transaction: Type={record_json['type']}, Amount={record_json['amount']}")
                # Wait for 1-3 seconds to simulate a real-time stream
                time.sleep(time.uniform(1, 3))
            print("Completed a full loop over the data. Restarting...")
            time.sleep(5) # Wait before restarting the loop
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        producer.flush()
        producer.close()
        print("Kafka producer flushed and closed.")

def main():
    """
    Main function to run the Kafka producer.
    """
    producer = create_producer()
    stream_data(producer)

if __name__ == "__main__":
    main() 