import json
from kafka import KafkaConsumer

def create_consumer(topic_name='enhanced_transactions', bootstrap_servers='kafka-cluster.local:9092'):
    """
    Creates and returns a Kafka consumer subscribed to a specific topic.
    """
    print(f"Connecting to Kafka and subscribing to topic '{topic_name}'...")
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',  # Start reading from the latest message
            value_deserializer=lambda v: json.loads(v.decode('utf-8'))
        )
        print("Successfully connected and subscribed.")
        return consumer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        exit()

def consume_messages(consumer):
    """
    Continuously listens for and prints messages from the Kafka consumer.
    """
    print("\n--- Listening for Enhanced Fraud Predictions ---")
    print("Press Ctrl+C to stop the consumer.")
    
    try:
        for message in consumer:
            data = message.value
            print("\nReceived Enhanced Transaction:")
            print(f"  Type: {data.get('type', 'N/A')}, Amount: ${data.get('amount', 0):,.2f}")
            prediction = 'FRAUD' if data.get('fraud_prediction') == 1.0 else 'Not Fraud'
            print(f"  Prediction: {prediction} (Probability: {data.get('fraud_probability', 0.0):.4f})")
            print("---------------------------------------------")
            
    except KeyboardInterrupt:
        print("\nConsumer stopped by user.")
    finally:
        consumer.close()
        print("Kafka consumer closed.")

def main():
    """
    Main function to run the Kafka consumer.
    """
    consumer = create_consumer()
    consume_messages(consumer)

if __name__ == "__main__":
    main() 