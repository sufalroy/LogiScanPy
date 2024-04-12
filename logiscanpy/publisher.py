"""Publisher module for publishing object count messages."""

import logging
import time

import pika
import pika.exceptions

logger = logging.getLogger(__name__)

hostname = "localhost"
port = 5672
username = "guest"
password = "guest"
exchange_name = "object-count-exchange"
routing_key = "object-count.new"


class Publisher:
    """Publisher class to publish object count messages."""

    def __init__(self):
        self.connection = None
        self.channel = None
        self.connect_to_rabbitmq()

    def connect_to_rabbitmq(self):
        """Connect to RabbitMQ server."""
        try:
            credentials = pika.PlainCredentials(username, password)
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(hostname, port, "/", credentials)
            )
            self.channel = self.connection.channel()
            self.channel.exchange_declare(
                exchange=exchange_name, exchange_type="topic", durable=True
            )
            logger.info("Connected to RabbitMQ server.")
        except pika.exceptions.AMQPError as e:
            logger.error(f"Error connecting to RabbitMQ server: {e}")

    def publish_message(self, object_count):
        """Publish object count message."""
        message = {
            "count": object_count,
            "name": "sack",
            "timestamp": time.time(),
        }

        try:
            self.channel.basic_publish(
                exchange=exchange_name, routing_key=routing_key, body=str(message)
            )
            logger.info(f"Published message: {message}")
        except (pika.exceptions.AMQPError, ConnectionResetError) as e:
            logger.error(f"Error publishing message: {e}")
            self.reconnect()
            self.publish_message(object_count)  # Retry publishing the message

    def reconnect(self):
        """Reconnect to RabbitMQ server."""
        self.close_connection()
        self.connect_to_rabbitmq()

    def close_connection(self):
        """Close the RabbitMQ connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("RabbitMQ connection closed.")
            except pika.exceptions.AMQPError as e:
                logger.error(f"Error closing RabbitMQ connection: {e}")
