import logging
import time

import pika
import pika.exceptions

logger = logging.getLogger(__name__)

_HOSTNAME = "localhost"
_PORT = 5672
_USERNAME = "guest"
_PASSWORD = "guest"
_EXCHANGE_NAME = "object-count-exchange"
_ROUTING_KEY = "object-count.new"


class Publisher:
    """Publisher class to publish object count messages."""

    def __init__(self):
        self._connection = None
        self._channel = None
        self._connect_to_rabbitmq()

    def _connect_to_rabbitmq(self):
        """Connect to RabbitMQ server."""
        try:
            credentials = pika.PlainCredentials(_USERNAME, _PASSWORD)
            self._connection = pika.BlockingConnection(pika.ConnectionParameters(_HOSTNAME, _PORT, "/", credentials))
            self._channel = self._connection.channel()
            self._channel.exchange_declare(exchange=_EXCHANGE_NAME, exchange_type="topic", durable=True)
            logger.info("Connected to RabbitMQ server.")
        except pika.exceptions.AMQPError as e:
            logger.error(f"Error connecting to RabbitMQ server: {e}")

    def publish_message(self, object_name, object_count):
        """Publish object count message."""
        message = {
            "count": object_count,
            "name": object_name,
            "timestamp": time.time(),
        }

        try:
            self._channel.basic_publish(exchange=_EXCHANGE_NAME, routing_key=_ROUTING_KEY, body=str(message))
            logger.info(f"Published message: {message}")
        except (pika.exceptions.AMQPError, ConnectionResetError) as e:
            logger.error(f"Error publishing message: {e}")
            self._reconnect()
            self.publish_message(object_name, object_count)

    def _reconnect(self):
        """Reconnect to RabbitMQ server."""
        self.close_connection()
        self._connect_to_rabbitmq()

    def close_connection(self):
        """Close the RabbitMQ connection."""
        if self._connection:
            try:
                self._connection.close()
                logger.info("RabbitMQ connection closed.")
            except pika.exceptions.AMQPError as e:
                logger.error(f"Error closing RabbitMQ connection: {e}")
