import concurrent.futures
import logging
from typing import Dict, List

from logiscanpy.core.processor import Processor

_LOGGER = logging.getLogger(__name__)


class Application:
    """Application class for managing multiple video analytics processors."""

    def __init__(self, configs: List[Dict[str, str]]):
        """Initialize the Application instance.

        Args:
            configs (List[Dict[str, str]]): List of configuration dictionaries for each camera stream.
        """
        self._configs = configs
        self._processors = []
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(configs))

    def initialize(self) -> bool:
        """Initialize the Application instance.

        Returns:
            bool: True if initialization is successful for all processors, False otherwise.
        """
        _LOGGER.info("[INIT] Initializing Application...")

        for config in self._configs:
            processor = Processor(config)
            if not processor.initialize():
                _LOGGER.error("[INIT] Failed to initialize processor for camera: %s", config["id"])
                return False

            self._processors.append(processor)

        _LOGGER.info("[INIT] Application initialized successfully")
        return True

    def run(self) -> None:
        """Start video processing for all camera streams in parallel."""
        _LOGGER.info("[RUN] Starting processors...")

        futures = [self._executor.submit(self._process_and_cleanup, processor) for processor in self._processors]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if isinstance(result, Exception):
                _LOGGER.error("An error occurred during processing:", exc_info=result)

    def _process_and_cleanup(self, processor):
        """Wrapper function to process and clean up a single processor."""
        try:
            processor.run()
        finally:
            processor.cleanup()

    def cleanup(self) -> None:
        """Shutdown the executor and clean up remaining resources."""
        _LOGGER.info("[CLEANUP] Shutting down executor...")
        self._executor.shutdown(wait=True)
        _LOGGER.info("[CLEANUP] Executor shutdown completed.")
