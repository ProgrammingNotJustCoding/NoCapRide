import os
import logging
import threading
import time


class CustomLogger:
    def __init__(self, name: str, log_file: str):
        
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  

        
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)  
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        
        self.logger.propagate = False

        
        self._start_log_flusher(file_handler)

        
        self.logger.debug("Logger initialized for %s", name)

    def _start_log_flusher(self, file_handler):
        def flush_logs():
            while True:
                time.sleep(5)
                file_handler.flush()

        flusher_thread = threading.Thread(target=flush_logs, daemon=True)
        flusher_thread.start()

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def critical(self, message: str):
        self.logger.critical(message)
