import logging
import os
import sys
import time
from datetime import datetime

def setup_logging(log_dir="logs"):
    """
    Set up logging to both console and file
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate a timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/tangram_training_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_filename}")
    
    return logger, log_filename

class LoggerWriter:
    """
    Redirect stdout/stderr to logger
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buf = ""

    def write(self, message):
        self.buf = self.buf + message
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if line.strip() != "":
                self.logger.log(self.level, line.rstrip())

    def flush(self):
        if self.buf != "":
            self.logger.log(self.level, self.buf.rstrip())
            self.buf = "" 