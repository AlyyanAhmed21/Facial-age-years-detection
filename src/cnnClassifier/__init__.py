import os
import sys
import logging

# Define the logging format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define the directory for log files
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


# Configure the logging
logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),  # Log to a file
        logging.StreamHandler(sys.stdout)   # Also print to the console
    ]
)

# Create the logger object
logger = logging.getLogger("cnnClassifierLogger")