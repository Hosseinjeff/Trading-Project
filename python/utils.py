import logging
import os

# Configure a shared logger
def setup_logger(module_name, log_file='project_log.log'):
    """Set up a logger for a module."""
    logger = logging.getLogger(module_name)
    if not logger.hasHandlers():
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger

def log_step(logger, message, script_name=None):
    """Log a step with an optional script name."""
    if script_name is None:
        script_name = os.path.basename(__file__)
    logger.info(f"{script_name} - {message}")
