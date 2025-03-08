# daanish/config_utils.py
import os
import sys
from configparser import ConfigParser
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """
    Load configuration from config.ini or environment variables.
    Returns a ConfigParser object.
    """
    # Load environment variables from .env
    load_dotenv()

    # First, check if CONFIG_PATH is set in the environment
    config_path = os.getenv('CONFIG_PATH')

    # If CONFIG_PATH is not set, look for config.ini in the project root
    if not config_path:
        project_root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), ".."))  # Moves up one directory
        config_path = os.path.join(project_root, "config.ini")

    # Load global config from config.ini
    global_config = ConfigParser()
    global_config.read(config_path)

    # Validate configuration
    if not global_config.sections():
        raise FileNotFoundError(f"config.ini not found at {config_path}")

    return global_config


def load_project_config(project_root):
    """
    Load project-specific configuration from project_config.ini.

    Args:
        project_root (str): The root directory of the project.

    Returns:
        ConfigParser: The project-specific configuration.
    """
    # Define the path to project_config.ini
    config_path = os.path.join(project_root, 'project_config.ini')
    logger.info(f"Loading project configuration from {config_path}")

    # Load the configuration
    project_config = ConfigParser()
    project_config.read(config_path)

    # Validate the configuration
    if not project_config.sections():
        raise FileNotFoundError(
            f"project_config.ini not found at {config_path}")

    logger.info("Project configuration loaded successfully")
    return project_config


def setup_python_path(global_config):
    """
    Set up PYTHONPATH from global configuration.
    """
    project_root = global_config.get('global', 'PYTHONPATH', fallback=None)
    if project_root and project_root not in sys.path:
        sys.path.append(project_root)


def initialize_daanish():
    """
    Initialize Daanish core setup (load config and set up Python path).
    """
    global_config = load_config()
    setup_python_path(global_config)
    print("Daanish core setup complete.")
    return global_config


def get_database_config(global_config):
    """
    Extract database configuration from the global config.

    Args:
        global_config (ConfigParser): The global configuration object.

    Returns:
        dict: A dictionary containing database configuration values.
    """
    if not global_config.has_section('DATABASE'):
        raise ValueError("Missing [DATABASE] section in config.ini")

    db_config = {
        'server': global_config.get('DATABASE', 'server', fallback=None),
        'database': global_config.get('DATABASE', 'database', fallback=None),
        'username': global_config.get('DATABASE', 'username', fallback=None),
        'password': global_config.get('DATABASE', 'password', fallback=None),
        'use_database': global_config.getboolean('DATABASE', 'use_database', fallback=False)
    }

    # Check if any required field is missing or empty
    missing_fields = [key for key, value in db_config.items(
    ) if not value and key != 'use_database']
    if missing_fields:
        raise ValueError(
            f"Missing or empty database configuration in config.ini: {missing_fields}")

    return db_config
