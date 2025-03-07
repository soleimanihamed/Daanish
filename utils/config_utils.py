# daanish/config_utils.py
import os
import sys
from configparser import ConfigParser
from dotenv import load_dotenv


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
