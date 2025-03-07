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
    # Step 1: Load Environment Variables from .env or config.ini
    load_dotenv()

    # Step 2: Define the path to config.ini
    config_path = os.getenv('CONFIG_PATH')
    if not config_path:
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 'config.ini'))

    # Step 3: Load global config from config.ini
    global_config = ConfigParser()
    global_config.read(config_path)

    # Step 4: Validate configuration
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
