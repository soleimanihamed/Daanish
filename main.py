import sys
import os
from dotenv import load_dotenv
from configparser import ConfigParser

# Step 1: Load Environment Variables from .env or config.ini
load_dotenv()

# Load global config from config.ini
global_config = ConfigParser()
global_config.read(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'config.ini')))

# Set PYTHONPATH from global configuration
project_root = global_config.get('global', 'PYTHONPATH', fallback=None)
if project_root and project_root not in sys.path:
    sys.path.append(project_root)

print("Daanish core setup complete.")
