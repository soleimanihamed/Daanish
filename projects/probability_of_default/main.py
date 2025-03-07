from Daanish.utils.config_utils import initialize_daanish
import os
import sys

# Add Daanish root to Python path (if not already added)
daanish_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if daanish_root not in sys.path:
    sys.path.append(daanish_root)


# Initialize Daanish core setup
config = initialize_daanish()
