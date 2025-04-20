# Daanish/main.py

import os
import sys
from utils.core.config import initialize_daanish

# Add Daanish root to Python path
daanish_root = os.path.abspath(os.path.dirname(__file__))
if daanish_root not in sys.path:
    sys.path.append(daanish_root)


# Initialize Daanish core setup
config = initialize_daanish()

# Your main application logic here
print("Daanish is ready!")
