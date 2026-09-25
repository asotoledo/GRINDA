# Configuration file for the GRINDA project
import os

# Define project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
