"""
Load saved models and generate rounded predictions for comparison
"""
import numpy as np
import pandas as pd
import sys
import os

# Read comparison script to extract data loading logic
print("Extracting predictions from saved models...")
print("="*80)

# Just run the model loading code from existing script
exec(open('compare_models_real_data.py').read())
