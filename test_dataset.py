import os
import sys
import torch
from basicsr.utils.registry import DATASET_REGISTRY

# Add the current directory to the path
sys.path.insert(0, os.path.abspath('.'))

# Import the dataset modules
import hat.data

# Print all registered dataset classes
print("Registered dataset classes:")
for name in DATASET_REGISTRY.keys():
    print(f"  - {name}")

# Check if ERN5Dataset is registered
if 'ERN5Dataset' in DATASET_REGISTRY.keys():
    print("ERN5Dataset is registered correctly!")
else:
    print("ERN5Dataset is NOT registered!")
    
    # Try to import it directly
    print("Trying to import ERN5Dataset directly...")
    try:
        from hat.data.ern5_dataset import ERN5Dataset
        print("Import successful!")
        
        # Check if it's registered after direct import
        if 'ERN5Dataset' in DATASET_REGISTRY.keys():
            print("ERN5Dataset is now registered!")
        else:
            print("ERN5Dataset is still NOT registered!")
    except Exception as e:
        print(f"Import failed: {e}") 