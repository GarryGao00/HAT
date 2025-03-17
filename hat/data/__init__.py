import importlib
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.registry import DATASET_REGISTRY

# automatically scan and import dataset modules for registry
# scan all the files that end with '_dataset.py' under the data folder
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]
# import all the dataset modules
print("Importing dataset modules:")
print(dataset_filenames)
_dataset_modules = [importlib.import_module(f'hat.data.{file_name}') for file_name in dataset_filenames]

# Explicitly import the evaluation dataset to ensure it's registered
try:
    import hat.data.ern5_eval_dataset
except ImportError:
    print("Warning: Could not import ern5_eval_dataset")

# Print all registered datasets for debugging
print("Registered datasets:", list(DATASET_REGISTRY.keys()))