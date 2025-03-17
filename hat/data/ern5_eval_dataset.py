import cv2
import numpy as np
import os.path as osp
import torch
import h5py
import torch.nn.functional as F
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import threading
import math
import os

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ERN5EvalDataset(data.Dataset):
    """ERN5 dataset for HAT model evaluation.
    
    Adapted to work with ERA5 HDF5 files with shape (730, 14, 721, 1440).
    Designed for super-resolution evaluation with the HAT model.
    """
    
    def __init__(self, opt):
        super(ERN5EvalDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.gt_file = osp.join(self.gt_folder, '2016.h5')
        
        # Scale factor for super-resolution
        self.factor = opt.get('scale', 8)
        
        # Normalization constants from E5_eval
        self.mean = 278.06824
        self.std = 21.676298
        
        # Check if the file exists
        if not osp.exists(self.gt_file):
            print(f"HDF5 file not found: {self.gt_file}")
            raise FileNotFoundError(f"HDF5 file not found: {self.gt_file}")
        
        # Open the file temporarily to get metadata
        with h5py.File(self.gt_file, 'r') as f:
            # Get the dataset from the file
            if 'fields' in f:
                self.dataset_key = 'fields'
            else:
                # If 'fields' doesn't exist, try to get the first dataset in the file
                self.dataset_key = list(f.keys())[0]
            
            # Get data shape
            self.data_shape = f[self.dataset_key].shape
            print(f"GT dataset shape: {self.data_shape}")
            
            # Handle different data shapes
            if len(self.data_shape) == 4 and self.data_shape[1] > 3:
                # Format is (time, channels, height, width)
                # We'll use temperature field (index 2)
                print(f"Using 4D data with channel index 2")
                self.use_channel_index = 2
                self.effective_shape = (self.data_shape[0], self.data_shape[2], self.data_shape[3])
            elif len(self.data_shape) == 3:
                # Format is (time, height, width)
                print(f"Using 3D data directly")
                self.use_channel_index = None
                self.effective_shape = self.data_shape
            else:
                print(f"Unsupported data shape: {self.data_shape}")
                raise ValueError(f"Unsupported data shape: {self.data_shape}")
            
            # Check if the dataset has chunks (for efficient access)
            dataset = f[self.dataset_key]
            if not dataset.chunks:
                print("Warning: H5 dataset is not chunked. This may impact performance.")
            else:
                print(f"H5 dataset chunk size: {dataset.chunks}")
        
        # Set patch size
        self.gt_size = opt.get('gt_size', 256)
        # Calculate LQ size based on scale factor
        self.lq_size = self.gt_size // self.factor
        
        # Pre-compute valid crop regions to avoid repeated calculations
        self.max_h_idx = max(0, self.effective_shape[1] - self.gt_size)
        self.max_w_idx = max(0, self.effective_shape[2] - self.gt_size)
        
        # Calculate number of possible crops in height and width dimensions
        self.h_crops = max(1, math.floor(self.effective_shape[1] / self.gt_size))
        self.w_crops = max(1, math.floor(self.effective_shape[2] / self.gt_size))
        
        # Total number of possible crops per time index
        self.crops_per_time = self.h_crops * self.w_crops
        
        # For evaluation, we'll use a fixed set of time indices
        self.eval_time_indices = np.arange(min(70, self.effective_shape[0]))
        
        # Instance-specific file handle - will be initialized in __getitem__
        self._file_handle = None
        
        print(f"Evaluation dataset initialized with {len(self.eval_time_indices)} time indices")
        print(f"Each time index can be divided into {self.crops_per_time} crops of size {self.gt_size}×{self.gt_size}")
        print(f"Total number of evaluation samples: {len(self)}")

    def normalize(self, x):
        """Normalize data."""
        return (x - self.mean) / self.std
    
    def _get_file_handle(self):
        """Get a file handle for this instance."""
        if self._file_handle is None or not self._file_handle.id.valid:
            self._file_handle = h5py.File(self.gt_file, 'r')
        return self._file_handle
    
    def __getitem__(self, index):
        # Get file handle for this worker
        f = self._get_file_handle()
        dataset = f[self.dataset_key]
        
        # Determine time index and position from index
        time_idx = index // self.crops_per_time
        time_idx = self.eval_time_indices[time_idx % len(self.eval_time_indices)]
        
        # Determine which crop within that time index
        crop_idx = index % self.crops_per_time
        h_idx = crop_idx // self.w_crops
        w_idx = crop_idx % self.w_crops
        
        # Calculate the exact position
        rnd_h = h_idx * self.gt_size
        rnd_w = w_idx * self.gt_size
        
        # Ensure we don't go out of bounds
        rnd_h = min(rnd_h, self.max_h_idx)
        rnd_w = min(rnd_w, self.max_w_idx)
        
        # Extract GT patch based on the data format
        if self.use_channel_index is not None:
            # For multi-channel datasets (like climate data), select the temperature field
            img_gt = dataset[time_idx, self.use_channel_index, rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size]
        else:
            # For single-channel datasets
            img_gt = dataset[time_idx, rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size]
        
        # Convert to numpy array (in case it's a h5py dataset)
        img_gt = np.asarray(img_gt)
        
        # Convert to tensor and normalize
        img_gt = torch.from_numpy(img_gt).float()
        img_gt = self.normalize(img_gt)
        
        # Generate LQ image (downsampled)
        img_gt_expanded = img_gt.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        # Directly downsample to the correct LQ size
        img_lq = F.interpolate(
            img_gt_expanded, 
            size=(self.lq_size, self.lq_size),
            mode='bicubic'
        )[0, 0]  # Remove batch and channel dimensions
        
        # Add channel dimension
        img_gt = img_gt.unsqueeze(0)
        img_lq = img_lq.unsqueeze(0)
        
        # Get file path for reference
        gt_path = f"{self.gt_file}:{time_idx}"
        
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}
    
    def __len__(self):
        # For evaluation, limit the size to a reasonable number
        length = len(self.eval_time_indices) * self.crops_per_time
        # Ensure at least 1 sample
        return max(1, min(1000, length))
            
    def __del__(self):
        """Clean up resources when the dataset is deleted."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None