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

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ERN5Dataset(data.Dataset):
    """ERN5 dataset for HAT model training.
    
    Adapted to work with a single HDF5 file with shape (27010, 721, 1440).
    Optimized for large h5 files with file handle caching and efficient data loading.
    """
    
    # Class-level file handle cache with thread lock for safety
    _file_handle = None
    _file_lock = threading.Lock()
    
    def __init__(self, opt):
        super(ERN5Dataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.gt_file = osp.join(self.gt_folder, 'gt_rechunked.h5')
        self.factor = opt.get('scale', 8)
        self.mean = 278.06824
        self.std = 21.676298
        
        # Check if the file exists
        if not osp.exists(self.gt_file):
            raise FileNotFoundError(f"HDF5 file not found: {self.gt_file}")
        
        # Open the file to get metadata and keep it open for faster access
        self._init_file_handle()
        
        # Set patch size
        self.gt_size = opt.get('gt_size', 256)
        # Calculate LQ size based on scale factor
        self.lq_size = self.gt_size // self.factor
        
        # Set up transforms
        self.use_hflip = opt.get('use_hflip', True)
        self.use_rot = opt.get('use_rot', True)
        
        # Pre-compute valid crop regions to avoid repeated calculations
        self.max_h_idx = max(0, self.data_shape[1] - self.gt_size)
        self.max_w_idx = max(0, self.data_shape[2] - self.gt_size)
        
        # Calculate number of possible crops in height and width dimensions
        self.h_crops = math.floor(self.data_shape[1] / self.gt_size)
        self.w_crops = math.floor(self.data_shape[2] / self.gt_size)
        
        # Total number of possible crops per time index
        self.crops_per_time = self.h_crops * self.w_crops
        
        # Cache for time indices to improve access patterns
        self.time_indices = np.arange(self.data_shape[0])
        
        # Determine if we're using random or sequential cropping
        self.random_crop = opt.get('random_crop', True)
        
        # Prefetch settings
        self.prefetch = opt.get('prefetch_mode', None) is not None
        
        print(f"Dataset initialized with {self.data_shape[0]} time indices")
        print(f"Each time index can be divided into {self.crops_per_time} crops of size {self.gt_size}×{self.gt_size}")
        print(f"Crop mode: {'Random' if self.random_crop else 'Sequential'}")
        
    def _init_file_handle(self):
        """Initialize and cache the file handle at class level for reuse."""
        with ERN5Dataset._file_lock:
            if ERN5Dataset._file_handle is None:
                ERN5Dataset._file_handle = h5py.File(self.gt_file, 'r', swmr=True)
                
                # Get the dataset from the file
                if 'fields' in ERN5Dataset._file_handle:
                    self.dataset_key = 'fields'
                else:
                    # If 'fields' doesn't exist, try to get the first dataset in the file
                    self.dataset_key = list(ERN5Dataset._file_handle.keys())[0]
                
                # Get data shape
                self.data_shape = ERN5Dataset._file_handle[self.dataset_key].shape
                print(f"GT dataset shape: {self.data_shape}")
                
                # Check if the dataset has chunks (for efficient access)
                dataset = ERN5Dataset._file_handle[self.dataset_key]
                if not dataset.chunks:
                    print("Warning: H5 dataset is not chunked. This may impact performance.")
                else:
                    print(f"H5 dataset chunk size: {dataset.chunks}")
                print(f"!!! H5 read, H5 dataset shape: {dataset.shape}")

    def normalize(self, x):
        """Normalize data."""
        return (x - self.mean) / self.std
    
    def get_file_handle(self):
        """Get the cached file handle or create a new one if needed."""
        with ERN5Dataset._file_lock:
            if ERN5Dataset._file_handle is None or not ERN5Dataset._file_handle.id.valid:
                self._init_file_handle()
            return ERN5Dataset._file_handle
    
    def __getitem__(self, index):
        # Get the cached file handle
        f = self.get_file_handle()
        dataset = f[self.dataset_key]
        
        if self.random_crop:
            # Random crop mode - select random time index and position
            time_idx = np.random.choice(self.time_indices)
            rnd_h = np.random.randint(0, self.max_h_idx + 1)
            rnd_w = np.random.randint(0, self.max_w_idx + 1)
        else:
            # Sequential crop mode - determine time index and position from index
            # First, determine which time index to use
            time_idx = index // self.crops_per_time
            time_idx = time_idx % self.data_shape[0]  # Wrap around if needed
            
            # Then, determine which crop within that time index
            crop_idx = index % self.crops_per_time
            h_idx = crop_idx // self.w_crops
            w_idx = crop_idx % self.w_crops
            
            # Calculate the exact position
            rnd_h = h_idx * self.gt_size
            rnd_w = w_idx * self.gt_size
            
            # Ensure we don't go out of bounds
            rnd_h = min(rnd_h, self.max_h_idx)
            rnd_w = min(rnd_w, self.max_w_idx)
        
        # Extract GT patch - use direct slicing for efficiency
        img_gt = dataset[time_idx, rnd_h:rnd_h + self.gt_size, rnd_w:rnd_w + self.gt_size]
        
        # Convert to numpy array (in case it's a h5py dataset)
        # Use asarray instead of array for zero-copy when possible
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
        
        # Apply augmentation manually instead of using the augment function
        if self.use_hflip and np.random.random() < 0.5:
            img_gt = torch.flip(img_gt, [0])
            img_lq = torch.flip(img_lq, [0])
            
        if self.use_rot and np.random.random() < 0.5:
            img_gt = torch.rot90(img_gt, k=1, dims=[0, 1])
            img_lq = torch.rot90(img_lq, k=1, dims=[0, 1])
        
        # Add channel dimension
        img_gt = img_gt.unsqueeze(0)
        img_lq = img_lq.unsqueeze(0)
        
        # Get file path for reference
        gt_path = self.gt_file
        
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}
    
    def __len__(self):
        if self.opt['phase'] == 'train':
            if self.random_crop:
                # For random crop mode, return a large number
                return 100000
            else:
                # For sequential crop mode, return the actual number of possible crops
                return self.data_shape[0] * self.crops_per_time
        else:
            # For validation, limit the size
            return min(10000, self.data_shape[0] * self.crops_per_time)
            
    def __del__(self):
        """Clean up resources when the dataset is deleted."""
        # File handle is managed at class level, so we don't close it here
        pass
        
    @classmethod
    def cleanup(cls):
        """Class method to explicitly close the file handle."""
        with cls._file_lock:
            if cls._file_handle is not None:
                cls._file_handle.close()
                cls._file_handle = None 