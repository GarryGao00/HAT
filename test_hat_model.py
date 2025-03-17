#!/usr/bin/env python3
import os
import os.path as osp
import h5py
import numpy as np
from scipy.ndimage import zoom
import argparse
import sys

# Ensure HAT and custom datasets are imported properly
import hat.archs
import hat.data  # This should register all custom datasets
import hat.models
from basicsr.test import test_pipeline
from basicsr.utils.registry import DATASET_REGISTRY

def prepare_test_data(input_file, output_dir_gt, output_dir_lq):
    """
    Prepares test data by downsampling the original high resolution data by a factor of 8.
    """
    # Create output directories if they don't exist
    os.makedirs(output_dir_gt, exist_ok=True)
    os.makedirs(output_dir_lq, exist_ok=True)
    
    # Define output file paths - use filenames that match training data
    output_file_gt = os.path.join(output_dir_gt, 'gt.h5')  # Same filename as training
    output_file_lq = os.path.join(output_dir_lq, 'lq.h5')  # Same filename as training
    
    print(f"Processing test file: {input_file}")
    
    try:
        with h5py.File(input_file, 'r') as f_in, \
             h5py.File(output_file_gt, 'w') as f_gt, \
             h5py.File(output_file_lq, 'w') as f_lq:
            
            if 'fields' not in f_in:
                print(f"Dataset 'fields' not found in {input_file}")
                return False
            
            # Extract channel 2 data (similar to training data)
            gt_data = f_in['fields'][:, 2, :, :]  # Shape should be (time, height, width)
            
            # Create downscaled version (8x smaller) - only scale spatial dimensions
            scale_factor = 1/8
            lq_data = np.zeros((gt_data.shape[0], int(gt_data.shape[1]*scale_factor), int(gt_data.shape[2]*scale_factor)))
            
            print(f"GT data shape: {gt_data.shape}")
            print(f"LQ data shape (after downsampling): {lq_data.shape}")
            
            # Perform downsampling for each time step
            for i in range(gt_data.shape[0]):
                if i % 100 == 0:
                    print(f"Processing frame {i}/{gt_data.shape[0]}")
                lq_data[i] = zoom(gt_data[i], (scale_factor, scale_factor), order=1)
            
            # Create datasets in output files
            f_gt.create_dataset('fields', data=gt_data, dtype='float32', compression='gzip')
            f_lq.create_dataset('fields', data=lq_data, dtype='float32', compression='gzip')
            
            print(f"Successfully created test datasets:")
            print(f"GT: {output_file_gt} - Shape: {gt_data.shape}")
            print(f"LQ: {output_file_lq} - Shape: {lq_data.shape}")
            
            return True
            
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Prepare test data and run HAT model inference')
    parser.add_argument('--prepare_data', action='store_true', 
                       help='Whether to prepare test data by downsampling')
    parser.add_argument('--input_file', type=str, 
                       default='/global/cfs/cdirs/m4633/foundationmodel/ERA5/ERA5processed/test/2016.h5',
                       help='Path to the input test file')
    parser.add_argument('--output_dir_gt', type=str, 
                       default='/global/cfs/cdirs/m4633/foundationmodel/HATdata/test_gt',
                       help='Output directory for ground truth data')
    parser.add_argument('--output_dir_lq', type=str, 
                       default='/global/cfs/cdirs/m4633/foundationmodel/HATdata/test_lq',
                       help='Output directory for low quality (downsampled) data')
    parser.add_argument('-opt', type=str, 
                       default='options/test/HAT_ERN5_SRx8.yml',
                       help='Path to the test configuration file')
    
    args = parser.parse_args()
    
    # Step 1: Prepare test data if requested
    if args.prepare_data:
        print("Preparing test data...")
        prepare_test_data(args.input_file, args.output_dir_gt, args.output_dir_lq)
    
    # Step 2: Debug dataset registration
    print("\n======= Debugging Dataset Registration =======")
    print("Available registered datasets:")
    for name in sorted(DATASET_REGISTRY.keys()):
        print(f"  - {name}")
    
    # Print the modules that are available in the Python path
    print("\nPython module search paths:")
    for path in sys.path:
        print(f"  - {path}")
    
    # Verify that ERN5EvalDataset is properly registered
    if 'ERN5EvalDataset' in DATASET_REGISTRY.__module__:
        print("\nERN5EvalDataset is successfully registered!")
        print(f"Class: {DATASET_REGISTRY.__module__['ERN5EvalDataset']}")
    else:
        print("\nWARNING: ERN5EvalDataset is NOT registered!")
        print("This may be why the test is failing.")
        
        # Try to explicitly register the dataset class
        try:
            print("\nAttempting to manually import ERN5EvalDataset...")
            from hat.data.ern5_eval_dataset import ERN5EvalDataset
            print("Import successful. Class:", ERN5EvalDataset)
        except Exception as e:
            print(f"Error importing ERN5EvalDataset: {e}")
            
    print("==============================================\n")
    
    # Step 3: Run the model test
    print("Running model test...")
    # Explicitly set the sys.argv to match what the parse_options function in test_pipeline expects
    sys.argv = [sys.argv[0], '-opt', args.opt]
    
    # Now run the test pipeline with the updated sys.argv
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)

if __name__ == '__main__':
    main() 