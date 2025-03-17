#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:10:38 2024

@author: yanggao
"""

import os
import sys
import torch
import numpy as np
import argparse
import hat.data
from os import path as osp

# Add the BasicSR path to the Python path
basicsr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'myenv', 'lib', 'python3.11', 'site-packages', 'basicsr'))
if basicsr_path not in sys.path:
    sys.path.append(basicsr_path)

from basicsr.test import test_pipeline
from basicsr.utils.options import parse_options

def main():
    """Main function for HAT model evaluation on climate data."""
    parser = argparse.ArgumentParser(description='Evaluate HAT model on climate data')
    parser.add_argument('--opt', type=str, default='options/test/HAT_ERN5_SRx8.yml', 
                        help='Path to the test configuration file')
    parser.add_argument('--model_path', type=str, 
                        help='Path to the pretrained model')
    parser.add_argument('--data_path', type=str, 
                        help='Path to the climate data directory')
    parser.add_argument('--scale', type=int, default=8,
                        help='Super-resolution scale factor')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--save_results', action='store_true',
                        help='Save the super-resolution results')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['PYTHONPATH'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Get the root path
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Save original argv
    original_argv = sys.argv.copy()
    
    # Modify argv to match BasicSR's expected format
    sys.argv = [sys.argv[0]]
    
    # '-opt' is required by BasicSR
    sys.argv.extend(['-opt', args.opt])
    
    # Add other arguments as needed
    if args.model_path:
        sys.argv.extend(['--model_path', args.model_path])
    if args.data_path:
        # Update the data path in the configuration
        sys.argv.extend(['--force_yml', f'datasets.test_1.dataroot_gt={args.data_path}'])
    if args.scale:
        sys.argv.extend(['--force_yml', f'scale={args.scale}'])
    if args.batch_size:
        sys.argv.extend(['--force_yml', f'datasets.test_1.batch_size={args.batch_size}'])
    if args.num_gpu:
        sys.argv.extend(['--force_yml', f'num_gpu={args.num_gpu}'])
    if args.save_results:
        sys.argv.extend(['--force_yml', 'val.save_img=true'])
        
    print("Running with arguments:", sys.argv)
    
    # Run the test pipeline
    test_pipeline(root_path)
    
    # Restore original argv
    sys.argv = original_argv
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main() 