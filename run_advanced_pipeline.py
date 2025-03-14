#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Sentiment Analysis Pipeline

This script runs the entire advanced pipeline for sentiment analysis:
1. Data augmentation
2. Advanced hyperparameter tuning
3. Ensemble methods

Author: Sentiment Analysis Team
"""

import os
import time
import subprocess
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"Completed: {description}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Exit code: {process.returncode}")
    print(f"{'='*80}\n")
    
    return process.returncode

def create_experiment_directory():
    """Create a directory for the experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/experiment_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{experiment_dir}/augmentation", exist_ok=True)
    os.makedirs(f"{experiment_dir}/optimization", exist_ok=True)
    os.makedirs(f"{experiment_dir}/ensemble", exist_ok=True)
    
    return experiment_dir

def run_pipeline(args):
    """Run the entire pipeline"""
    print("\n" + "="*80)
    print("Starting Advanced Sentiment Analysis Pipeline")
    print("="*80 + "\n")
    
    # Record start time
    pipeline_start_time = time.time()
    
    # Create experiment directory
    experiment_dir = create_experiment_directory()
    print(f"Experiment results will be saved to: {experiment_dir}")
    
    # Step 1: Data Augmentation
    if not args.skip_augmentation:
        augmentation_exit_code = run_command(
            "python data_augmentation.py",
            "Data Augmentation"
        )
        
        if augmentation_exit_code != 0:
            print("Data augmentation failed. Exiting pipeline.")
            return
        
        # Copy augmentation results to experiment directory
        run_command(
            f"cp data/augmented_data.csv {experiment_dir}/augmentation/",
            "Copying augmentation results"
        )
    else:
        print("Skipping data augmentation step.")
    
    # Step 2: Advanced Hyperparameter Tuning
    if not args.skip_tuning:
        tuning_exit_code = run_command(
            "python advanced_hyperparameter_tuning.py",
            "Advanced Hyperparameter Tuning"
        )
        
        if tuning_exit_code != 0:
            print("Hyperparameter tuning failed. Exiting pipeline.")
            return
        
        # Copy optimization results to experiment directory
        run_command(
            f"cp -r results/optimization/* {experiment_dir}/optimization/",
            "Copying optimization results"
        )
        
        # Copy optimized models to experiment directory
        run_command(
            f"mkdir -p {experiment_dir}/models/optimized",
            "Creating models directory"
        )
        
        run_command(
            f"cp -r models/optimized/* {experiment_dir}/models/optimized/",
            "Copying optimized models"
        )
    else:
        print("Skipping hyperparameter tuning step.")
    
    # Step 3: Ensemble Methods
    if not args.skip_ensemble:
        ensemble_exit_code = run_command(
            "python ensemble_models.py",
            "Ensemble Methods"
        )
        
        if ensemble_exit_code != 0:
            print("Ensemble methods failed. Exiting pipeline.")
            return
        
        # Copy ensemble results to experiment directory
        run_command(
            f"cp -r results/ensemble/* {experiment_dir}/ensemble/",
            "Copying ensemble results"
        )
        
        # Copy ensemble models to experiment directory
        run_command(
            f"mkdir -p {experiment_dir}/models/ensemble",
            "Creating ensemble models directory"
        )
        
        run_command(
            f"cp -r models/ensemble/* {experiment_dir}/models/ensemble/",
            "Copying ensemble models"
        )
    else:
        print("Skipping ensemble methods step.")
    
    # Record end time
    pipeline_end_time = time.time()
    total_time = pipeline_end_time - pipeline_start_time
    
    # Generate summary report
    summary = f"""
Advanced Sentiment Analysis Pipeline Summary
===========================================

Experiment Directory: {experiment_dir}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)

Steps Completed:
- Data Augmentation: {"Skipped" if args.skip_augmentation else "Completed"}
- Hyperparameter Tuning: {"Skipped" if args.skip_tuning else "Completed"}
- Ensemble Methods: {"Skipped" if args.skip_ensemble else "Completed"}

Results:
- Augmented data: {experiment_dir}/augmentation/augmented_data.csv
- Optimization results: {experiment_dir}/optimization/
- Ensemble results: {experiment_dir}/ensemble/
- Optimized models: {experiment_dir}/models/optimized/
- Ensemble models: {experiment_dir}/models/ensemble/

Next Steps:
1. Review the results in the experiment directory
2. Use the optimized models for predictions
3. Deploy the best model for production use
"""
    
    # Save summary to file
    with open(f"{experiment_dir}/summary.txt", "w") as f:
        f.write(summary)
    
    print("\n" + "="*80)
    print("Advanced Sentiment Analysis Pipeline Completed")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {experiment_dir}")
    print("="*80 + "\n")
    
    # Print summary
    print(summary)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run the advanced sentiment analysis pipeline"
    )
    
    parser.add_argument(
        "--skip-augmentation",
        action="store_true",
        help="Skip the data augmentation step"
    )
    
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip the hyperparameter tuning step"
    )
    
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Skip the ensemble methods step"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(args) 