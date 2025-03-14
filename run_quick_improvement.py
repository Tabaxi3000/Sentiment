#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick Improvement Pipeline Script

This script runs all the necessary steps to improve the sentiment analysis models
within a 12-hour timeframe. It combines datasets, optimizes models, and creates
an improved ensemble.

Author: Sentiment Analysis Team
"""

import os
import subprocess
import time
import argparse
from datetime import datetime, timedelta

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command, description):
    """Run a shell command and print its output"""
    print_section(description)
    print(f"Running command: {command}\n")
    
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
        print(line, end="")
    
    process.wait()
    end_time = time.time()
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds")
    print(f"Exit code: {process.returncode}")
    
    return process.returncode

def main():
    """Main function to run the improvement pipeline"""
    parser = argparse.ArgumentParser(description="Run the quick improvement pipeline")
    parser.add_argument("--skip-combine", action="store_true", help="Skip dataset combination step")
    parser.add_argument("--skip-optimize", action="store_true", help="Skip model optimization step")
    parser.add_argument("--skip-streamlit", action="store_true", help="Skip Streamlit app creation")
    args = parser.parse_args()
    
    # Record start time
    start_time = datetime.now()
    print(f"Starting improvement pipeline at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated completion time: {(start_time + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models/optimized", exist_ok=True)
    os.makedirs("results/optimized", exist_ok=True)
    
    # Step 1: Combine datasets
    if not args.skip_combine:
        print_section("STEP 1: COMBINING DATASETS")
        run_command("python combine_datasets.py", "Combining datasets")
    else:
        print_section("STEP 1: COMBINING DATASETS (SKIPPED)")
    
    # Step 2: Optimize models
    if not args.skip_optimize:
        print_section("STEP 2: OPTIMIZING MODELS")
        run_command("python cpu_hyperparameter_tuning.py", "Optimizing models")
    else:
        print_section("STEP 2: OPTIMIZING MODELS (SKIPPED)")
    
    # Step 3: Create improved Streamlit app
    if not args.skip_streamlit:
        print_section("STEP 3: CREATING IMPROVED STREAMLIT APP")
        print("The improved Streamlit app has been created at improved_streamlit_app.py")
    else:
        print_section("STEP 3: CREATING IMPROVED STREAMLIT APP (SKIPPED)")
    
    # Record end time
    end_time = datetime.now()
    total_time = end_time - start_time
    
    print_section("PIPELINE COMPLETED")
    print(f"Started at:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time:  {total_time}")
    
    # Print next steps
    print_section("NEXT STEPS")
    print("1. Run the improved Streamlit app:")
    print("   source sentiment_env/bin/activate && streamlit run improved_streamlit_app.py")
    print()
    print("2. Commit changes to GitHub:")
    print("   git add .")
    print("   git commit -m \"Improved models and added enhanced Streamlit app\"")
    print("   git push origin master")
    print()
    print("3. Share the improved app with others:")
    print("   - Send them the GitHub repository URL")
    print("   - They can clone the repository and run the Streamlit app")
    print()
    print("4. Further improvements (if time permits):")
    print("   - Add more datasets")
    print("   - Implement more advanced preprocessing techniques")
    print("   - Try more sophisticated ensemble methods")

if __name__ == "__main__":
    main() 