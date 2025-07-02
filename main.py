#!/usr/bin/env python3
"""Main script to train, finalize, and score hurdle model."""

import argparse
import sys
from datetime import datetime

# Import required modules
from src.models.train_hurdle_model import main as train_hurdle_model
from src.models.finalize_model import finalize_model
from src.models.score import score_data

def main():
    """Main script to run hurdle model pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run Hurdle Model Pipeline")
    parser.add_argument("--train-end-year", type=int, default=2022,
                        help="Last year to include in training data (exclusive)")
    parser.add_argument("--valid-window", type=int, default=2,
                        help="Number of years to use for validation")
    parser.add_argument("--test-window", type=int, default=2,
                        help="Number of years to use for testing")
    parser.add_argument("--experiment-name", type=str, 
                        default=None,
                        help="Custom experiment name (optional)")
    
    args = parser.parse_args()

    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"hurdle_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Calculate year ranges
    tune_start_year = args.train_end_year
    tune_end_year = args.train_end_year + args.valid_window - 1
    test_start_year = tune_end_year + 1
    test_end_year = test_start_year + args.test_window - 1

    # Print year ranges
    print("\nYear Ranges:")
    print(f"Training:    through {args.train_end_year-1} (exclusive {args.train_end_year})")
    print(f"Validation:  {tune_start_year}-{tune_end_year}")
    print(f"Testing:     {test_start_year}-{test_end_year}\n")

    # Modify sys.argv to pass parameters to training script
    original_argv = sys.argv[:]
    sys.argv = [
        sys.argv[0],
        "--train-end-year", str(args.train_end_year),
        "--tune-start-year", str(tune_start_year),
        "--tune-end-year", str(tune_end_year),
        "--test-start-year", str(test_start_year),
        "--test-end-year", str(test_end_year),
        "--experiment-name", args.experiment_name
    ]
    
    # Train hurdle model
    print("Training Hurdle Model...")
    train_hurdle_model()
    
    # Finalize model
    print("Finalizing Hurdle Model...")
    finalize_model(
        model_type="hurdle",
        experiment_name=args.experiment_name,
        end_year=args.train_end_year
    )
    
    # # Score model
    # print("Scoring Hurdle Model...")
    # score_data(
    #     experiment_name=args.experiment_name,
    #     start_year=args.train_end_year + args.valid_window,
    #     end_year=datetime.now().year + 5  # Use current year + 5 to capture future games
    # )
    
    print("Hurdle Model Pipeline Complete!")

if __name__ == "__main__":
    main()
