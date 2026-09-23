import argparse

from ai.preprocessing.preprocess_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the preprocessing pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum videos per class.")
    run_pipeline(limit=parser.parse_args().limit)