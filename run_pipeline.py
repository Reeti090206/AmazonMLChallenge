"""
run_pipeline.py — Entry-point wrapper that adds the src directory to
sys.path and then runs the full submission pipeline.

Usage from the project root:
    python run_pipeline.py --dataset-dir dataset --output-dir output --model-dir models [options]
"""
import sys
import os

# Add the src directory to the Python path
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "code", "business_entity_resolution", "src")
sys.path.insert(0, src_dir)

from submission import main

if __name__ == "__main__":
    main()
