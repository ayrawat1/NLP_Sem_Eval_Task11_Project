#!/usr/bin/env python3
"""
Run preprocessing pipeline
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.preprocessing import run_preprocessing_pipeline
from data.data_exploration import run_data_exploration

if __name__ == "__main__":
    exploration_results = run_data_exploration()
    preprocessing_results = run_preprocessing_pipeline(exploration_results)
    print("✅ Preprocessing complete!")