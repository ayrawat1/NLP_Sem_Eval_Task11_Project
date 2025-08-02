#!/usr/bin/env python3
"""
Run baseline model training
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.baseline_models import run_all_baseline_experiments
from data.data_exploration import run_data_exploration
from data.preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    exploration_results = run_data_exploration()
    preprocessing_results = run_preprocessing_pipeline(exploration_results)
    baseline_results = run_all_baseline_experiments(preprocessing_results)
    print("✅ Baseline training complete!")