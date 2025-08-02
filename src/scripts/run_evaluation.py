#!/usr/bin/env python3
"""
Run comprehensive evaluation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from evaluation.evaluator import run_comprehensive_evaluation
from models.baseline_models import run_all_baseline_experiments
from models.transformer_models import run_transformer_pipeline
from data.data_exploration import run_data_exploration
from data.preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    exploration_results = run_data_exploration()
    preprocessing_results = run_preprocessing_pipeline(exploration_results)
    baseline_results = run_all_baseline_experiments(preprocessing_results)
    transformer_results = run_transformer_pipeline(preprocessing_results)
    evaluation_results = run_comprehensive_evaluation(baseline_results, transformer_results)
    print("✅ Evaluation complete!")