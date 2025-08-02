#!/usr/bin/env python3
"""
Run transformer model training
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.transformer_models import run_transformer_pipeline
from data.data_exploration import run_data_exploration
from data.preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    exploration_results = run_data_exploration()
    preprocessing_results = run_preprocessing_pipeline(exploration_results)
    transformer_results = run_transformer_pipeline(preprocessing_results)
    print("✅ Transformer training complete!")