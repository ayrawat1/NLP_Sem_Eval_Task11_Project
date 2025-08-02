#!/usr/bin/env python3
"""
Run data exploration pipeline
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_exploration import run_data_exploration

if __name__ == "__main__":
    results = run_data_exploration()
    print("✅ Data exploration complete!")