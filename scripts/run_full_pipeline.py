#!/usr/bin/env python3
"""
SemEval 2025 Task 11: Complete Pipeline Runner
=============================================

This script runs the entire workflow from data exploration to evaluation
for multilingual emotion detection across English, German, and Portuguese Brazilian.

Usage:
    python scripts/run_full_pipeline.py

Author: SemEval 2025 Task 11 Team
Date: 2025
"""

import sys
import os
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

print(f"🔧 Project root: {project_root}")
print(f"🔧 Source path: {src_path}")
print(f"🔧 Python path includes: {src_path}")

# Test imports first
print("\n🔍 Testing module imports...")
try:
    from data.data_exploration import run_data_exploration

    print("✅ Data exploration module imported")
except ImportError as e:
    print(f"❌ Failed to import data exploration: {e}")
    sys.exit(1)

try:
    from data.preprocessing import run_preprocessing_pipeline

    print("✅ Preprocessing module imported")
except ImportError as e:
    print(f"❌ Failed to import preprocessing: {e}")
    sys.exit(1)

try:
    from models.baseline_models import run_all_baseline_experiments

    print("✅ Baseline models module imported")
except ImportError as e:
    print(f"❌ Failed to import baseline models: {e}")
    sys.exit(1)

try:
    from models.transformer_models import run_transformer_pipeline

    print("✅ Transformer models module imported")
except ImportError as e:
    print(f"❌ Failed to import transformer models: {e}")
    sys.exit(1)

try:
    from evaluation.evaluator import run_comprehensive_evaluation

    print("✅ Evaluation module imported")
except ImportError as e:
    print(f"❌ Failed to import evaluation: {e}")
    sys.exit(1)


def setup_logging():
    """Setup basic logging for the pipeline"""
    import logging

    # Create logs directory
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = logs_dir / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run SemEval 2025 Task 11 complete pipeline'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='dataset',
        help='Path to dataset directory (default: dataset)'
    )

    parser.add_argument(
        '--skip-exploration',
        action='store_true',
        help='Skip data exploration step'
    )

    parser.add_argument(
        '--skip-baselines',
        action='store_true',
        help='Skip baseline model training'
    )

    parser.add_argument(
        '--skip-transformers',
        action='store_true',
        help='Skip transformer model training'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )

    return parser.parse_args()


def save_results(results, output_dir, stage_name):
    """
    Save results to output directory

    Args:
        results: Results to save
        output_dir (str): Output directory path
        stage_name (str): Name of the pipeline stage
    """
    import pickle
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle for complete objects
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pickle_path = output_path / f"{stage_name}_{timestamp}.pkl"

    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ {stage_name} results saved to: {pickle_path}")
    except Exception as e:
        print(f"⚠️  Could not save {stage_name} results: {e}")


def check_dataset_exists_and_has_data(data_path):
    """
    Check if dataset exists and contains actual data files
    """
    import pandas as pd

    # Resolve path relative to project root
    project_root = Path(__file__).parent.parent
    if Path(data_path).is_absolute():
        dataset_path = Path(data_path)
    else:
        dataset_path = project_root / data_path

    print(f"🔍 Looking for dataset at: {dataset_path.absolute()}")

    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return False, str(dataset_path)

    # Check for track folders and data files
    tracks = ['track_a', 'track_b', 'track_c']
    found_tracks = []
    data_files_found = False

    for track in tracks:
        track_path = dataset_path / track
        if track_path.exists():
            found_tracks.append(track)

            # Check for data files in subfolders
            for subfolder in ['train', 'dev', 'test']:
                subfolder_path = track_path / subfolder
                if subfolder_path.exists():
                    # Look for data files
                    data_files = (list(subfolder_path.glob("*.csv")) +
                                  list(subfolder_path.glob("*.tsv")) +
                                  list(subfolder_path.glob("*.txt")) +
                                  list(subfolder_path.glob("*.json")))

                    if data_files:
                        print(f"   📄 Found {len(data_files)} files in {track}/{subfolder}")
                        data_files_found = True

                        # Test loading one file to verify format
                        for file_path in data_files[:1]:  # Test first file
                            try:
                                if file_path.suffix.lower() == '.csv':
                                    df = pd.read_csv(file_path)
                                elif file_path.suffix.lower() == '.tsv':
                                    df = pd.read_csv(file_path, sep='\t')
                                else:
                                    df = pd.read_csv(file_path)

                                print(f"      ✅ Sample file {file_path.name}: {df.shape} - Columns: {list(df.columns)}")

                            except Exception as e:
                                print(f"      ⚠️  Cannot read {file_path.name}: {e}")

    if found_tracks:
        print(f"✅ Dataset found with tracks: {found_tracks}")
        if data_files_found:
            print(f"✅ Data files found and readable")
            return True, str(dataset_path)
        else:
            print(f"⚠️  Track folders exist but no readable data files found")
            print(f"   Expected files: *.csv, *.tsv, *.txt, *.json in train/dev/test subfolders")
            return False, str(dataset_path)
    else:
        print(f"❌ No track folders found")
        print(f"   Looking for: {tracks}")
        print(f"   In directory: {dataset_path.absolute()}")
        return False, str(dataset_path)


def main():
    """Main pipeline execution function"""
    args = parse_arguments()
    logger = setup_logging()

    print("🚀 SemEval 2025 Task 11: Multilingual Emotion Detection Pipeline")
    print("=" * 70)
    print(f"📂 Data path: {args.data_path}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"🌍 Languages: English, German, Portuguese Brazilian")
    print(f"🎯 Tasks: Multi-label Classification + Intensity Prediction")

    # Check if dataset exists and has data
    dataset_ok, resolved_data_path = check_dataset_exists_and_has_data(args.data_path)
    if not dataset_ok:
        print("\n❌ Cannot proceed without valid dataset. Please:")
        print("   1. Ensure your dataset files are in the correct location")
        print("   2. Check that track_a/, track_b/, track_c/ folders contain train/dev/test subfolders")
        print("   3. Verify that data files (*.csv, *.tsv, *.txt, *.json) exist in subfolders")
        print("   4. Use --data-path argument if dataset is elsewhere")
        print(f"\n📁 Expected structure:")
        print(f"   {resolved_data_path}/")
        print(f"   ├── track_a/train/*.csv")
        print(f"   ├── track_a/dev/*.csv")
        print(f"   ├── track_a/test/*.csv")
        print(f"   ├── track_b/train/*.csv")
        print(f"   └── ... (similar for track_b and track_c)")
        return 1

    results = {}

    try:
        # Step 1: Data Exploration and Language Filtering
        if not args.skip_exploration:
            print("\n" + "=" * 50)
            print("1️⃣ DATA EXPLORATION AND LANGUAGE FILTERING")
            print("=" * 50)

            try:
                # Use resolved path for data exploration
                exploration_results = run_data_exploration(resolved_data_path)
                results['exploration'] = exploration_results
                save_results(exploration_results, args.output_dir, 'exploration')

                # Check if exploration actually found data
                if exploration_results and exploration_results.get('unified_datasets'):
                    datasets = exploration_results['unified_datasets']
                    found_data = any(df is not None for df in datasets.values())

                    if not found_data:
                        print("\n⚠️  WARNING: Data exploration completed but no usable datasets were created.")
                        print("    This might indicate:")
                        print("    - File format issues (unsupported format)")
                        print("    - Language filtering too strict (no files match selected languages)")
                        print("    - Empty or corrupted data files")
                        print("    - Different file naming convention than expected")
                        print("\n🔧 Suggestions:")
                        print("    1. Check your data file contents and format")
                        print("    2. Verify language codes in filenames match: ['eng', 'deu', 'ptbr']")
                        print("    3. Ensure files contain actual data (not empty)")

                        # Continue anyway to see what happens

                logger.info("Data exploration completed successfully")
            except Exception as e:
                print(f"❌ Error in data exploration: {e}")
                traceback.print_exc()
                return 1
        else:
            print("\n⏭️  Skipping data exploration step")
            exploration_results = None

        # Step 2: Multilingual Data Preprocessing
        print("\n" + "=" * 50)
        print("2️⃣ MULTILINGUAL DATA PREPROCESSING")
        print("=" * 50)

        if exploration_results:
            try:
                preprocessing_results = run_preprocessing_pipeline(exploration_results)
                results['preprocessing'] = preprocessing_results
                save_results(preprocessing_results, args.output_dir, 'preprocessing')
                logger.info("Data preprocessing completed successfully")
            except Exception as e:
                print(f"❌ Error in preprocessing: {e}")
                traceback.print_exc()
                return 1
        else:
            print("❌ Cannot run preprocessing without exploration results")
            return 1

        # Step 3: Baseline Model Training
        if not args.skip_baselines:
            print("\n" + "=" * 50)
            print("3️⃣ BASELINE MODEL TRAINING")
            print("=" * 50)

            if preprocessing_results:
                try:
                    baseline_results = run_all_baseline_experiments(preprocessing_results)
                    results['baselines'] = baseline_results
                    save_results(baseline_results, args.output_dir, 'baselines')
                    logger.info("Baseline model training completed successfully")
                except Exception as e:
                    print(f"❌ Error in baseline training: {e}")
                    traceback.print_exc()
                    baseline_results = None
            else:
                print("❌ Cannot run baseline training without preprocessing results")
                baseline_results = None
        else:
            print("\n⏭️  Skipping baseline model training")
            baseline_results = None

        # Step 4: Transformer Model Training
        if not args.skip_transformers:
            print("\n" + "=" * 50)
            print("4️⃣ TRANSFORMER MODEL TRAINING")
            print("=" * 50)

            if preprocessing_results:
                try:
                    transformer_results = run_transformer_pipeline(preprocessing_results)
                    results['transformers'] = transformer_results
                    save_results(transformer_results, args.output_dir, 'transformers')
                    logger.info("Transformer model training completed successfully")
                except Exception as e:
                    print(f"❌ Error in transformer training: {e}")
                    traceback.print_exc()
                    transformer_results = None
            else:
                print("❌ Cannot run transformer training without preprocessing results")
                transformer_results = None
        else:
            print("\n⏭️  Skipping transformer model training")
            transformer_results = None

        # Step 5: Comprehensive Evaluation
        print("\n" + "=" * 50)
        print("5️⃣ COMPREHENSIVE EVALUATION")
        print("=" * 50)

        if baseline_results or transformer_results:
            try:
                evaluation_results = run_comprehensive_evaluation(
                    baseline_results, transformer_results
                )
                results['evaluation'] = evaluation_results
                save_results(evaluation_results, args.output_dir, 'evaluation')
                logger.info("Comprehensive evaluation completed successfully")
            except Exception as e:
                print(f"❌ Error in evaluation: {e}")
                traceback.print_exc()
                evaluation_results = None
        else:
            print("❌ Cannot run evaluation without model results")
            evaluation_results = None

        # Final Summary
        print("\n" + "=" * 70)
        print("🎉 PIPELINE EXECUTION COMPLETE!")
        print("=" * 70)

        print(f"\n📊 PIPELINE SUMMARY:")
        stages_completed = []
        if 'exploration' in results: stages_completed.append("Data Exploration")
        if 'preprocessing' in results: stages_completed.append("Preprocessing")
        if 'baselines' in results: stages_completed.append("Baseline Models")
        if 'transformers' in results: stages_completed.append("Transformer Models")
        if 'evaluation' in results: stages_completed.append("Evaluation")

        print(f"   ✅ Completed stages: {', '.join(stages_completed)}")
        print(f"   📂 Results saved to: {args.output_dir}")

        # Performance Summary
        if evaluation_results:
            print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
            print(f"   Check the generated evaluation report and visualizations")
            print(f"   in {args.output_dir}/evaluation_results/")
        else:
            print(f"\n⚠️  NO MODEL RESULTS:")
            print(f"   The pipeline completed but no models were trained successfully.")
            print(f"   This indicates data loading or preprocessing issues.")
            print(f"   Please check the error messages above and verify your dataset format.")

        print(f"\n🎯 NEXT STEPS:")
        if evaluation_results:
            print(f"   1. Review the generated evaluation report")
            print(f"   2. Analyze model performance visualizations")
            print(f"   3. Consider hyperparameter tuning for better results")
            print(f"   4. Prepare submission for SemEval 2025 Task 11")
        else:
            print(f"   1. Fix dataset format and file structure issues")
            print(f"   2. Verify file naming conventions match expected patterns")
            print(f"   3. Check that data files contain valid emotion detection data")
            print(f"   4. Re-run pipeline after fixing data issues")

        logger.info("Pipeline completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\nPipeline finished with exit code: {exit_code}")
    sys.exit(exit_code)