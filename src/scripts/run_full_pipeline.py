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
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import pipeline modules
from data.data_exploration import run_data_exploration
from data.preprocessing import run_preprocessing_pipeline
from models.baseline_models import run_all_baseline_experiments
from models.transformer_models import run_transformer_pipeline
from evaluation.evaluator import run_comprehensive_evaluation


def setup_logging():
    """Setup basic logging for the pipeline"""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
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
        default='./data/raw/dataset',
        help='Path to dataset directory (default: ./data/raw/dataset)'
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

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional)'
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
    import json
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle for complete objects
    pickle_path = output_path / f"{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ {stage_name} results saved to: {pickle_path}")

    # Save summary as JSON (if possible)
    try:
        summary = extract_summary(results, stage_name)
        if summary:
            json_path = output_path / f"{stage_name}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"✅ {stage_name} summary saved to: {json_path}")
    except Exception as e:
        print(f"⚠️  Could not save JSON summary for {stage_name}: {e}")


def extract_summary(results, stage_name):
    """Extract JSON-serializable summary from results"""
    if stage_name == "exploration":
        return {
            "languages": results.get('languages', []),
            "emotions": results.get('emotions', []),
            "tracks_found": list(results.get('track_info', {}).keys()),
            "datasets_created": {
                k: v is not None for k, v in results.get('unified_datasets', {}).items()
            }
        }
    elif stage_name == "preprocessing":
        return {
            "tracks_processed": [k for k, v in results.items() if v is not None and 'processed' in k],
            "languages": results.get('languages', []),
            "emotions": results.get('emotions', [])
        }
    elif stage_name == "baselines":
        summary = {}
        for track, result in results.items():
            if result and 'classification_results' in result:
                best_f1 = max(
                    [v.get('f1_macro', 0) for v in result['classification_results'].values() if 'error' not in v],
                    default=0)
                summary[f"{track}_best_f1"] = best_f1
            if result and 'regression_results' in result:
                best_pearson = max(
                    [v.get('pearson_avg', 0) for v in result['regression_results'].values() if 'error' not in v],
                    default=0)
                summary[f"{track}_best_pearson"] = best_pearson
        return summary
    elif stage_name == "transformers":
        summary = {}
        for track, result in results.items():
            if result and 'history' in result:
                history = result['history']
                if history.get('val_f1_macro'):
                    summary[f"{track}_best_f1"] = max(history['val_f1_macro'])
                if history.get('val_pearson'):
                    summary[f"{track}_best_pearson"] = max(history['val_pearson'])
        return summary
    return None


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

    results = {}

    try:
        # Step 1: Data Exploration and Language Filtering
        if not args.skip_exploration:
            print("\n" + "=" * 50)
            print("1️⃣ DATA EXPLORATION AND LANGUAGE FILTERING")
            print("=" * 50)

            exploration_results = run_data_exploration(args.data_path)
            results['exploration'] = exploration_results
            save_results(exploration_results, args.output_dir, 'exploration')

            logger.info("Data exploration completed successfully")
        else:
            print("\n⏭️  Skipping data exploration step")
            exploration_results = None

        # Step 2: Multilingual Data Preprocessing
        print("\n" + "=" * 50)
        print("2️⃣ MULTILINGUAL DATA PREPROCESSING")
        print("=" * 50)

        if exploration_results:
            preprocessing_results = run_preprocessing_pipeline(exploration_results)
            results['preprocessing'] = preprocessing_results
            save_results(preprocessing_results, args.output_dir, 'preprocessing')

            logger.info("Data preprocessing completed successfully")
        else:
            print("❌ Cannot run preprocessing without exploration results")
            preprocessing_results = None

        # Step 3: Baseline Model Training
        if not args.skip_baselines:
            print("\n" + "=" * 50)
            print("3️⃣ BASELINE MODEL TRAINING")
            print("=" * 50)

            if preprocessing_results:
                baseline_results = run_all_baseline_experiments(preprocessing_results)
                results['baselines'] = baseline_results
                save_results(baseline_results, args.output_dir, 'baselines')

                logger.info("Baseline model training completed successfully")
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
                transformer_results = run_transformer_pipeline(preprocessing_results)
                results['transformers'] = transformer_results
                save_results(transformer_results, args.output_dir, 'transformers')

                logger.info("Transformer model training completed successfully")
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
            evaluation_results = run_comprehensive_evaluation(
                baseline_results, transformer_results
            )
            results['evaluation'] = evaluation_results
            save_results(evaluation_results, args.output_dir, 'evaluation')

            logger.info("Comprehensive evaluation completed successfully")
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
            print(f"\n🏆 BEST PERFORMANCE:")
            baseline_summary = evaluation_results.get('baseline_summary', {})
            transformer_summary = evaluation_results.get('transformer_summary', {})

            # Track A
            track_a_scores = []
            if baseline_summary.get('track_a'):
                track_a_scores.extend([v['f1_macro'] for v in baseline_summary['track_a'].values()])
            if transformer_summary.get('track_a'):
                track_a_scores.extend([v['f1_macro'] for v in transformer_summary['track_a'].values()])

            if track_a_scores:
                best_a = max(track_a_scores)
                print(f"   🎯 Track A (Classification): F1-Macro = {best_a:.4f}")

            # Track B
            track_b_scores = []
            if baseline_summary.get('track_b'):
                track_b_scores.extend([v['pearson_avg'] for v in baseline_summary['track_b'].values()])
            if transformer_summary.get('track_b'):
                track_b_scores.extend([v['pearson_avg'] for v in transformer_summary['track_b'].values()])

            if track_b_scores:
                best_b = max(track_b_scores)
                print(f"   📊 Track B (Intensity): Pearson = {best_b:.4f}")

        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Analyze the generated evaluation report")
        print(f"   2. Review model performance visualizations")
        print(f"   3. Consider additional experiments or improvements")
        print(f"   4. Prepare submission for SemEval 2025 Task 11")

        logger.info("Pipeline completed successfully")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)