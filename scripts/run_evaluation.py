#!/usr/bin/env python3
"""
SemEval 2025 Task 11: Run Comprehensive Evaluation
=================================================

This script runs comprehensive evaluation comparing baseline and transformer models.

Usage:
    python scripts/run_evaluation.py

Author: SemEval 2025 Task 11 Team
Date: 2025
"""

import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

print(f"🔧 Project root: {project_root}")
print(f"🔧 Source path: {src_path}")

# Import evaluation modules
try:
    from evaluation.evaluator import run_comprehensive_evaluation

    print("✅ Evaluation module imported")
except ImportError as e:
    print(f"❌ Failed to import evaluation: {e}")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive evaluation for SemEval 2025 Task 11'
    )

    parser.add_argument(
        '--baseline-results',
        type=str,
        help='Path to baseline results pickle file'
    )

    parser.add_argument(
        '--transformer-results',
        type=str,
        help='Path to transformer results pickle file'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='./results',
        help='Directory containing result files (default: ./results)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/evaluation_results',
        help='Output directory for evaluation results (default: ./results/evaluation_results)'
    )

    return parser.parse_args()


def find_latest_results_files(results_dir):
    """
    Find the latest baseline and transformer results files

    Args:
        results_dir (str): Directory to search for results

    Returns:
        tuple: (baseline_file, transformer_file)
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        return None, None

    # Find latest baseline results
    baseline_files = list(results_path.glob("baselines_*.pkl"))
    baseline_file = max(baseline_files, key=lambda f: f.stat().st_mtime) if baseline_files else None

    # Find latest transformer results
    transformer_files = list(results_path.glob("transformers_*.pkl"))
    transformer_file = max(transformer_files, key=lambda f: f.stat().st_mtime) if transformer_files else None

    return baseline_file, transformer_file


def load_results(file_path):
    """
    Load results from pickle file

    Args:
        file_path (str or Path): Path to pickle file

    Returns:
        dict or None: Loaded results or None if failed
    """
    if file_path is None:
        return None

    file_path = Path(file_path)
    if not file_path.exists():
        print(f"⚠️  Results file not found: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Loaded results from: {file_path}")
        return results
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None


def save_evaluation_results(results, output_dir):
    """
    Save evaluation results to output directory

    Args:
        results: Evaluation results to save
        output_dir (str): Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pickle_path = output_path / f"evaluation_results_{timestamp}.pkl"

    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"✅ Evaluation results saved to: {pickle_path}")
    except Exception as e:
        print(f"⚠️  Could not save evaluation results: {e}")


def main():
    """Main evaluation execution function"""
    args = parse_arguments()

    print("📊 SemEval 2025 Task 11: Comprehensive Model Evaluation")
    print("=" * 60)
    print("🌍 Languages: English, German, Portuguese Brazilian")
    print("🎯 Tasks: Multi-label Classification + Intensity Prediction")

    # Determine baseline and transformer results files
    if args.baseline_results and args.transformer_results:
        # Use specified files
        baseline_file = Path(args.baseline_results)
        transformer_file = Path(args.transformer_results)
        print(f"\n📂 Using specified results files:")
        print(f"   Baseline: {baseline_file}")
        print(f"   Transformer: {transformer_file}")
    else:
        # Auto-find latest files
        print(f"\n🔍 Searching for latest results in: {args.results_dir}")
        baseline_file, transformer_file = find_latest_results_files(args.results_dir)

        if baseline_file:
            print(f"   📄 Found baseline results: {baseline_file}")
        else:
            print(f"   ⚠️  No baseline results found")

        if transformer_file:
            print(f"   📄 Found transformer results: {transformer_file}")
        else:
            print(f"   ⚠️  No transformer results found")

    # Load results
    print(f"\n📥 Loading results...")
    baseline_results = load_results(baseline_file)
    transformer_results = load_results(transformer_file)

    # Check if we have any results to evaluate
    if baseline_results is None and transformer_results is None:
        print("❌ No results files found or loaded successfully!")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure you have run the training pipeline first")
        print("   2. Check that results directory exists and contains *.pkl files")
        print("   3. Specify file paths manually with --baseline-results and --transformer-results")
        print(f"\n📁 Looking in: {Path(args.results_dir).absolute()}")
        return 1

    # Show what we have
    print(f"\n📊 Available Results:")
    if baseline_results:
        print(f"   ✅ Baseline results loaded")
        if isinstance(baseline_results, dict):
            available_tracks = [k for k, v in baseline_results.items() if v is not None]
            print(f"      Available tracks: {available_tracks}")
    else:
        print(f"   ❌ No baseline results")

    if transformer_results:
        print(f"   ✅ Transformer results loaded")
        if isinstance(transformer_results, dict):
            available_tracks = [k for k, v in transformer_results.items() if v is not None]
            print(f"      Available tracks: {available_tracks}")
    else:
        print(f"   ❌ No transformer results")

    # Run comprehensive evaluation
    print(f"\n" + "=" * 50)
    print("🚀 STARTING COMPREHENSIVE EVALUATION")
    print("=" * 50)

    try:
        evaluation_results = run_comprehensive_evaluation(
            baseline_results, transformer_results
        )

        # Save evaluation results
        save_evaluation_results(evaluation_results, args.output_dir)

        print(f"\n" + "=" * 60)
        print("🎉 EVALUATION COMPLETE!")
        print("=" * 60)

        print(f"\n📊 EVALUATION SUMMARY:")
        if evaluation_results:
            # Show key results
            baseline_summary = evaluation_results.get('baseline_summary', {})
            transformer_summary = evaluation_results.get('transformer_summary', {})

            # Track A Results
            track_a_baseline = baseline_summary.get('track_a', {})
            track_a_transformer = transformer_summary.get('track_a', {})

            if track_a_baseline or track_a_transformer:
                print(f"\n🎯 TRACK A (Classification) Results:")

                if track_a_baseline:
                    best_baseline_a = max(track_a_baseline.values(), key=lambda x: x.get('f1_macro', 0))
                    print(f"   Best Baseline: F1-Macro = {best_baseline_a.get('f1_macro', 0):.4f}")

                if track_a_transformer:
                    best_transformer_a = max(track_a_transformer.values(), key=lambda x: x.get('f1_macro', 0))
                    print(f"   Best Transformer: F1-Macro = {best_transformer_a.get('f1_macro', 0):.4f}")

                if track_a_baseline and track_a_transformer:
                    baseline_score = best_baseline_a.get('f1_macro', 0)
                    transformer_score = best_transformer_a.get('f1_macro', 0)
                    if baseline_score > 0:
                        improvement = ((transformer_score - baseline_score) / baseline_score) * 100
                        print(f"   🚀 Improvement: {improvement:+.1f}%")

            # Track B Results
            track_b_baseline = baseline_summary.get('track_b', {})
            track_b_transformer = transformer_summary.get('track_b', {})

            if track_b_baseline or track_b_transformer:
                print(f"\n📊 TRACK B (Intensity) Results:")

                if track_b_baseline:
                    best_baseline_b = max(track_b_baseline.values(), key=lambda x: x.get('pearson_avg', 0))
                    print(f"   Best Baseline: Pearson = {best_baseline_b.get('pearson_avg', 0):.4f}")

                if track_b_transformer:
                    best_transformer_b = max(track_b_transformer.values(), key=lambda x: x.get('pearson_avg', 0))
                    print(f"   Best Transformer: Pearson = {best_transformer_b.get('pearson_avg', 0):.4f}")

                if track_b_baseline and track_b_transformer:
                    baseline_score = best_baseline_b.get('pearson_avg', 0)
                    transformer_score = best_transformer_b.get('pearson_avg', 0)
                    if baseline_score > 0:
                        improvement = ((transformer_score - baseline_score) / baseline_score) * 100
                        print(f"   📈 Improvement: {improvement:+.1f}%")

        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Review detailed evaluation results in: {args.output_dir}")
        print(f"   2. Check generated visualizations and performance comparisons")
        print(f"   3. Use insights for model improvements and hyperparameter tuning")
        print(f"   4. Prepare final submission for SemEval 2025 Task 11")

        return 0

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    print(f"\nEvaluation finished with exit code: {exit_code}")
    sys.exit(exit_code)