"""
SemEval 2025 Task 11: Data Exploration and Language Filtering
============================================================

This module handles the initial data exploration, dataset structure analysis,
and multilingual filtering for English, German, and Portuguese Brazilian.

Author: SemEval 2025 Task 11 Team
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Configuration
SELECTED_LANGUAGES = ['eng', 'deu', 'ptbr']  # English, German, Portuguese Brazilian
COMMON_EMOTIONS = ['joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']


def explore_dataset_structure(base_path="./dataset"):
    """
    Explore the dataset structure and identify available tracks

    Args:
        base_path (str): Path to dataset directory

    Returns:
        dict: Information about available tracks and files
    """
    print("🚀 SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection")
    print("=" * 70)
    print("\n📁 Dataset Structure Exploration")
    print("-" * 30)

    base_path = Path(base_path)
    tracks = ["track_a", "track_b", "track_c"]
    track_info = {}

    for track in tracks:
        track_path = base_path / track
        if track_path.exists():
            print(f"✅ {track} folder found")
            subfolders = [f.name for f in track_path.iterdir() if f.is_dir()]
            files = [f.name for f in track_path.iterdir() if f.is_file()]
            print(f"   Subfolders: {subfolders}")
            if files:
                print(f"   Files: {files}")

            track_info[track] = {
                'path': track_path,
                'subfolders': subfolders,
                'files': files,
                'exists': True
            }
        else:
            print(f"❌ {track} folder not found")
            track_info[track] = {'exists': False}

    print("\n" + "=" * 70)
    return track_info


def load_track_data_nested(track_name, base_path="./dataset"):
    """
    Load data from track with nested folder structure (dev/test/train folders)

    Args:
        track_name (str): Name of the track (e.g., 'track_a')
        base_path (str): Base path to dataset

    Returns:
        dict: Dictionary with loaded dataframes
    """
    track_path = Path(base_path) / track_name

    if not track_path.exists():
        print(f"❌ {track_name} folder not found!")
        return None

    print(f"\n📊 Loading {track_name.upper()} Data (Nested Structure)")
    print("-" * 40)

    data_dict = {}

    # Look for subfolders: dev, test, train
    for subfolder_name in ['train', 'dev', 'test']:
        subfolder_path = track_path / subfolder_name

        if subfolder_path.exists() and subfolder_path.is_dir():
            print(f"\n📂 Checking {subfolder_name} folder...")

            # Find all data files in this subfolder
            data_files = (list(subfolder_path.glob("*.csv")) +
                          list(subfolder_path.glob("*.tsv")) +
                          list(subfolder_path.glob("*.txt")) +
                          list(subfolder_path.glob("*.json")))

            if data_files:
                for file_path in data_files:
                    try:
                        print(f"  📄 Found: {file_path.name}")

                        # Try different file formats
                        if file_path.suffix.lower() == '.csv':
                            df = pd.read_csv(file_path)
                        elif file_path.suffix.lower() == '.tsv':
                            df = pd.read_csv(file_path, sep='\t')
                        elif file_path.suffix.lower() == '.txt':
                            # Try tab-separated first, then comma
                            try:
                                df = pd.read_csv(file_path, sep='\t')
                            except:
                                df = pd.read_csv(file_path)
                        elif file_path.suffix.lower() == '.json':
                            df = pd.read_json(file_path, lines=True)
                        else:
                            # Default: try comma-separated
                            df = pd.read_csv(file_path)

                        file_key = f"{subfolder_name}_{file_path.stem}"
                        data_dict[file_key] = df

                        print(f"    ✅ Loaded successfully")
                        print(f"    📏 Shape: {df.shape}")
                        print(f"    📋 Columns: {list(df.columns)}")

                        # Show first few rows
                        print(f"    📄 Sample data:")
                        print(f"    {df.head(2).to_string()}")
                        print()

                    except Exception as e:
                        print(f"    ❌ Error loading {file_path.name}: {e}")
            else:
                print(f"  ⚠️  No data files found in {subfolder_name}")
        else:
            print(f"  ⚠️  {subfolder_name} folder not found")

    if not data_dict:
        print(f"❌ No data files found in {track_name}")
        return None

    return data_dict


def analyze_emotions_in_data(data_dict, track_name):
    """
    Analyze emotion distribution in a track's data

    Args:
        data_dict (dict): Dictionary of loaded dataframes
        track_name (str): Name of the track
    """
    if not data_dict:
        print(f"❌ No data available for {track_name}")
        return

    print(f"\n📈 {track_name.upper()} Emotion Analysis:")

    for file_name, df in data_dict.items():
        print(f"\n  📄 File: {file_name}")
        print(f"    📊 Shape: {df.shape}")
        print(f"    📋 Columns: {list(df.columns)}")

        # Check which emotions are present as columns
        emotion_cols = [col for col in df.columns
                        if any(emotion in col.lower() for emotion in COMMON_EMOTIONS)]
        intensity_cols = [col for col in df.columns if 'intensity' in col.lower()]

        if emotion_cols:
            print(f"    🎭 Emotion columns found: {emotion_cols}")

            # Calculate emotion distribution for binary columns
            for emotion in emotion_cols:
                if df[emotion].dtype in ['int64', 'float64', 'bool']:
                    positive_count = (df[emotion] > 0).sum()
                    total_count = len(df)
                    percentage = (positive_count / total_count) * 100
                    print(f"      {emotion}: {positive_count}/{total_count} ({percentage:.1f}%)")

        if intensity_cols:
            print(f"    📊 Intensity columns found: {intensity_cols}")

            # Show intensity distribution
            for col in intensity_cols[:3]:  # Show first 3 intensity columns
                if df[col].dtype in ['int64', 'float64']:
                    value_counts = df[col].value_counts().sort_index()
                    print(f"      {col}: {dict(value_counts)}")

        # Look for text columns
        text_cols = [col for col in df.columns
                     if any(keyword in col.lower()
                            for keyword in ['text', 'sentence', 'content', 'message'])]
        if text_cols:
            print(f"    📝 Text columns found: {text_cols}")
            for text_col in text_cols[:2]:  # Show first 2 text columns
                sample_text = str(df[text_col].iloc[0]) if len(df) > 0 else "No data"
                print(f"      {text_col} sample: {sample_text[:100]}...")


def filter_data_by_language(data_dict, languages, track_name):
    """
    Filter data to include only selected languages

    Args:
        data_dict (dict): Dictionary of loaded dataframes
        languages (list): List of language codes to keep
        track_name (str): Name of the track

    Returns:
        dict: Filtered dictionary containing only selected languages
    """
    if not data_dict:
        print(f"❌ No data available for {track_name}")
        return None

    filtered_dict = {}
    total_files = len(data_dict)
    found_languages = set()

    print(f"\n📊 Filtering {track_name}:")

    for file_name, df in data_dict.items():
        # Check if this file contains data for our selected languages
        for lang in languages:
            # The dictionary keys are like 'train_eng', 'dev_deu', 'test_ptbr'
            # So we look for the language code after the underscore
            if f"_{lang}" in file_name.lower():
                filtered_dict[file_name] = df
                found_languages.add(lang)
                print(f"   ✅ Found {lang}: {file_name} ({len(df)} samples)")
                break

    missing_languages = set(languages) - found_languages
    if missing_languages:
        print(f"   ⚠️  Missing languages: {missing_languages}")
    else:
        print(f"   🎉 All target languages found!")

    print(f"   📈 Result: {len(filtered_dict)} files from {len(found_languages)} languages")
    return filtered_dict if filtered_dict else None


def create_unified_dataset_nested(data_dict, track_name):
    """
    Create a unified dataset from multiple files in a track

    Args:
        data_dict (dict): Dictionary of loaded dataframes
        track_name (str): Name of the track

    Returns:
        pd.DataFrame: Unified dataset
    """
    if not data_dict:
        print(f"❌ No data available for {track_name}")
        return None

    # Look for training data first
    train_df = None
    dev_df = None
    test_df = None

    for file_name, df in data_dict.items():
        if 'train' in file_name.lower():
            train_df = df
        elif 'dev' in file_name.lower() or 'val' in file_name.lower():
            dev_df = df
        elif 'test' in file_name.lower():
            test_df = df

    # Report what we found
    print(f"\n📊 {track_name} Dataset Splits:")
    if train_df is not None:
        print(f"  📚 Training: {len(train_df)} samples")
    if dev_df is not None:
        print(f"  🔍 Development: {len(dev_df)} samples")
    if test_df is not None:
        print(f"  🧪 Test: {len(test_df)} samples")

    # Return the training set as the primary dataset for exploration
    if train_df is not None:
        print(f"  ✅ Using training set for main analysis")
        return train_df
    elif dev_df is not None:
        print(f"  ✅ Using development set for main analysis")
        return dev_df
    else:
        # Return the largest dataset
        largest_df = max(data_dict.values(), key=len)
        largest_name = max(data_dict.keys(), key=lambda k: len(data_dict[k]))
        print(f"  ✅ Using largest dataset ({largest_name}): {len(largest_df)} samples")
        return largest_df


def show_basic_stats(df, track_name):
    """
    Show basic statistics for a dataset

    Args:
        df (pd.DataFrame): Dataset to analyze
        track_name (str): Name of the track
    """
    if df is None:
        print(f"❌ No data available for {track_name}")
        return

    print(f"\n{track_name} Statistics:")
    print(f"  📝 Total samples: {len(df)}")
    print(f"  📋 Total columns: {len(df.columns)}")
    print(f"  📄 Column names: {list(df.columns)}")

    # Check for text column
    text_cols = [col for col in df.columns
                 if any(keyword in col.lower()
                        for keyword in ['text', 'sentence', 'content', 'message'])]
    if text_cols:
        text_col = text_cols[0]
        print(f"  📏 Text column: '{text_col}'")

        # Text length statistics
        df['text_length'] = df[text_col].astype(str).str.len()
        print(f"    Average length: {df['text_length'].mean():.1f} characters")
        print(f"    Min length: {df['text_length'].min()}")
        print(f"    Max length: {df['text_length'].max()}")

        # Show sample texts
        print(f"  📄 Sample texts:")
        for i, text in enumerate(df[text_col].head(3)):
            print(f"    {i + 1}. {str(text)[:100]}{'...' if len(str(text)) > 100 else ''}")
    else:
        print(f"  ⚠️  No obvious text column found")


def run_data_exploration(base_path="./dataset"):
    """
    Main function to run the complete data exploration pipeline

    Args:
        base_path (str): Path to dataset directory

    Returns:
        dict: Complete exploration results
    """
    print("🌍 LANGUAGE SELECTION AND FILTERING")
    print("-" * 40)
    print(f"🎯 Selected Languages: {SELECTED_LANGUAGES}")
    print(f"   - eng: English (Germanic family)")
    print(f"   - deu: German (Germanic family)")
    print(f"   - ptbr: Portuguese Brazilian (Romance family)")

    # Step 1: Explore dataset structure
    track_info = explore_dataset_structure(base_path)

    # Step 2: Load all track data
    track_a_data = load_track_data_nested("track_a", base_path) if track_info.get('track_a', {}).get('exists') else None
    track_b_data = load_track_data_nested("track_b", base_path) if track_info.get('track_b', {}).get('exists') else None
    track_c_data = load_track_data_nested("track_c", base_path) if track_info.get('track_c', {}).get('exists') else None

    print("\n" + "=" * 70)

    # Step 3: Analyze emotions
    print("\n🎭 Emotion Analysis")
    print("-" * 20)

    analyze_emotions_in_data(track_a_data, "Track A")
    analyze_emotions_in_data(track_b_data, "Track B")
    analyze_emotions_in_data(track_c_data, "Track C")

    print("\n" + "=" * 70)

    # Step 4: Language filtering
    track_a_data_filtered = filter_data_by_language(track_a_data, SELECTED_LANGUAGES, "Track A")
    track_b_data_filtered = filter_data_by_language(track_b_data, SELECTED_LANGUAGES, "Track B")
    track_c_data_filtered = filter_data_by_language(track_c_data, SELECTED_LANGUAGES, "Track C")

    # Step 5: Create unified datasets
    print(f"\n🔄 Creating Unified Datasets (Filtered)")
    print("-" * 45)

    track_a_df = create_unified_dataset_nested(track_a_data_filtered, "Track A (Filtered)")
    track_b_df = create_unified_dataset_nested(track_b_data_filtered, "Track B (Filtered)")
    track_c_df = create_unified_dataset_nested(track_c_data_filtered, "Track C (Filtered)")

    # Step 6: Show statistics
    print(f"\n📊 Final Dataset Statistics (3 Languages)")
    print("-" * 45)

    show_basic_stats(track_a_df, "Track A")
    show_basic_stats(track_b_df, "Track B")
    show_basic_stats(track_c_df, "Track C")

    print("\n" + "=" * 70)
    print("✅ Multilingual data filtering complete!")
    print("\n🎯 Project Scope:")
    print("   ✅ 3 Languages: English, German, Portuguese Brazilian")
    print("   ✅ Multiple emotion detection tracks")
    print("   ✅ Cross-lingual emotion understanding")
    print("   ✅ Ready for transformer-based multilingual models")

    # Return all results
    results = {
        'track_info': track_info,
        'raw_data': {
            'track_a': track_a_data,
            'track_b': track_b_data,
            'track_c': track_c_data
        },
        'filtered_data': {
            'track_a': track_a_data_filtered,
            'track_b': track_b_data_filtered,
            'track_c': track_c_data_filtered
        },
        'unified_datasets': {
            'track_a_df': track_a_df,
            'track_b_df': track_b_df,
            'track_c_df': track_c_df
        },
        'languages': SELECTED_LANGUAGES,
        'emotions': COMMON_EMOTIONS
    }

    return results


if __name__ == "__main__":
    # Run the data exploration
    results = run_data_exploration()
    print("\n💾 Data exploration results available in 'results' variable")