"""
SemEval 2025 Task 11: Multilingual Data Preprocessing
====================================================

This module handles multilingual text preprocessing, emotion label extraction,
and data preparation for English, German, and Portuguese Brazilian.

Author: SemEval 2025 Task 11 Team
Date: 2025
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# Configuration
EMOTIONS = ['joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']
LANGUAGES = ['eng', 'deu', 'ptbr']

# Multilingual stopwords
MULTILINGUAL_STOPWORDS = {
    'eng': set(stopwords.words('english')),
    'deu': set(stopwords.words('german')),
    'ptbr': set(stopwords.words('portuguese'))
}


class MultilingualEmotionDataPreprocessor:
    """
    Comprehensive multilingual data preprocessor for emotion detection
    """

    def __init__(self, emotions=EMOTIONS, languages=LANGUAGES):
        """
        Initialize the preprocessor

        Args:
            emotions (list): List of emotion categories
            languages (list): List of language codes
        """
        self.emotions = emotions
        self.languages = languages
        self.stop_words = {}

        # Load stopwords for each language
        for lang in languages:
            if lang in MULTILINGUAL_STOPWORDS:
                self.stop_words[lang] = MULTILINGUAL_STOPWORDS[lang]
            else:
                self.stop_words[lang] = set()  # Fallback to empty set

        print(f"✅ Initialized for languages: {languages}")

    def detect_language_simple(self, text):
        """
        Simple language detection based on common words

        Args:
            text (str): Input text

        Returns:
            str: Detected language code
        """
        text_lower = str(text).lower()

        # Simple heuristics (you could use a proper language detector)
        if any(word in text_lower for word in ['the', 'and', 'is', 'are', 'you', 'that']):
            return 'eng'
        elif any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'sind']):
            return 'deu'
        elif any(word in text_lower for word in ['que', 'com', 'uma', 'para', 'este', 'está']):
            return 'ptbr'
        else:
            return 'eng'  # Default to English

    def clean_text_multilingual(self, text, language=None):
        """
        Clean and preprocess text data for multiple languages

        Args:
            text (str): Input text
            language (str, optional): Language code

        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""

        text = str(text).lower()

        # Detect language if not provided
        if language is None:
            language = self.detect_language_simple(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags (but keep the content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)  # Keep hashtag content

        # Handle contractions (language-specific)
        if language == 'eng':
            contractions = {
                "won't": "will not", "can't": "cannot", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will",
                "'d": " would", "'m": " am"
            }
            for contraction, expansion in contractions.items():
                text = text.replace(contraction, expansion)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def analyze_multilingual_distribution(self, df, text_col):
        """
        Analyze language distribution in the dataset

        Args:
            df (pd.DataFrame): Input dataframe
            text_col (str): Name of text column

        Returns:
            Counter: Language distribution
        """
        if text_col not in df.columns:
            return None

        print(f"\n🌍 Language Distribution Analysis:")

        # Detect language for each text
        languages_detected = []
        for text in df[text_col].head(100):  # Sample first 100 for speed
            lang = self.detect_language_simple(text)
            languages_detected.append(lang)

        lang_counts = Counter(languages_detected)
        total_sampled = len(languages_detected)

        for lang, count in lang_counts.items():
            percentage = (count / total_sampled) * 100
            lang_name = {'eng': 'English', 'deu': 'German', 'ptbr': 'Portuguese'}.get(lang, lang)
            print(f"  {lang_name} ({lang}): {count}/{total_sampled} ({percentage:.1f}%)")

        return lang_counts

    def extract_emotion_labels(self, df):
        """
        Extract emotion labels from dataframe
        Returns both binary labels and intensity labels if available

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            tuple: (binary_labels, intensity_labels, emotion_cols, intensity_cols)
        """
        # Find emotion columns in the dataframe
        emotion_cols = []
        intensity_cols = []

        for col in df.columns:
            col_lower = col.lower()
            for emotion in self.emotions:
                if emotion in col_lower:
                    if 'intensity' in col_lower:
                        intensity_cols.append(col)
                    else:
                        emotion_cols.append(col)

        print(f"Found emotion columns: {emotion_cols}")
        print(f"Found intensity columns: {intensity_cols}")

        # Create binary emotion matrix for Track A
        binary_labels = None
        if emotion_cols:
            # Ensure we have all emotions in the right order
            ordered_emotion_cols = []
            for emotion in self.emotions:
                matching_cols = [col for col in emotion_cols if emotion in col.lower()]
                if matching_cols:
                    ordered_emotion_cols.append(matching_cols[0])
                else:
                    print(f"⚠️  Warning: No column found for emotion '{emotion}'")

            if ordered_emotion_cols:
                binary_labels = df[ordered_emotion_cols].values.astype(float)
                print(f"Binary labels shape: {binary_labels.shape}")

        # Create intensity matrix for Track B
        intensity_labels = None
        if intensity_cols:
            # Order intensity columns to match emotions
            ordered_intensity_cols = []
            for emotion in self.emotions:
                matching_cols = [col for col in intensity_cols if emotion in col.lower()]
                if matching_cols:
                    ordered_intensity_cols.append(matching_cols[0])

            if ordered_intensity_cols:
                intensity_labels = df[ordered_intensity_cols].values.astype(float)
                print(f"Intensity labels shape: {intensity_labels.shape}")

        return binary_labels, intensity_labels, emotion_cols, intensity_cols

    def analyze_label_distribution(self, binary_labels, intensity_labels=None):
        """
        Analyze the distribution of emotion labels

        Args:
            binary_labels (np.ndarray): Binary emotion labels
            intensity_labels (np.ndarray, optional): Intensity labels
        """
        if binary_labels is not None:
            print("\n📊 Binary Label Distribution (Track A):")
            for i, emotion in enumerate(self.emotions[:binary_labels.shape[1]]):
                positive_count = np.sum(binary_labels[:, i] > 0)
                total_count = len(binary_labels)
                percentage = (positive_count / total_count) * 100
                print(f"  {emotion}: {positive_count}/{total_count} ({percentage:.1f}%)")

        if intensity_labels is not None:
            print("\n📊 Intensity Label Distribution (Track B):")
            for i, emotion in enumerate(self.emotions[:intensity_labels.shape[1]]):
                intensity_counts = Counter(intensity_labels[:, i])
                print(f"  {emotion}: {dict(sorted(intensity_counts.items()))}")

    def create_train_val_split(self, df, binary_labels, intensity_labels=None,
                               test_size=0.2, random_state=42):
        """
        Create train/validation splits while maintaining label distribution

        Args:
            df (pd.DataFrame): Input dataframe
            binary_labels (np.ndarray): Binary labels
            intensity_labels (np.ndarray, optional): Intensity labels
            test_size (float): Proportion of test set
            random_state (int): Random seed

        Returns:
            dict: Split data
        """
        print(f"\n🔄 Creating train/validation split ({int((1 - test_size) * 100)}%/{int(test_size * 100)}%)")

        # Find text column
        text_cols = [col for col in df.columns if 'text' in col.lower()]
        if not text_cols:
            raise ValueError("No text column found in dataframe")

        text_col = text_cols[0]
        X = df[text_col].values

        # Use binary labels for stratification
        y_stratify = binary_labels if binary_labels is not None else None

        if y_stratify is not None and y_stratify.shape[1] > 1:
            # For multi-label stratification, create a string representation
            y_stratify_str = [''.join(map(str, row.astype(int))) for row in y_stratify]

            # Check if we have enough samples for each class
            unique_labels, counts = np.unique(y_stratify_str, return_counts=True)
            min_count = min(counts)

            if min_count < 2:
                print("⚠️  Warning: Some label combinations have only 1 sample. Using random split.")
                y_stratify_str = None
        else:
            y_stratify_str = None

        # Split the data
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=y_stratify_str if y_stratify_str else None
        )

        # Create splits
        X_train, X_val = X[train_idx], X[val_idx]

        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'train_idx': train_idx,
            'val_idx': val_idx
        }

        if binary_labels is not None:
            splits['y_train'] = binary_labels[train_idx]
            splits['y_val'] = binary_labels[val_idx]

        if intensity_labels is not None:
            splits['intensity_train'] = intensity_labels[train_idx]
            splits['intensity_val'] = intensity_labels[val_idx]

        print(f"✅ Train set: {len(X_train)} samples")
        print(f"✅ Validation set: {len(X_val)} samples")

        return splits


def preprocess_track_data(df, track_name="Unknown"):
    """
    Complete preprocessing pipeline for a track's data

    Args:
        df (pd.DataFrame): Input dataframe
        track_name (str): Name of the track

    Returns:
        dict: Complete preprocessing results
    """
    print(f"\n🎯 Preprocessing {track_name} Data")
    print("-" * 30)

    if df is None or len(df) == 0:
        print("❌ No data to process")
        return None

    # Initialize preprocessor
    preprocessor = MultilingualEmotionDataPreprocessor()

    # Find and clean text column
    text_cols = [col for col in df.columns if 'text' in col.lower()]
    if not text_cols:
        print("❌ No text column found")
        return None

    text_col = text_cols[0]
    print(f"📝 Processing text column: '{text_col}'")

    # Clean text
    df_clean = df.copy()
    df_clean['clean_text'] = df_clean[text_col].apply(lambda x: preprocessor.clean_text_multilingual(x))

    # Analyze language distribution
    lang_distribution = preprocessor.analyze_multilingual_distribution(df_clean, text_col)

    # Show cleaning example
    print("\n🧹 Text Cleaning Example:")
    original_text = str(df[text_col].iloc[0])
    cleaned_text = df_clean['clean_text'].iloc[0]
    print(f"  Original: {original_text}")
    print(f"  Cleaned:  {cleaned_text}")

    # Extract emotion labels
    binary_labels, intensity_labels, emotion_cols, intensity_cols = preprocessor.extract_emotion_labels(df_clean)

    # Analyze label distribution
    preprocessor.analyze_label_distribution(binary_labels, intensity_labels)

    # Create train/val split
    splits = preprocessor.create_train_val_split(df_clean, binary_labels, intensity_labels)

    # Return all processed data
    return {
        'dataframe': df_clean,
        'preprocessor': preprocessor,
        'splits': splits,
        'binary_labels': binary_labels,
        'intensity_labels': intensity_labels,
        'emotion_cols': emotion_cols,
        'intensity_cols': intensity_cols,
        'text_col': text_col,
        'lang_distribution': lang_distribution
    }


def convert_labels_to_intensities(labels, method='normalize'):
    """
    Convert discrete emotion labels to continuous intensity values

    Args:
        labels (np.ndarray): Array with discrete values [0, 1, 2, 3]
        method (str): 'normalize' or 'scale'

    Returns:
        np.ndarray: Array with continuous intensity values [0.0-1.0]
    """
    if method == 'normalize':
        # Normalize to 0-1 range: 0->0.0, 1->0.33, 2->0.67, 3->1.0
        max_label = labels.max()
        if max_label > 0:
            intensities = labels / max_label
        else:
            intensities = labels.copy()
    elif method == 'scale':
        # Alternative scaling: 0->0.0, 1->0.25, 2->0.5, 3->0.75
        intensities = labels / 4.0
    else:
        raise ValueError("Method must be 'normalize' or 'scale'")

    return intensities.astype(np.float32)


def fix_track_b_intensity_data(track_b_processed):
    """
    Fix Track B data structure by converting discrete labels to continuous intensities

    Args:
        track_b_processed (dict): Processed Track B data

    Returns:
        dict: Fixed Track B data with intensity values
    """
    print("🔧 FIXING TRACK B DATA STRUCTURE")
    print("=" * 40)

    if track_b_processed is None:
        print("❌ track_b_processed not found!")
        return None

    splits = track_b_processed['splits']

    print(f"📊 Current Track B Structure:")
    print(f"   Current y_train shape: {splits['y_train'].shape}")
    print(f"   Current y_train unique: {sorted(np.unique(splits['y_train']))}")
    print(f"   Current y_train sample:\n{splits['y_train'][:3]}")

    # Convert discrete labels to continuous intensities
    print(f"\n🔄 Converting discrete labels to continuous intensities...")

    # Convert training data
    intensity_train = convert_labels_to_intensities(splits['y_train'], method='normalize')
    intensity_val = convert_labels_to_intensities(splits['y_val'], method='normalize')

    # Add intensity data to Track B
    splits['intensity_train'] = intensity_train
    splits['intensity_val'] = intensity_val

    print(f"✅ Conversion complete!")
    print(f"   New intensity_train shape: {intensity_train.shape}")
    print(f"   New intensity_train range: [{intensity_train.min():.3f}, {intensity_train.max():.3f}]")
    print(f"   New intensity_train sample:\n{intensity_train[:3]}")
    print(f"   New intensity_train dtype: {intensity_train.dtype}")

    # Verify the conversion mapping
    print(f"\n📋 Label to Intensity Mapping:")
    original_labels = sorted(np.unique(splits['y_train']))
    for label in original_labels:
        intensity = label / np.max(original_labels) if np.max(original_labels) > 0 else label
        print(f"   Label {int(label)} → Intensity {intensity:.3f}")

    print(f"\n✅ Track B now has both:")
    print(f"   - y_train: discrete labels for fallback classification")
    print(f"   - intensity_train: continuous values for regression")

    # Update the processed data
    track_b_processed['splits'] = splits

    print(f"\n🎯 Track B is now ready for intensity prediction!")
    print(f"   - Regression task: ✅")
    print(f"   - Continuous values: ✅")
    print(f"   - Pearson correlation: ✅")

    return track_b_processed


def run_preprocessing_pipeline(exploration_results):
    """
    Main function to run the complete preprocessing pipeline

    Args:
        exploration_results (dict): Results from data exploration

    Returns:
        dict: Complete preprocessing results for all tracks
    """
    print("\n" + "=" * 50)
    print("🚀 STARTING MULTILINGUAL DATA PREPROCESSING")
    print("=" * 50)

    if exploration_results is None:
        print("❌ No exploration results provided")
        return None

    # Get unified datasets from exploration
    unified_datasets = exploration_results.get('unified_datasets', {})

    # Process Track A (Multi-label Classification)
    track_a_df = unified_datasets.get('track_a_df')
    if track_a_df is not None:
        track_a_processed = preprocess_track_data(track_a_df, "Track A (English, German, Portuguese)")
    else:
        print("⚠️  Track A data not available")
        track_a_processed = None

    # Process Track B (Intensity Prediction)
    track_b_df = unified_datasets.get('track_b_df')
    if track_b_df is not None:
        track_b_processed = preprocess_track_data(track_b_df, "Track B (English, German, Portuguese)")
        # Fix Track B for intensity prediction
        if track_b_processed:
            track_b_processed = fix_track_b_intensity_data(track_b_processed)
    else:
        print("⚠️  Track B data not available")
        track_b_processed = None

    # Process Track C (Cross-lingual)
    track_c_df = unified_datasets.get('track_c_df')
    if track_c_df is not None:
        track_c_processed = preprocess_track_data(track_c_df, "Track C (Cross-lingual: English, German, Portuguese)")
    else:
        print("⚠️  Track C data not available")
        track_c_processed = None

    print("\n" + "=" * 50)
    print("✅ MULTILINGUAL DATA PREPROCESSING COMPLETE!")
    print("=" * 50)

    # Summary
    processed_tracks = []
    if track_a_processed: processed_tracks.append("Track A (Multi-label)")
    if track_b_processed: processed_tracks.append("Track B (Intensity)")
    if track_c_processed: processed_tracks.append("Track C (Cross-lingual)")

    print(f"\n📊 Processed tracks: {', '.join(processed_tracks)}")
    print("\n🎯 Ready for model development!")

    results = {
        'track_a_processed': track_a_processed,
        'track_b_processed': track_b_processed,
        'track_c_processed': track_c_processed,
        'languages': LANGUAGES,
        'emotions': EMOTIONS
    }

    print("\nAvailable results:")
    if track_a_processed:
        print("  - track_a_processed: Complete Track A preprocessing results")
    if track_b_processed:
        print("  - track_b_processed: Complete Track B preprocessing results")
    if track_c_processed:
        print("  - track_c_processed: Complete Track C preprocessing results")

    print(f"\n🌍 Cross-lingual Capability:")
    print(f"   ✅ Train on Track A/B (labeled data)")
    print(f"   ✅ Test on Track C (cross-lingual evaluation)")
    print(f"   ✅ Demonstrates multilingual emotion understanding")

    return results


if __name__ == "__main__":
    # This would typically be called after data exploration
    print("🔧 Multilingual Data Preprocessing Module")
    print("Run this after data exploration to preprocess the datasets.")
    print("Usage: results = run_preprocessing_pipeline(exploration_results)")