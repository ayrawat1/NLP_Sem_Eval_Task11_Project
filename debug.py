# Create this file: debug_dataset.py
from pathlib import Path
import pandas as pd


def debug_dataset_loading():
    base_path = Path("dataset")

    print("🔍 DATASET DEBUG ANALYSIS")
    print("=" * 40)

    for track in ['track_a', 'track_b', 'track_c']:
        track_path = base_path / track
        print(f"\n📁 {track}:")

        if track_path.exists():
            # List all files and folders
            for item in track_path.rglob("*"):
                if item.is_file():
                    print(f"   📄 {item.relative_to(base_path)} ({item.stat().st_size} bytes)")
                elif item.is_dir():
                    print(f"   📁 {item.relative_to(base_path)}/")

            # Try to load some files
            for subfolder in ['train', 'dev', 'test']:
                subfolder_path = track_path / subfolder
                if subfolder_path.exists():
                    print(f"\n   🔍 Checking {subfolder}:")

                    # Look for data files
                    data_files = (list(subfolder_path.glob("*.csv")) +
                                  list(subfolder_path.glob("*.tsv")) +
                                  list(subfolder_path.glob("*.txt")) +
                                  list(subfolder_path.glob("*.json")))

                    for file_path in data_files:
                        try:
                            print(f"      📄 Found: {file_path.name}")

                            # Try to read the file
                            if file_path.suffix.lower() == '.csv':
                                df = pd.read_csv(file_path)
                            elif file_path.suffix.lower() == '.tsv':
                                df = pd.read_csv(file_path, sep='\t')
                            else:
                                df = pd.read_csv(file_path)

                            print(f"         ✅ Loaded: {df.shape}")
                            print(f"         📋 Columns: {list(df.columns)}")
                            print(f"         📄 First row: {df.iloc[0].to_dict()}")

                        except Exception as e:
                            print(f"         ❌ Error: {e}")


if __name__ == "__main__":
    debug_dataset_loading()