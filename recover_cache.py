import os
import pickle
import shutil
from collections import defaultdict
from settings import (
    PAPER_ABSTRACTS_PATH,
    PAPER_SUMMARIZATIONS_PATH,
    PAPER_FULL_CONTENTS_PATH,
)


def backup_corrupted_files():
    """Backup corrupted files before attempting recovery"""
    backup_dir = "cache_backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    files_to_backup = [
        PAPER_ABSTRACTS_PATH,
        PAPER_FULL_CONTENTS_PATH,
        PAPER_SUMMARIZATIONS_PATH,
    ]

    for file_path in files_to_backup:
        if os.path.exists(file_path):
            backup_path = os.path.join(backup_dir, os.path.basename(file_path))
            shutil.copy2(file_path, backup_path)
            print(f"Backed up: {file_path} -> {backup_path}")


def attempt_partial_recovery(file_path):
    """Attempt to recover partial data from corrupted pickle file"""
    print(f"Attempting partial recovery for: {file_path}")

    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return defaultdict(str)

    # Try different recovery strategies
    strategies = [
        partial_read_recovery,
        chunk_read_recovery,
        manual_unpickle_recovery,
    ]

    for strategy in strategies:
        try:
            result = strategy(file_path)
            if result:
                print(
                    f"Successfully recovered {len(result)} items using {strategy.__name__}"
                )
                return result
        except Exception as e:
            print(f"Recovery strategy {strategy.__name__} failed: {e}")

    print(f"All recovery strategies failed for {file_path}")
    return defaultdict(str)


def partial_read_recovery(file_path):
    """Try to read file in chunks to find recoverable data"""
    recovered_data = defaultdict(str)

    with open(file_path, "rb") as f:
        # Try to read file in smaller chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        position = 0

        while True:
            try:
                f.seek(position)
                chunk = f.read(chunk_size)
                if not chunk:
                    break

                # Try to find pickle protocol markers
                if b"\x80" in chunk:  # Common pickle protocol marker
                    # Attempt to unpickle from this position
                    f.seek(position)
                    try:
                        data = pickle.load(f)
                        if isinstance(data, dict):
                            recovered_data.update(data)
                            return recovered_data
                    except:
                        pass

                position += chunk_size // 2  # Overlap chunks

            except Exception as e:
                position += chunk_size
                if position >= os.path.getsize(file_path):
                    break

    return recovered_data


def chunk_read_recovery(file_path):
    """Try to read file by skipping corrupted parts"""
    recovered_data = defaultdict(str)

    with open(file_path, "rb") as f:
        file_size = os.path.getsize(file_path)

        # Try reading from different positions
        for start_pos in range(0, file_size, 1024 * 1024):  # Every 1MB
            try:
                f.seek(start_pos)
                data = pickle.load(f)
                if isinstance(data, dict):
                    recovered_data.update(data)
                    return recovered_data
            except:
                continue

    return recovered_data


def manual_unpickle_recovery(file_path):
    """Manual unpickling with error handling"""
    recovered_data = defaultdict(str)

    try:
        with open(file_path, "rb") as f:
            # Try to read the file byte by byte to find valid pickle data
            content = f.read()

            # Look for pickle opcodes and try to reconstruct
            for i in range(len(content) - 100):  # Skip near end
                try:
                    if content[i : i + 1] == b"\x80":  # Pickle protocol marker
                        test_data = content[i:]
                        data = pickle.loads(test_data)
                        if isinstance(data, dict):
                            recovered_data.update(data)
                            return recovered_data
                except:
                    continue

    except Exception as e:
        print(f"Manual recovery failed: {e}")

    return recovered_data


def main():
    print("Starting cache recovery process...")

    # First, backup corrupted files
    backup_corrupted_files()

    # Attempt to recover each cache file
    files_to_recover = [
        ("paper_abstracts", PAPER_ABSTRACTS_PATH),
        ("paper_full_contents", PAPER_FULL_CONTENTS_PATH),
        ("paper_summarizations", PAPER_SUMMARIZATIONS_PATH),
    ]

    recovered_data = {}

    for name, file_path in files_to_recover:
        print(f"\n--- Recovering {name} ---")
        recovered_data[name] = attempt_partial_recovery(file_path)

        # Show recovery statistics
        if recovered_data[name]:
            print(f"Recovered {len(recovered_data[name])} items from {name}")

            # Show a sample of recovered data
            sample_keys = list(recovered_data[name].keys())[:3]
            for key in sample_keys:
                value_preview = str(recovered_data[name][key])[:100]
                print(f"  Sample: {key} -> {value_preview}...")
        else:
            print(f"No data recovered from {name}")

    # Save recovered data to new files
    for name, data in recovered_data.items():
        if data:
            file_path = files_to_recover[
                next(i for i, (n, _) in enumerate(files_to_recover) if n == name)
            ][1]

            # Save recovered data
            recovered_file_path = file_path.replace(".pickle", "_recovered.pickle")
            with open(recovered_file_path, "wb") as f:
                pickle.dump(dict(data), f)
            print(f"Saved recovered data to: {recovered_file_path}")

    print("\nRecovery process completed!")
    print("Next steps:")
    print("1. Check the recovered files")
    print("2. If recovery is successful, replace original files")
    print("3. If recovery fails, use the updated CacheManager with error handling")


if __name__ == "__main__":
    main()
