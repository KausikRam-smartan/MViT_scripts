import os
import random
import shutil
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split


def discover_classes(data_root):
    """
    Discover class folders where each class contains subfolders with .mp4 files.
    
    Structure expected:
        data_root/
        ├── class1/
        │   ├── subfolder1/
        │   │   └── video.mp4
        │   └── subfolder2/
        │       └── video.mp4
        ├── class2/
        │   └── subfolder/
        │       └── video.mp4
    
    Returns:
        class_dict: {class_name: index}
        class_videos: {class_name: [list of video paths]}
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    
    class_videos = {}
    
    # Iterate through immediate subdirectories (these are class folders)
    for class_folder in sorted(data_root.iterdir()):
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        video_paths = []
        
        # Search recursively within this class folder for all .mp4 files
        for video_file in class_folder.rglob("*.mp4"):
            video_paths.append(video_file)
        
        if video_paths:
            class_videos[class_name] = video_paths
    
    if not class_videos:
        raise ValueError(f"No valid class folders with .mp4 files found in {data_root}")
    
    # Create dictionary with automatic indexing (sorted alphabetically)
    class_dict = {class_name: idx for idx, class_name in enumerate(sorted(class_videos.keys()))}
    
    print(f"\n{'='*60}")
    print(f"DISCOVERED CLASSES")
    print(f"{'='*60}")
    print(f"Found {len(class_dict)} classes:\n")
    
    total_videos = 0
    for class_name, idx in sorted(class_dict.items(), key=lambda x: x[1]):
        num_videos = len(class_videos[class_name])
        total_videos += num_videos
        
        # Show subfolders
        subfolders = set()
        for video_path in class_videos[class_name]:
            # Get the immediate parent folder name (person/subfolder name)
            subfolder = video_path.parent.name
            subfolders.add(subfolder)
        
        subfolder_str = ", ".join(sorted(list(subfolders))[:3])
        if len(subfolders) > 3:
            subfolder_str += f", ... (+{len(subfolders)-3} more)"
        
        print(f"  [{idx:2d}] {class_name:30s} - {num_videos:4d} videos from {len(subfolders)} subfolder(s)")
        print(f"       Subfolders: {subfolder_str}")
    
    print(f"\n  Total videos across all classes: {total_videos}")
    print(f"{'='*60}\n")
    
    return class_dict, class_videos


def split_dataset_balanced(data_root, output_root, seed=42, class_dict=None):
    """
    For each class:

    - If the class has >= 75 videos:
        Use balanced split → 60 train, 15 val

    - If the class has < 75 videos:
        Use fallback split → 80% train, 20% val
    """

    random.seed(seed)
    data_root = Path(data_root)
    output_root = Path(output_root)

    # Auto-discover classes
    if class_dict is None:
        class_dict, class_videos = discover_classes(data_root)
    else:
        class_videos = {
            class_name: list((data_root / class_name).rglob("*.mp4"))
            for class_name in class_dict
        }

    print("\n========================================================")
    print(" SPLITTING DATASET (Balanced + 80/20 fallback)")
    print("========================================================\n")

    train_split = []
    val_split   = []

    for class_name, videos in class_videos.items():
        videos = list(videos)
        random.shuffle(videos)
        num_videos = len(videos)

        print(f"\nClass: {class_name}")
        print(f"  Total videos found: {num_videos}")

        if num_videos >= 100:
            # Balanced split
            print("  → Using BALANCED split (70 train / 15 val)")
            train_videos = videos[:80]
            val_videos   = videos[80:97]

        else:
            # Fallback split (80/20)
            print("  → Using FALLBACK split (80% train / 20% val)")
            n_train = max(1, int(num_videos * 0.8))
            n_val   = num_videos - n_train
            print(f"    - Train: {n_train}")
            print(f"    - Val:   {n_val}")

            train_videos = videos[:n_train]
            val_videos   = videos[n_train:]

        # Add to global split lists
        for v in train_videos:
            train_split.append((v, class_name))
        for v in val_videos:
            val_split.append((v, class_name))

    # WRITE OUTPUT -------------------------------------------------------------
    sets = {"train": train_split, "val": val_split}

    for split_name, samples in sets.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_root / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=" ")

            for src_path, class_name in samples:
                dst_dir = split_dir / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)

                dst_path = dst_dir / src_path.name

                # Handle duplicates
                if dst_path.exists():
                    parent = src_path.parent.name
                    dst_path = dst_dir / f"{parent}_{src_path.name}"

                shutil.copy2(src_path, dst_path)

                rel_path = f"{class_name}/{dst_path.name}"
                class_idx = class_dict[class_name]
                writer.writerow([rel_path, class_idx])

        print(f"\n{split_name.upper()} SET WRITTEN:")
        print(f"  Videos: {len(samples)}")

    # Save class mapping
    class_dict_path = output_root / "class_mapping.txt"
    with open(class_dict_path, "w") as f:
        f.write("CLASS_DICT = {\n")
        for cname, idx in sorted(class_dict.items(), key=lambda x: x[1]):
            f.write(f"    '{cname}': {idx},\n")
        f.write("}\n")

    print(f"\nClass mapping saved to {class_dict_path}")
    print("\nDataset split complete.\n")

    return class_dict



def analyze_dataset(data_root):
    """
    Analyze dataset structure and provide statistics without splitting.
    """
    data_root = Path(data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Path: {data_root}\n")
    
    class_dict, class_videos = discover_classes(data_root)
    
    total_videos = 0
    min_videos = float('inf')
    max_videos = 0
    
    for class_name, videos in class_videos.items():
        num_videos = len(videos)
        total_videos += num_videos
        min_videos = min(min_videos, num_videos)
        max_videos = max(max_videos, num_videos)
    
    print(f"\nSummary Statistics:")
    print(f"  Total videos: {total_videos}")
    print(f"  Total classes: {len(class_dict)}")
    print(f"  Average videos per class: {total_videos / len(class_dict):.1f}")
    print(f"  Min videos in a class: {min_videos}")
    print(f"  Max videos in a class: {max_videos}")
    print(f"  Imbalance ratio (max/min): {max_videos / min_videos:.2f}")
    
    if max_videos / min_videos > 3:
        print(f"\n  ⚠️  WARNING: Significant class imbalance detected!")
        print(f"     Consider balancing your dataset or using class weights during training.")
    
    return class_dict


if __name__ == "__main__":
    # ============================================
    # YOUR DATASET PATH
    # ============================================
    DATASET_FOLDER = "/home/smartan5070/Downloads/NEW_updated_classes/classes"
    OUTPUT_FOLDER = "/home/smartan5070/Downloads/data_15_12_25"
    
    # ============================================
    # STEP 1: ANALYZE DATASET FIRST
    # ============================================
    print("="*60)
    print("STEP 1: ANALYZING DATASET")
    print("="*60)
    analyze_dataset(DATASET_FOLDER)
    
    # ============================================
    # STEP 2: SPLIT DATASET
    # ============================================
    print("\n" + "="*60)
    print("STEP 2: SPLITTING DATASET")
    print("="*60)
    
    # Uncomment the line below to actually perform the split
    split_dataset_balanced(DATASET_FOLDER, OUTPUT_FOLDER, seed=42)
    
    print("\n✅ Done! Your dataset is ready for training.")
    print(f"\nTo use in training, copy the CLASS_DICT from:")
    print(f"  {OUTPUT_FOLDER}/class_mapping.txt")