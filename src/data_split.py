import os
import random
import shutil
from pathlib import Path

def create_stratified_patient_split(raw_data_dir, output_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Performs a stratified, patient-disjoint split for the ORCHID dataset.
    Since every folder represents a unique patient, we split proportionally 
    within each class to maintain class balance across Train/Val/Test.
    """
    random.seed(seed)
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)
    classes = ['normal', 'osmf', 'wdoscc', 'mdoscc', 'pdoscc']
    
    print(f"🚀 Starting Stratified Split from {raw_path} to {out_path}...\n")
    total_patients_processed = 0

    for cls in classes:
        cls_raw_dir = raw_path / cls
        if not cls_raw_dir.exists():
            print(f"⚠️ Warning: Folder {cls_raw_dir} not found. Skipping.")
            continue
            
        # 1. Get all patient folders for this class
        patient_folders = [f for f in cls_raw_dir.iterdir() if f.is_dir()]
        total_in_class = len(patient_folders)
        total_patients_processed += total_in_class
        
        # 2. Shuffle them safely
        random.shuffle(patient_folders)
        
        # 3. Calculate split indices
        train_end = int(total_in_class * train_ratio)
        val_end = train_end + int(total_in_class * val_ratio)
        
        splits = {
            'train': patient_folders[:train_end],
            'val': patient_folders[train_end:val_end],
            'test': patient_folders[val_end:]
        }
        
        # 4. Copy the files
        print(f"📂 Processing '{cls}': {total_in_class} patients -> "
              f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
              
        for split_name, folders in splits.items():
            for folder in folders:
                # Recreate the structure: output_dir / split / class / folder_name
                dest_folder = out_path / split_name / cls / folder.name
                dest_folder.mkdir(parents=True, exist_ok=True)
                
                # Copy all png patches
                for img_file in folder.glob('*.png'):
                    shutil.copy2(img_file, dest_folder / img_file.name)

    print(f"\n✅ Success! Total Unique Patients Processed: {total_patients_processed} (Should be 150)")

if __name__ == "__main__":
    create_stratified_patient_split("data/raw", "data/processed", train_ratio=0.7, val_ratio=0.2, seed=42)