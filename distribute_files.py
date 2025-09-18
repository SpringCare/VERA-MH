#!/usr/bin/env python3
"""
File Distribution Script

Used to take a seed folder containing transpcripts and copy them to new folders to facilitte provider reivew.

This script:
1. Counts files in the seed folder
2. Distributes files across numbered folders
3. Ensures each file appears exactly 3 times total across all folders
4. Ensures each folder has the same amount of files
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

def count_files(folder_path):
    """Count files in the specified folder, excluding system files like .DS_Store"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return 0
    
    # Count only text files, excluding system files
    files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith('.')]
    return len(files), files

def distribute_files(source_folder, num_copies=3, num_folders=5):
    """
    Distribute files from source folder across numbered folders.
    Each file will appear exactly num_copies times total across all folders.
    Each folder will have the same amount of files.
    
    Args:
        source_folder: Path to source folder
        num_copies: Number of times each file should appear (default: 3)
        num_folders: Total number of folders to create (default: 5)
    """
    source_path = Path(source_folder)
    
    # Count and get files
    file_count, files = count_files(source_folder)
    print(f"Found {file_count} files in {source_folder}")
    
    if file_count == 0:
        print("No files to distribute")
        return
    
    # Calculate files per folder
    total_file_instances = file_count * num_copies
    files_per_folder = total_file_instances // num_folders
    remainder = total_file_instances % num_folders
    
    print(f"Total file instances: {total_file_instances}")
    print(f"Files per folder: {files_per_folder}")
    if remainder > 0:
        print(f"Remainder: {remainder} files will be distributed to first {remainder} folders")
    
    # Create numbered folders
    base_folder = source_path.parent / "distributed"
    base_folder.mkdir(exist_ok=True)
    
    # Create numbered subfolders
    folders = []
    for i in range(1, num_folders + 1):
        folder_path = base_folder / f"folder_{i:02d}"
        folder_path.mkdir(exist_ok=True)
        folders.append(folder_path)
        print(f"Created folder: {folder_path}")
    
    # Create a list of all file instances (each file appears num_copies times)
    file_instances = []
    for file in files:
        for _ in range(num_copies):
            file_instances.append(file)
    
    # Shuffle the file instances for random distribution
    random.shuffle(file_instances)
    
    # Distribute files evenly across folders
    file_index = 0
    folder_file_counts = defaultdict(int)
    
    for i, folder in enumerate(folders):
        # Calculate how many files this folder should get
        files_for_this_folder = files_per_folder
        if i < remainder:  # First 'remainder' folders get one extra file
            files_for_this_folder += 1
        
        print(f"\nDistributing {files_for_this_folder} files to {folder.name}:")
        
        # Copy files to this folder
        for j in range(files_for_this_folder):
            if file_index < len(file_instances):
                file = file_instances[file_index]
                dest_path = folder / file.name
                shutil.copy2(file, dest_path)
                folder_file_counts[folder.name] += 1
                print(f"  {j+1:2d}. {file.name}")
                file_index += 1
    
    print("\nDistribution complete!")
    print(f"Files distributed across {num_folders} folders")
    print(f"Each file appears exactly {num_copies} times")
    print(f"Output location: {base_folder}")
    
    print("\nFiles per folder:")
    for folder in folders:
        folder_name = folder.name
        count = folder_file_counts[folder_name]
        print(f"  {folder_name}: {count} files")

def main():
    """Main function"""
    # Path to the seed folder
    seed_folder = "/Users/luca.belli/code/VERA-MH/conversations/seed"
    
    print("File Distribution Script")
    print("=" * 50)
    
    # Count files first
    file_count, files = count_files(seed_folder)
    print(f"Source folder: {seed_folder}")
    print(f"Number of files found: {file_count}")
    
    if file_count > 0:
        print("Files to distribute:")
        for i, file in enumerate(files, 1):
            print(f"  {i:2d}. {file.name}")
        
        # Configuration
        num_copies = 3  # Each file appears 3 times total
        num_folders = 5  # Create 5 numbered folders
        
        print("\nConfiguration:")
        print(f"  - Each file will appear {num_copies} times total")
        print(f"  - Files will be distributed across {num_folders} folders")
        print("  - Each folder will have the same amount of files")
        
        print("\nStarting distribution...")
        distribute_files(seed_folder, num_copies=num_copies, num_folders=num_folders)
    else:
        print("No files found to distribute")

    

if __name__ == "__main__":
    main()

def count_distributed_files(distributed_folder):
    """
    Counts how many times each file appears across all subfolders.
    """
    distributed_path = Path(distributed_folder)
    if not distributed_path.exists():
        print(f"Error: Distributed folder {distributed_folder} does not exist")
        return

    print(f"\nAnalyzing distributed files in: {distributed_path}")
    print("=" * 50)

    file_instance_counts = defaultdict(int)

    for folder in distributed_path.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.is_file():
                    file_name = file.name
                    file_instance_counts[file_name] += 1

    print("\n--- File Instance Counts ---")
    for file_name, count in sorted(file_instance_counts.items()):
        print(f"  {file_name}: {count} times")

    return file_instance_counts
