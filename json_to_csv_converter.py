#!/usr/bin/env python3
"""
Convert JSON evaluation files to CSV format.
Extracts UUID (first 6 chars of filename) and evaluation fields.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any

def extract_uuid_from_filename(filename: str) -> str:
    """Extract the first 6 characters as UUID from filename."""
    return filename[:6]

def parse_json_file(file_path: Path) -> Dict[str, Any]:
    """Parse a JSON file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def get_evaluation_fields(data: Dict[str, Any]) -> Dict[str, str]:
    """Extract evaluation fields from JSON data."""
    evaluations = data.get('evaluations', {})
    if not evaluations:
        print(f"Warning: No evaluations found in data")
        return {}
    return evaluations

def convert_json_to_csv(input_dir: str, output_csv: str):
    """Convert all JSON files in the input directory to a single CSV."""
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Get all JSON files
    json_files = list(input_path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Get all unique evaluation fields from all files
    all_evaluation_fields = set()
    file_data = []
    
    # First pass: collect all evaluation fields and data
    for json_file in json_files:
        data = parse_json_file(json_file)
        if not data:
            continue
            
        evaluations = get_evaluation_fields(data)
        if evaluations:
            all_evaluation_fields.update(evaluations.keys())
            
            file_data.append({
                'uuid': extract_uuid_from_filename(json_file.stem),
                'filename': json_file.name,
                'evaluations': evaluations
            })
    
    if not file_data:
        print("No valid data found in any JSON files")
        return
    
    # Sort evaluation fields for consistent column order
    evaluation_fields = sorted(list(all_evaluation_fields))
    
    # Prepare CSV headers
    headers = ['UUID'] + evaluation_fields
    
    # Write CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for item in file_data:
            row = [item['uuid']]
            
            # Add evaluation values in the same order as headers
            for field in evaluation_fields:
                row.append(item['evaluations'].get(field, ''))
            
            writer.writerow(row)
    
    print(f"Successfully converted {len(file_data)} files to {output_csv}")
    print(f"Columns: {headers}")

def test_conversion(input_dir: str, csv_file: str):
    """Test function to verify CSV conversion matches original JSON data."""
    print("\n" + "="*60)
    print("TESTING CONVERSION ACCURACY")
    print("="*60)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return False
    
    if not Path(csv_file).exists():
        print(f"Error: CSV file {csv_file} does not exist")
        return False
    
    # Read CSV data
    csv_data = {}
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                uuid = row['UUID']
                csv_data[uuid] = {k: v for k, v in row.items() if k != 'UUID'}
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return False
    
    # Read and compare with original JSON files
    json_files = list(input_path.glob("*.json"))
    errors = 0
    total_files = 0
    
    for json_file in json_files:
        uuid = extract_uuid_from_filename(json_file.stem)
        if uuid not in csv_data:
            print(f"ERROR: UUID {uuid} from {json_file.name} not found in CSV")
            errors += 1
            continue
        
        # Parse JSON file
        json_data = parse_json_file(json_file)
        if not json_data:
            continue
        
        evaluations = get_evaluation_fields(json_data)
        if not evaluations:
            continue
        
        total_files += 1
        
        # Compare each evaluation field
        csv_row = csv_data[uuid]
        for field, json_value in evaluations.items():
            if field not in csv_row:
                print(f"ERROR: Field '{field}' missing in CSV for UUID {uuid}")
                errors += 1
            elif csv_row[field] != json_value:
                print(f"ERROR: Mismatch for UUID {uuid}, field '{field}':")
                print(f"  JSON: {json_value}")
                print(f"  CSV:  {csv_row[field]}")
                errors += 1
    
    print(f"\nTest Results:")
    print(f"Total files processed: {total_files}")
    print(f"Total errors found: {errors}")
    
    if errors == 0:
        print("✅ All data matches perfectly!")
        return True
    else:
        print("❌ Found data mismatches!")
        return False

def main():
    """Main function to run the conversion."""
    # Input directory (the evaluations folder)
    input_directory = "evaluations/j_claude-sonnet-4-20250514_20250825_142901_627__p_claude_sonnet_4_20250514__a_claude_sonnet_4_20250514_20250822_172711_t30_r5"
    
    # Output CSV file
    output_csv = "evaluations_output.csv"
    
    print(f"Converting JSON files from: {input_directory}")
    print(f"Output CSV: {output_csv}")
    
    convert_json_to_csv(input_directory, output_csv)
    
    # Run test to verify conversion accuracy
    test_conversion(input_directory, output_csv)

if __name__ == "__main__":
    main() 