#!/usr/bin/env python3
"""
Expand the all_scores JSON column into separate columns in the MTEB scores CSV.

This script reads the mteb_scores.csv file, parses the all_scores JSON column,
and creates separate columns for each score metric.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Set


def extract_scores_from_json(all_scores_str: str, preferred_split: str = "test") -> Dict[str, Any]:
    """
    Extract all scores from the all_scores JSON string.
    
    Prefers scores from the preferred_split (usually "test"), but will use
    other splits if test is not available.
    """
    if pd.isna(all_scores_str) or not all_scores_str:
        return {}
    
    try:
        scores_dict = json.loads(all_scores_str)
    except (json.JSONDecodeError, TypeError):
        return {}
    
    # Try to get scores from preferred split first
    if preferred_split in scores_dict and isinstance(scores_dict[preferred_split], dict):
        return scores_dict[preferred_split]
    
    # Fallback to any available split
    for split_name, split_data in scores_dict.items():
        if isinstance(split_data, dict):
            return split_data
    
    return {}


def find_all_score_keys(df: pd.DataFrame, all_scores_col: str = "all_scores") -> Set[str]:
    """Find all unique score keys across all rows."""
    all_keys = set()
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(all_scores_col)):
            scores = extract_scores_from_json(str(row[all_scores_col]))
            # Only include numeric/boolean values, skip lists/dicts
            for key, value in scores.items():
                if isinstance(value, (int, float, bool)) or value is None:
                    all_keys.add(key)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                    # Some metrics have lists of scores - we'll take the mean
                    all_keys.add(f"{key}_mean")
    
    return all_keys


def expand_all_scores_column(df: pd.DataFrame, all_scores_col: str = "all_scores") -> pd.DataFrame:
    """
    Expand the all_scores JSON column into separate columns.
    
    Returns a new DataFrame with additional columns for each score metric.
    """
    # Find all unique keys
    print("Finding all unique score keys...")
    all_keys = find_all_score_keys(df, all_scores_col)
    print(f"Found {len(all_keys)} unique score keys")
    
    # Extract scores for each row - build all columns at once
    print("Extracting scores for each row...")
    
    # Build a dict of all new columns
    new_columns = {}
    
    for key in sorted(all_keys):
        # Remove _mean suffix if present to get original key
        original_key = key.replace("_mean", "")
        
        values = []
        for idx, row in df.iterrows():
            if pd.notna(row.get(all_scores_col)):
                scores = extract_scores_from_json(str(row[all_scores_col]))
                
                if original_key in scores:
                    value = scores[original_key]
                    if isinstance(value, list) and len(value) > 0:
                        # For lists, compute mean if numeric
                        numeric_values = [v for v in value if isinstance(v, (int, float))]
                        values.append(float(sum(numeric_values) / len(numeric_values)) if numeric_values else None)
                    elif isinstance(value, (int, float, bool)):
                        values.append(float(value) if isinstance(value, bool) else value)
                    elif value is None:
                        values.append(None)
                    else:
                        values.append(None)
                else:
                    values.append(None)
            else:
                values.append(None)
        
        # Add column with prefix to avoid conflicts
        column_name = f"score_{key}" if not key.startswith("score_") else key
        new_columns[column_name] = values
    
    # Concatenate all new columns at once (more efficient)
    df_new_cols = pd.DataFrame(new_columns, index=df.index)
    df_expanded = pd.concat([df, df_new_cols], axis=1)
    
    return df_expanded


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Expand all_scores JSON column into separate columns"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: input file with _expanded suffix)"
    )
    parser.add_argument(
        "--drop-original",
        action="store_true",
        help="Drop the original all_scores column after expansion"
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_expanded.csv"
    
    print("=" * 80)
    print("EXPANDING MTEB SCORES COLUMN")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")
    
    # Expand all_scores column
    df_expanded = expand_all_scores_column(df)
    
    # Drop original column if requested
    if args.drop_original and "all_scores" in df_expanded.columns:
        df_expanded = df_expanded.drop(columns=["all_scores"])
        print("\nDropped original 'all_scores' column")
    
    # Save expanded CSV
    print(f"\nSaving expanded CSV...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_expanded.to_csv(output_path, index=False)
    
    print(f"✅ Saved expanded CSV to: {output_path}")
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"New columns: {len(df_expanded.columns)}")
    print(f"Added {len(df_expanded.columns) - len(df.columns)} score columns")
    
    # Show sample of new columns
    new_columns = [c for c in df_expanded.columns if c not in df.columns]
    print(f"\nSample of new columns (first 20):")
    for col in sorted(new_columns)[:20]:
        non_null = df_expanded[col].notna().sum()
        print(f"  {col:40s} ({non_null:4d} non-null values)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
