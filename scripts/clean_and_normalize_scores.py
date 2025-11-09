"""
Data Quality Check and Fix for the Scored Municipalities Dataset

This script identifies and fixes data quality issues like:
- Percentages exceeding 100%
- Outliers in scores
- Duplicate municipalities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse as ap


def check_and_fix_data_quality(input_path, output_path):
    """
    Check and fix data quality issues in the scored dataset.
    """
    print("\n" + "="*80)
    print("DATA QUALITY CHECK AND FIX")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records")
    
    # 1. Fix percentage issues
    print("\nFixing percentage calculations...")
    
    # Percentages should be between 0 and 1
    if df['perc_over_65'].max() > 1:
        print("  Normalizing percentages (detected values > 1)")
        df['perc_over_65'] = df['perc_over_65'] / 100
        if 'perc_under_30' in df.columns:
            df['perc_under_30'] = df['perc_under_30'] / 100
        if 'perc_under_35' in df.columns:
            df['perc_under_35'] = df['perc_under_35'] / 100
    
    # Cap percentages at realistic values
    df['perc_over_65'] = df['perc_over_65'].clip(0, 0.5)
    if 'perc_under_30' in df.columns:
        df['perc_under_30'] = df['perc_under_30'].clip(0, 0.4)
    if 'perc_under_35' in df.columns:
        df['perc_under_35'] = df['perc_under_35'].clip(0, 0.5)
    
    # # 2. Check for duplicate municipalities
    # print("\nChecking for duplicates...")
    # duplicates = df[df.duplicated(subset=['municipio', 'provincia'], keep=False)]
    # if len(duplicates) > 0:
    #     print(f"  Found {len(duplicates)} duplicate records, keeping first occurrence")
    #     df = df.drop_duplicates(subset=['municipio', 'provincia'], keep='first')
    # else:
    #     print("  No duplicates found")
    
    # 3. Check for extreme outliers in scores
    print("\nChecking for extreme outliers...")
    
    for score_col in ['economic_score', 'social_score']:
        q99 = df[score_col].quantile(0.99)
        outliers = df[df[score_col] > q99]
        
        if len(outliers) > 0:
            print(f"  {score_col}: Capping {len(outliers)} values above 99th percentile ({q99:.2f})")
            df[score_col] = df[score_col].clip(upper=q99)
    
    # Recalculate total scores with fixed data
    print("\nRecalculating total scores with cleaned data...")
    
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        col_name = f'total_score_alpha_{int(alpha*100)}'
        df[col_name] = alpha * df['economic_score'] + (1 - alpha) * df['social_score']
    
    # 4. Add normalized scores (0-100 scale) for easier interpretation
    print("\nAdding normalized scores (0-100 scale)...")
    
    def normalize_to_100(series):
        """Normalize a series to 0-100 scale using min-max normalization"""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series * 0  # All zeros if no variance
        return ((series - min_val) / (max_val - min_val)) * 100
    
    df['economic_score_normalized'] = normalize_to_100(df['economic_score'])
    df['social_score_normalized'] = normalize_to_100(df['social_score'])
    df['total_score_alpha_50_normalized'] = normalize_to_100(df['total_score_alpha_50'])
    
    # 5. Add interpretive categories
    print("\nAdding interpretive categories...")
    
    # Economic score categories
    df['economic_category'] = pd.cut(
        df['economic_score_normalized'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Social score categories
    df['social_category'] = pd.cut(
        df['social_score_normalized'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # 6. Save cleaned data
    print(f"\nSaving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records with {len(df.columns)} columns")
    
    # 7. Generate summary report
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total municipalities: {len(df)}")
    # print(f"Provinces covered: {df['provincia'].nunique()}")
    print(f"\nScore Statistics (Normalized 0-100):")
    print(df[['economic_score_normalized', 'social_score_normalized', 'total_score_alpha_50_normalized']].describe().round(2))
    
    print("\nTop 10 by Economic Score:")
    top_econ = df.nlargest(10, 'economic_score_normalized')[
        ['municipio', 'poblacion_total', 'num_bancos', 'economic_score_normalized']
    ]
    print(top_econ.to_string(index=False))
    
    print("\nTop 10 by Social Score:")
    top_social = df.nlargest(10, 'social_score_normalized')[
        ['municipio', 'poblacion_total', 'num_bancos', 'social_score_normalized']
    ]
    print(top_social.to_string(index=False))
    
    print("\nTop 10 by Balanced Score (alpha=0.5):")
    top_balanced = df.nlargest(10, 'total_score_alpha_50_normalized')[
        ['municipio', 'poblacion_total', 'num_bancos', 'total_score_alpha_50_normalized']
    ]
    print(top_balanced.to_string(index=False))
    print("="*80)
    
    return df


if __name__ == "__main__":
    ap = ap.ArgumentParser(description="Clean and Normalize Scores in Municipalities Dataset")
    ap.add_argument("--input-path", type=str, required=True, help="Path to input CSV file with scored municipalities")
    ap.add_argument("--output-path", type=str, required=True, help="Path to output CSV file for cleaned municipalities")
    args = ap.parse_args()

    df = check_and_fix_data_quality(args.input_path, args.output_path)

    print(f"\nOutput file: {args.output_path}")
    print("Data quality check completed.\n")
