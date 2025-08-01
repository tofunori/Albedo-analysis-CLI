#!/usr/bin/env python3
"""
Diagnostic script to check what methods are available in each glacier's data files.
This helps identify why some plots are showing "No data available".
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration - same as main script
CONFIG = {
    'data_paths': {
        'athabasca': {
            'modis': "D:/Documents/Projects/athabasca_analysis/data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv",
            'aws': "D:/Documents/Projects/athabasca_analysis/data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
        },
        'haig': {
            'modis': "D:/Documents/Projects/Haig_analysis/data/csv/Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv",
            'aws': "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv"
        },
        'coropuna': {
            'modis': "D:/Documents/Projects/Coropuna_glacier/data/csv/coropuna_glacier_2014-01-01_to_2025-01-01.csv",
            'aws': "D:/Documents/Projects/Coropuna_glacier/data/csv/COROPUNA_simple.csv"
        }
    }
}

def check_glacier_data(glacier_id):
    """Check what methods are available in a glacier's MODIS data."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC CHECK: {glacier_id.upper()} GLACIER")
    print(f"{'='*60}")
    
    modis_path = CONFIG['data_paths'][glacier_id]['modis']
    aws_path = CONFIG['data_paths'][glacier_id]['aws']
    
    # Check if files exist
    if not Path(modis_path).exists():
        print(f"❌ MODIS file not found: {modis_path}")
        return
    if not Path(aws_path).exists():
        print(f"❌ AWS file not found: {aws_path}")
        return
    
    print(f"✅ Files found")
    
    # Load MODIS data
    try:
        modis_data = pd.read_csv(modis_path)
        print(f"✅ MODIS data loaded: {len(modis_data):,} rows")
        
        # Show column names
        print(f"\n📊 MODIS Columns ({len(modis_data.columns)} total):")
        for i, col in enumerate(modis_data.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Check for method column
        if 'method' in modis_data.columns:
            print(f"\n🎯 Found 'method' column - data is in LONG format")
            unique_methods = modis_data['method'].unique()
            print(f"   Available methods: {list(unique_methods)}")
            
            # Count observations per method
            method_counts = modis_data['method'].value_counts()
            print(f"\n📈 Observations per method:")
            for method, count in method_counts.items():
                print(f"   {method}: {count:,} observations")
        else:
            print(f"\n🎯 No 'method' column found - data is in WIDE format")
            
            # Look for method-specific columns
            method_patterns = ['MCD43A3', 'MOD09GA', 'MOD10A1', 'MYD09GA', 'MYD10A1']
            found_methods = []
            
            for pattern in method_patterns:
                matching_cols = [col for col in modis_data.columns if pattern.upper() in col.upper()]
                if matching_cols:
                    found_methods.append(pattern)
                    print(f"   ✅ {pattern}: {matching_cols}")
                    
                    # Count non-null values for albedo columns
                    albedo_cols = [col for col in matching_cols if 'albedo' in col.lower()]
                    for albedo_col in albedo_cols:
                        non_null_count = modis_data[albedo_col].notna().sum()
                        print(f"      └─ {albedo_col}: {non_null_count:,} non-null values")
                else:
                    print(f"   ❌ {pattern}: No matching columns")
            
            if found_methods:
                print(f"\n🎯 Found methods in wide format: {found_methods}")
            else:
                print(f"\n❌ No standard method patterns found!")
        
        # Check date range
        if 'date' in modis_data.columns:
            modis_data['date'] = pd.to_datetime(modis_data['date'])
            date_range = f"{modis_data['date'].min().date()} to {modis_data['date'].max().date()}"
            print(f"\n📅 MODIS Date range: {date_range}")
        
    except Exception as e:
        print(f"❌ Error loading MODIS data: {e}")
        return
    
    # Load AWS data
    try:
        if glacier_id == 'haig':
            aws_data = pd.read_csv(aws_path, sep=';', skiprows=6, decimal=',')
            aws_data.columns = aws_data.columns.str.strip()
        else:
            aws_data = pd.read_csv(aws_path)
        
        print(f"✅ AWS data loaded: {len(aws_data):,} rows")
        
        # Show AWS columns
        print(f"\n📊 AWS Columns ({len(aws_data.columns)} total):")
        for i, col in enumerate(aws_data.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Find albedo column
        albedo_cols = [col for col in aws_data.columns if 'albedo' in col.lower() or col == 'Albedo']
        if albedo_cols:
            albedo_col = albedo_cols[0]
            non_null_aws = aws_data[albedo_col].notna().sum()
            print(f"\n🎯 AWS Albedo column: '{albedo_col}' ({non_null_aws:,} non-null values)")
        else:
            print(f"\n❌ No AWS albedo column found!")
        
        # Check AWS date format
        date_cols = [col for col in aws_data.columns if any(word in col.lower() for word in ['time', 'date', 'timestamp'])]
        if date_cols:
            date_col = date_cols[0]
            print(f"🎯 AWS Date column: '{date_col}'")
            if glacier_id == 'haig':
                if 'Year' in aws_data.columns and 'Day' in aws_data.columns:
                    year_range = f"{aws_data['Year'].min():.0f} to {aws_data['Year'].max():.0f}"
                    print(f"📅 AWS Date range: {year_range} (Year/Day format)")
            else:
                try:
                    aws_data['date_parsed'] = pd.to_datetime(aws_data[date_col])
                    date_range = f"{aws_data['date_parsed'].min().date()} to {aws_data['date_parsed'].max().date()}"
                    print(f"📅 AWS Date range: {date_range}")
                except:
                    print(f"❌ Could not parse AWS dates from '{date_col}'")
        
    except Exception as e:
        print(f"❌ Error loading AWS data: {e}")
        return

def main():
    """Run diagnostic checks for all glaciers."""
    print("🔍 GLACIER DATA DIAGNOSTIC TOOL")
    print("=" * 60)
    print("This tool checks what methods are available in each glacier's data files")
    print("to help understand why some plots show 'No data available'.")
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        check_glacier_data(glacier_id)
    
    print(f"\n{'='*60}")
    print("📋 SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}")
    print("Based on the diagnostic results above:")
    print("1. Check which methods actually have data in each glacier's files")
    print("2. Verify that column names match expected patterns")
    print("3. Ensure AWS and MODIS date ranges overlap")
    print("4. Consider updating the method list in CONFIG to match available data")
    print("\nIf only some methods have data, that's normal - not all glaciers")
    print("have observations from all MODIS products for all time periods.")

if __name__ == "__main__":
    main()