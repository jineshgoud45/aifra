"""
Example Usage of Universal FRA/SFRA Parser
SIH 2025 PS 25190

This script demonstrates how to use the parser with different vendor formats
and displays the normalized output along with IEC compliance check results.
"""

import pandas as pd
import numpy as np
from parser import UniversalFRAParser, parse_fra_file, FRAParserError
import os


def create_example_files():
    """Create sample data files for demonstration."""
    
    # Create sample directory
    os.makedirs('sample_data', exist_ok=True)
    
    # Sample 1: Omicron FRANEO CSV
    with open('sample_data/omicron_sample.csv', 'w') as f:
        f.write("""# Omicron FRANEO FRA Measurement
# Device: FRANEO 800
# Test ID: Transformer_001_H1-H2
# Date: 2025-01-15 10:30:00
# Impedance: 50 Ohm
Frequency (Hz),Magnitude (dB),Phase (deg)
20.0,-45.2,85.3
50.0,-43.2,80.0
100.0,-40.8,74.1
500.0,-30.8,53.7
1000.0,-24.8,42.3
5000.0,-8.4,14.0
10000.0,-1.6,3.5
50000.0,10.9,-13.3
100000.0,14.5,-17.6
500000.0,19.5,-23.2
1000000.0,20.5,-24.4
2000000.0,21.0,-25.1
""")
    
    # Sample 2: Doble SFRA
    with open('sample_data/doble_sample.txt', 'w') as f:
        f.write("""Doble Engineering SFRA Test Data
Test ID: XFMR_5678_Phase_B
Date: 01/16/2025
Impedance: 50 Ohms

Frequency (Hz)\tMagnitude (dB)\tPhase (deg)
20.0\t-44.9\t84.8
100.0\t-40.5\t73.2
1000.0\t-24.1\t41.8
10000.0\t-1.2\t3.1
100000.0\t14.8\t-17.9
1000000.0\t20.7\t-24.6
2000000.0\t21.2\t-25.3
""")
    
    # Sample 3: Megger FRAX
    with open('sample_data/megger_sample.dat', 'w') as f:
        f.write("""[MEGGER FRAX 101]
TestID=POWER_TRAFO_LV_001
DateTime=2025-01-16 15:30:00
Impedance=50
[MEASUREMENT_DATA]
Frequency (Hz),Magnitude (dB),Phase (deg)
20.0,-46.1,86.2
100.0,-40.8,74.3
1000.0,-24.5,42.1
10000.0,-1.5,3.4
100000.0,14.6,-17.7
1000000.0,20.6,-24.5
2000000.0,21.1,-25.2
""")
    
    print("✓ Sample data files created in 'sample_data/' directory\n")


def example_basic_usage():
    """Example 1: Basic parsing with vendor specification."""
    print("="*70)
    print("EXAMPLE 1: Basic Parsing with Vendor Specification")
    print("="*70)
    
    parser = UniversalFRAParser()
    
    # Parse Omicron file
    df = parser.parse_file('sample_data/omicron_sample.csv', vendor='omicron')
    
    print("\nParsed Omicron FRANEO CSV:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print()


def example_auto_detection():
    """Example 2: Auto-detection of vendor and format."""
    print("="*70)
    print("EXAMPLE 2: Auto-Detection of Vendor and Format")
    print("="*70)
    
    parser = UniversalFRAParser()
    
    files = [
        'sample_data/omicron_sample.csv',
        'sample_data/doble_sample.txt',
        'sample_data/megger_sample.dat'
    ]
    
    for filepath in files:
        df = parser.parse_file(filepath)  # No vendor specified
        print(f"\nFile: {filepath}")
        print(f"  Detected Vendor: {df['vendor'].iloc[0]}")
        print(f"  Test ID: {df['test_id'].iloc[0]}")
        print(f"  Data Points: {len(df)}")
        print(f"  Freq Range: {df['frequency_hz'].min():.1f} Hz to {df['frequency_hz'].max():.2e} Hz")
    print()


def example_iec_compliance():
    """Example 3: IEC 60076-18 compliance checking."""
    print("="*70)
    print("EXAMPLE 3: IEC 60076-18 Compliance Checking")
    print("="*70)
    
    parser = UniversalFRAParser()
    
    filepath = 'sample_data/omicron_sample.csv'
    df = parser.parse_file(filepath)
    
    # Get QA results
    qa_results = parser.get_qa_results(filepath)
    
    print(f"\nQA Results for: {qa_results['test_id']}")
    print(f"Vendor: {qa_results['vendor']}")
    print("\nCompliance Checks:")
    
    for check_name, check_data in qa_results['checks'].items():
        passed = "✓ PASS" if check_data.get('passed', False) else "✗ FAIL"
        print(f"\n  {check_name.upper()}: {passed}")
        
        if check_name == 'frequency_range':
            print(f"    Min Frequency: {check_data['min_freq']:.1f} Hz")
            print(f"    Max Frequency: {check_data['max_freq']:.2e} Hz")
            print(f"    IEC Range: {parser.IEC_FREQ_MIN} Hz - {parser.IEC_FREQ_MAX:.2e} Hz")
        
        elif check_name == 'frequency_grid':
            print(f"    Number of Points: {check_data['num_points']}")
            print(f"    Log Spacing Std: {check_data['log_spacing_std']:.3f}")
        
        elif check_name == 'magnitude_stats':
            print(f"    Mean: {check_data['mean_db']:.2f} dB")
            print(f"    Std Dev: {check_data['std_db']:.2f} dB")
            print(f"    Range: {check_data['min_db']:.2f} to {check_data['max_db']:.2f} dB")
        
        elif check_name == 'artifacts':
            print(f"    Artifacts Detected: {check_data['num_artifacts']}")
            print(f"    Threshold: {check_data['threshold_db']} dB")
            print(f"    Max Deviation: {check_data['max_deviation_db']:.2f} dB")
    print()


def example_data_analysis():
    """Example 4: Basic data analysis and visualization."""
    print("="*70)
    print("EXAMPLE 4: Basic Data Analysis")
    print("="*70)
    
    parser = UniversalFRAParser()
    
    # Parse multiple files for comparison
    files = {
        'Omicron': 'sample_data/omicron_sample.csv',
        'Doble': 'sample_data/doble_sample.txt',
        'Megger': 'sample_data/megger_sample.dat'
    }
    
    print("\nComparative Analysis:")
    print(f"{'Vendor':<15} {'Points':<10} {'Freq Range':<30} {'Mag Range':<20}")
    print("-" * 75)
    
    for name, filepath in files.items():
        df = parser.parse_file(filepath)
        
        freq_range = f"{df['frequency_hz'].min():.0f} - {df['frequency_hz'].max():.2e} Hz"
        mag_range = f"{df['magnitude_db'].min():.1f} to {df['magnitude_db'].max():.1f} dB"
        
        print(f"{name:<15} {len(df):<10} {freq_range:<30} {mag_range:<20}")
    
    print("\n\nFrequency Point Distribution (Omicron sample):")
    df = parser.parse_file('sample_data/omicron_sample.csv')
    
    # Analyze frequency distribution in log space
    freq_log = np.log10(df['frequency_hz'].values)
    print(f"  Log10 frequency range: {freq_log.min():.2f} to {freq_log.max():.2f}")
    print(f"  Mean log spacing: {np.mean(np.diff(freq_log)):.3f}")
    print(f"  Median log spacing: {np.median(np.diff(freq_log)):.3f}")
    print()


def example_error_handling():
    """Example 5: Error handling demonstration."""
    print("="*70)
    print("EXAMPLE 5: Error Handling")
    print("="*70)
    
    parser = UniversalFRAParser()
    
    # Test 1: Missing file
    print("\n1. Handling missing file:")
    try:
        df = parser.parse_file('nonexistent_file.csv')
    except FileNotFoundError as e:
        print(f"   ✓ Caught FileNotFoundError: {e}")
    
    # Test 2: Invalid data file
    print("\n2. Handling invalid data format:")
    with open('sample_data/invalid.csv', 'w') as f:
        f.write("This is not valid FRA data\nJust random text\n")
    
    try:
        df = parser.parse_file('sample_data/invalid.csv', vendor='omicron')
    except FRAParserError as e:
        print(f"   ✓ Caught FRAParserError: {e}")
    
    # Clean up
    os.remove('sample_data/invalid.csv')
    print()


def example_convenience_function():
    """Example 6: Using the convenience function."""
    print("="*70)
    print("EXAMPLE 6: Convenience Function (parse_fra_file)")
    print("="*70)
    
    # Quick one-liner parsing
    df = parse_fra_file('sample_data/omicron_sample.csv')
    
    print("\nUsing parse_fra_file() convenience function:")
    print(f"  Result type: {type(df)}")
    print(f"  Shape: {df.shape}")
    print(f"  Vendor: {df['vendor'].iloc[0]}")
    print(f"  Test ID: {df['test_id'].iloc[0]}")
    print("\nThis is equivalent to:")
    print("  parser = UniversalFRAParser()")
    print("  df = parser.parse_file(filepath)")
    print()


def example_batch_processing():
    """Example 7: Batch processing multiple files."""
    print("="*70)
    print("EXAMPLE 7: Batch Processing Multiple Files")
    print("="*70)
    
    parser = UniversalFRAParser()
    
    # Get all sample files
    sample_files = [
        'sample_data/omicron_sample.csv',
        'sample_data/doble_sample.txt',
        'sample_data/megger_sample.dat'
    ]
    
    # Parse all files
    all_data = []
    for filepath in sample_files:
        try:
            df = parser.parse_file(filepath)
            all_data.append(df)
            print(f"✓ Parsed: {filepath}")
        except Exception as e:
            print(f"✗ Failed: {filepath} - {e}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Vendors: {combined_df['vendor'].unique()}")
    print(f"  Test IDs: {combined_df['test_id'].unique()}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(combined_df[['frequency_hz', 'magnitude_db', 'phase_deg']].describe())
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  Universal FRA/SFRA Parser - Usage Examples".center(68) + "║")
    print("║" + "  SIH 2025 PS 25190".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print()
    
    # Create sample files
    create_example_files()
    
    # Run examples
    try:
        example_basic_usage()
        example_auto_detection()
        example_iec_compliance()
        example_data_analysis()
        example_error_handling()
        example_convenience_function()
        example_batch_processing()
        
        print("="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Review the sample_data/ directory for example files")
        print("  2. Try parsing your own FRA/SFRA data files")
        print("  3. Run unit tests: python test_parser.py")
        print("  4. Check IEC compliance results for your data")
        print()
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
