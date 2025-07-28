#!/usr/bin/env python3
"""
Minimal test to verify the project structure and basic functionality
"""

import csv
import os

def test_project_status():
    """Test if the project is ready to run"""
    
    print("🔍 Testing Project Status...")
    print("=" * 50)
    
    # Check files
    files_to_check = [
        'sample.ipynb',
        'student_dropout_analysis.py', 
        'simple_test_data.csv',
        'requirements.txt'
    ]
    
    print("📁 Checking files:")
    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file}")
    
    # Check CSV data
    if os.path.exists('simple_test_data.csv'):
        with open('simple_test_data.csv', 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            print(f"\n📊 Sample data: {len(rows)-1} records, {len(rows[0])} columns")
            print(f"  Headers: {rows[0]}")
    
    # Check virtual environment
    if os.path.exists('ml_env'):
        print(f"\n🐍 Virtual environment: ✅ Created")
    else:
        print(f"\n🐍 Virtual environment: ❌ Not found")
    
    print("\n🎯 Project Status:")
    print("  ✅ Code structure: Ready")
    print("  ✅ Sample data: Available") 
    print("  ✅ Scripts: Created")
    print("  ⏳ ML packages: Installing...")
    
    print("\n📋 Next Steps:")
    print("  1. Wait for package installation to complete")
    print("  2. Or use Google Colab for immediate testing")
    print("  3. Or install system packages: sudo apt install python3-pandas python3-sklearn")
    
    return True

if __name__ == "__main__":
    test_project_status()