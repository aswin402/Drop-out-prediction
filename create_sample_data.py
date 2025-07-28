#!/usr/bin/env python3
"""
Create a sample student dropout dataset for testing
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def create_sample_dataset():
    """Create a synthetic student dropout dataset"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base features using sklearn
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )
    
    # Create realistic column names and data
    data = {
        'student_id': range(1, 1001),
        'School': np.random.choice(['GP', 'MS'], 1000),
        'Gender': np.random.choice(['M', 'F'], 1000),
        'Age': np.random.randint(15, 23, 1000),
        'Address': np.random.choice(['U', 'R'], 1000),
        'Family_Size': np.random.choice(['LE3', 'GT3'], 1000),
        'Parental_Status': np.random.choice(['T', 'A'], 1000),
        'Mother_Job': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], 1000),
        'Father_Job': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], 1000),
        'Reason_for_Choosing_School': np.random.choice(['home', 'reputation', 'course', 'other'], 1000),
        'Guardian': np.random.choice(['mother', 'father', 'other'], 1000),
        'Travel_Time': np.random.randint(1, 5, 1000),
        'Study_Time': np.random.randint(1, 5, 1000),
        'Failures': np.random.randint(0, 4, 1000),
        'School_Support': np.random.choice(['yes', 'no'], 1000),
        'Family_Support': np.random.choice(['yes', 'no'], 1000),
        'Extra_Paid_Class': np.random.choice(['yes', 'no'], 1000),
        'Extra_Curricular_Activities': np.random.choice(['yes', 'no'], 1000),
        'Attended_Nursery': np.random.choice(['yes', 'no'], 1000),
        'Wants_Higher_Education': np.random.choice(['yes', 'no'], 1000),
        'Internet_Access': np.random.choice(['yes', 'no'], 1000),
        'In_Relationship': np.random.choice(['yes', 'no'], 1000),
        'Free_Time': np.random.randint(1, 6, 1000),
        'Going_Out': np.random.randint(1, 6, 1000),
        'Daily_Alcohol': np.random.randint(1, 6, 1000),
        'Weekend_Alcohol': np.random.randint(1, 6, 1000),
        'Health': np.random.randint(1, 6, 1000),
        'Absences': np.random.randint(0, 94, 1000),
        'G1': np.random.randint(0, 21, 1000),
        'G2': np.random.randint(0, 21, 1000),
        'G3': np.random.randint(0, 21, 1000),
        'Dropped_Out': y  # Target variable
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlation between features and target
    # Students with more failures are more likely to drop out
    mask = df['Failures'] >= 2
    df.loc[mask, 'Dropped_Out'] = np.random.choice([0, 1], sum(mask), p=[0.3, 0.7])
    
    # Students with lower grades are more likely to drop out
    mask = df['G3'] < 10
    df.loc[mask, 'Dropped_Out'] = np.random.choice([0, 1], sum(mask), p=[0.4, 0.6])
    
    # Students without family support are more likely to drop out
    mask = df['Family_Support'] == 'no'
    df.loc[mask, 'Dropped_Out'] = np.random.choice([0, 1], sum(mask), p=[0.6, 0.4])
    
    return df

if __name__ == "__main__":
    print("Creating sample student dropout dataset...")
    
    # Create the dataset
    df = create_sample_dataset()
    
    # Save to CSV
    output_path = "/home/aswin/programming/@python/project1/Drop-out-prediction/sample_student_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ Sample dataset created successfully!")
    print(f"📁 Saved to: {output_path}")
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target distribution:")
    print(df['Dropped_Out'].value_counts())
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n🔍 First few rows:")
    print(df.head())