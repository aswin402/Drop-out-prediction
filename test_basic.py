#!/usr/bin/env python3
"""
Basic test to check if we can create sample data without external dependencies
"""

import csv
import random

def create_simple_test_data():
    """Create a simple test dataset"""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create sample data
    data = []
    headers = ['student_id', 'Age', 'Study_Time', 'Failures', 'G1', 'G2', 'G3', 'Dropped_Out']
    
    for i in range(100):
        student_id = i + 1
        age = random.randint(15, 22)
        study_time = random.randint(1, 4)
        failures = random.randint(0, 3)
        g1 = random.randint(0, 20)
        g2 = random.randint(0, 20)
        g3 = random.randint(0, 20)
        
        # Simple logic: more failures and lower grades = higher dropout chance
        dropout_chance = 0.1  # base chance
        if failures >= 2:
            dropout_chance += 0.4
        if g3 < 10:
            dropout_chance += 0.3
        if study_time <= 1:
            dropout_chance += 0.2
            
        dropped_out = 1 if random.random() < dropout_chance else 0
        
        data.append([student_id, age, study_time, failures, g1, g2, g3, dropped_out])
    
    # Write to CSV
    with open('simple_test_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(data)
    
    print("✅ Simple test data created successfully!")
    print(f"📁 File: simple_test_data.csv")
    print(f"📊 Records: {len(data)}")
    
    # Show some stats
    dropout_count = sum(1 for row in data if row[-1] == 1)
    print(f"🎯 Dropout rate: {dropout_count}/{len(data)} ({dropout_count/len(data)*100:.1f}%)")
    
    return True

if __name__ == "__main__":
    print("Creating simple test dataset...")
    create_simple_test_data()