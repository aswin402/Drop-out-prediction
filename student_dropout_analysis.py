#!/usr/bin/env python3
"""
Student Dropout Prediction Analysis
Fixed version with proper data handling and XGBoost compatibility
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def main():
    # 1. Load CSV - Using our simple test data
    print("Loading dataset...")
    try:
        df = pd.read_csv("simple_test_data.csv")  # Using our test data
    except FileNotFoundError:
        print("ERROR: CSV file not found!")
        print("Please update the file path in the script to point to your actual CSV file.")
        print("The file should contain student data with a 'Dropped_Out' target column.")
        return
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return

    print(f"Dataset loaded successfully! Shape: {df.shape}")

    # 2. Encode categorical columns (our simple data has no categorical columns)
    print("Checking for categorical features...")
    cat_cols = ['School','Gender','Address','Family_Size','Parental_Status',
                'Mother_Job','Father_Job','Reason_for_Choosing_School',
                'Guardian','School_Support','Family_Support','Extra_Paid_Class',
                'Extra_Curricular_Activities','Attended_Nursery',
                'Wants_Higher_Education','Internet_Access','In_Relationship']

    # Only encode columns that exist in the dataset
    existing_cat_cols = [col for col in cat_cols if col in df.columns]
    print(f"Found {len(existing_cat_cols)} categorical columns to encode")

    for col in existing_cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    if len(existing_cat_cols) == 0:
        print("No categorical columns found - using numerical data as-is")

    # 3. Separate features and target
    print("Preparing features and target...")
    X = df.drop(['student_id', 'Dropped_Out'], axis=1, errors='ignore')
    y = df['Dropped_Out'].astype(int)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # 4. Split dataset BEFORE applying SMOTE (to avoid leakage)
    print("Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 5. Apply SMOTE ONLY to training set
    print("Applying SMOTE to balance training data...")
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    print(f"After SMOTE - Training set size: {X_train.shape[0]}")

    # 6. Feature scaling
    print("Scaling features...")
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 7. Define models
    print("Initializing models...")
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.8, algorithm='SAMME', random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, reg_alpha=0.1, reg_lambda=1,
                                 eval_metric='logloss', learning_rate=0.1, random_state=42)
    }

    # 8. Training & Validation
    print("\n📊 Validation Results:")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        if name == 'XGBoost':
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)
        
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None

        print(f"\n{name} Validation Results:")
        print(f"Accuracy: {accuracy_score(y_val, val_pred):.3f}")
        if val_proba is not None:
            print(f"ROC-AUC Score: {roc_auc_score(y_val, val_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_pred))

    # 9. Final Test Evaluation
    print("\n🔍 Final Test Set Results:")
    print("=" * 50)
    
    for name, model in models.items():
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        print(f"\n{name} Test Results:")
        print(f"Accuracy: {accuracy_score(y_test, test_pred):.3f}")
        if test_proba is not None:
            print(f"ROC-AUC Score: {roc_auc_score(y_test, test_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))

        # Plot Confusion Matrix
        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"/home/aswin/programming/@python/project1/Drop-out-prediction/{name}_confusion_matrix.png")
        plt.show()

    print("\n✅ Analysis completed successfully!")
    print("Confusion matrix plots saved as PNG files.")

if __name__ == "__main__":
    main()