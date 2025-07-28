# Quick Start Guide - Student Dropout Prediction

## 🚀 Quick Start (3 steps)

### Step 1: Get Sample Data
```bash
cd /home/aswin/programming/@python/project1/Drop-out-prediction
python3 create_sample_data.py
```

### Step 2: Update the notebook
Open `sample.ipynb` and change the CSV path from:
```python
df = pd.read_csv(r"C:\Users\hyazi\Downloads\archive (1)\student dropout.csv")
```
to:
```python
df = pd.read_csv("sample_student_data.csv")
```

### Step 3: Run the analysis
Choose one of these methods:

**Method A: Google Colab (Easiest)**
1. Go to https://colab.research.google.com/
2. Upload `sample.ipynb`
3. Upload `sample_student_data.csv`
4. Run all cells

**Method B: Local Jupyter**
```bash
pip3 install pandas scikit-learn xgboost imbalanced-learn seaborn matplotlib jupyter
jupyter notebook
```

**Method C: Python Script**
```bash
# Update the path in student_dropout_analysis.py first
python3 student_dropout_analysis.py
```

## 📊 What the Code Does

1. **Data Loading**: Loads student data from CSV
2. **Data Preprocessing**: 
   - Encodes categorical variables
   - Splits data into train/validation/test sets
   - Applies SMOTE for class balancing
   - Scales features
3. **Model Training**: Trains 5 different models:
   - Decision Tree
   - SVM
   - Bagging Classifier
   - AdaBoost
   - XGBoost
4. **Evaluation**: 
   - Shows accuracy and ROC-AUC scores
   - Generates classification reports
   - Creates confusion matrices

## 🔧 Troubleshooting

**If you get "ModuleNotFoundError":**
```bash
pip3 install [missing_package_name]
```

**If you get "FileNotFoundError":**
- Make sure the CSV file path is correct
- Use forward slashes (/) in file paths
- Check that the file exists in the specified location

**If XGBoost gives errors:**
- The code has been fixed to work with newer XGBoost versions
- Make sure you're using the updated version of the notebook

## 📈 Expected Results

The analysis will show:
- Model performance comparisons
- Validation and test accuracies
- ROC-AUC scores for each model
- Confusion matrices
- Classification reports with precision, recall, F1-scores

## 🎯 Next Steps

After running the analysis:
1. Compare model performances
2. Analyze which features are most important
3. Try hyperparameter tuning
4. Experiment with different preprocessing techniques
5. Add more sophisticated evaluation metrics