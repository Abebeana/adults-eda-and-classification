# Usage Examples

This document provides detailed examples of how to use the Adult dataset analysis tools.

## Quick Start

### Option 1: Using the Python Script

The easiest way to run the complete analysis:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis (will attempt to download automatically)
python adult_analysis.py
```

If automatic download fails, manually download the dataset first:

```bash
# Download the dataset manually
python download_data.py

# Then run the analysis
python adult_analysis.py
```

### Option 2: Using the Jupyter Notebook

For interactive exploration and customization:

```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook

# Open adult_eda_and_classification.ipynb in your browser
# Run cells sequentially (Cell -> Run All or Shift+Enter for each cell)
```

## Manual Dataset Download

If you're behind a firewall or experiencing connection issues:

```bash
# Use the download utility
python download_data.py
```

Or download manually:

1. Visit: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
2. Download these files:
   - adult.data (training set)
   - adult.test (test set)
   - adult.names (metadata)
3. Place them in the project root directory

## Expected Output

### Console Output

The script will display:
- Dataset loading progress
- Data exploration statistics
- Model training progress
- Performance metrics for each model
- Best model identification

### Generated Files

The script creates several visualization files:

1. **income_distribution.png** - Bar chart of income categories
2. **correlation_matrix.png** - Heatmap of feature correlations
3. **model_comparison.png** - Side-by-side model performance
4. **confusion_matrices.png** - Confusion matrices for all models
5. **roc_curves.png** - ROC curves comparison

## Customization Examples

### Modifying the Python Script

#### Change the train/test split ratio:

```python
# In adult_analysis.py, in the preprocess_data() function, modify:
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # Changed from 0.2 to 0.3 (30% test set)
    random_state=42, stratify=y
)
```

#### Add more models:

```python
# After the Gradient Boosting section, add:
from sklearn.svm import SVC

print("\n4. Training Support Vector Machine...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_proba_svm = svm_model.predict_proba(X_test_scaled)[:, 1]

print("\nSVM Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_svm):.4f}")
```

#### Tune hyperparameters:

```python
# For Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,      # Increase trees
    max_depth=20,          # Limit tree depth
    min_samples_split=10,  # Require more samples to split
    random_state=42,
    n_jobs=-1
)
```

### Modifying the Jupyter Notebook

The notebook is fully interactive. You can:

1. **Add cells** - Click the "+" button or use `B` key (insert below) or `A` key (insert above)
2. **Modify visualizations** - Change colors, sizes, or chart types in any plotting cell
3. **Add custom analysis** - Insert new cells with your own exploratory questions
4. **Export results** - Use File > Download as > HTML/PDF to share results

## Common Issues and Solutions

### Issue: "ConnectionError: Error connecting to server"

**Solution**: The UCI repository may be temporarily unavailable. Use manual download:

```bash
python download_data.py
```

### Issue: "ModuleNotFoundError: No module named 'xyz'"

**Solution**: Install missing dependencies:

```bash
pip install -r requirements.txt
```

### Issue: Notebook kernel keeps dying

**Solution**: Large datasets may require more memory. Try:

1. Reduce the sample size for initial exploration
2. Use the Python script instead (more memory efficient)
3. Increase available RAM

### Issue: Plots not displaying in notebook

**Solution**: Add this at the start of the notebook:

```python
%matplotlib inline
```

## Advanced Usage

### Cross-Validation

Add cross-validation to the analysis:

```python
from sklearn.model_selection import cross_val_score

# Add after model training
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Hyperparameter Tuning

Use GridSearchCV for optimal hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
```

### Feature Selection

Identify and use only the most important features:

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 10 features
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected features: {selected_features}")
```

### Handling Class Imbalance

Use SMOTE to balance classes (requires imbalanced-learn package):

```bash
# First install the required package
pip install imbalanced-learn
```

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original training set: {X_train.shape}")
print(f"Balanced training set: {X_train_balanced.shape}")
```

## Performance Benchmarks

Expected performance on the full Adult dataset:

| Model | Accuracy | ROC-AUC | Training Time |
|-------|----------|---------|---------------|
| Logistic Regression | ~84% | ~87% | < 1 second |
| Random Forest | ~85% | ~90% | ~10 seconds |
| Gradient Boosting | ~87% | ~92% | ~30 seconds |

Note: Actual performance may vary based on preprocessing and hyperparameters.

## Contributing

If you'd like to extend this analysis:

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## Support

For questions or issues:
- Check the README.md for basic information
- Review this EXAMPLES.md for detailed usage
- Open an issue on GitHub for bugs or feature requests
