"""
Adult Dataset - Exploratory Data Analysis and Classification

This script downloads the Adult dataset from the UCI Machine Learning Repository,
performs exploratory data analysis, and builds classification models to predict
whether income exceeds $50K/year.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings

# Try to import ucimlrepo, but don't fail if not available
try:
    from ucimlrepo import fetch_ucirepo
    UCIMLREPO_AVAILABLE = True
except ImportError:
    UCIMLREPO_AVAILABLE = False
    print("Warning: ucimlrepo not available. Will try to load from local files.")

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Load the Adult dataset from UCI repository or local files"""
    print("=" * 80)
    print("LOADING ADULT DATASET")
    print("=" * 80)
    
    # Try to load from ucimlrepo first
    try:
        if not UCIMLREPO_AVAILABLE:
            raise ImportError("ucimlrepo not available")
            
        print("\nAttempt 1: Downloading from UCI repository using ucimlrepo...")
        adult = fetch_ucirepo(id=2)
        X = adult.data.features
        y = adult.data.targets
        df = pd.concat([X, y], axis=1)
        print(f"✓ Dataset loaded successfully from UCI repository!")
    except Exception as e:
        print(f"✗ Failed to download from UCI repository: {e}")
        
        # Try to load from local file
        print("\nAttempt 2: Loading from local file (adult.data)...")
        try:
            # Define column names based on UCI documentation
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'income'
            ]
            
            if os.path.exists('adult.data'):
                df = pd.read_csv('adult.data', names=column_names, 
                                skipinitialspace=True, na_values=['?'])
                print(f"✓ Dataset loaded successfully from local file!")
            else:
                print(f"✗ Local file 'adult.data' not found.")
                print("\nPlease download the dataset first:")
                print("  Option 1: Run 'python download_data.py'")
                print("  Option 2: Download manually from:")
                print("  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
                raise FileNotFoundError("adult.data not found")
                
        except Exception as e2:
            print(f"✗ Failed to load from local file: {e2}")
            raise
    
    print(f"✓ Shape: {df.shape}")
    print(f"✓ Columns: {list(df.columns)}")
    
    return df


def explore_data(df):
    """Perform initial data exploration"""
    print("\n" + "=" * 80)
    print("DATA EXPLORATION")
    print("=" * 80)
    
    print("\n1. First 5 rows:")
    print(df.head())
    
    print("\n2. Dataset Info:")
    print(df.info())
    
    print("\n3. Statistical Summary:")
    print(df.describe())
    
    print("\n4. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    print(f"\n5. Number of duplicate rows: {df.duplicated().sum()}")
    
    print("\n6. Target Variable Distribution:")
    print(df['income'].value_counts())
    print("\nPercentage distribution:")
    print(df['income'].value_counts(normalize=True) * 100)
    
    return df


def visualize_data(df):
    """Create visualizations for EDA"""
    print("\n" + "=" * 80)
    print("DATA VISUALIZATION")
    print("=" * 80)
    
    # Target variable distribution
    plt.figure(figsize=(8, 6))
    df['income'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Income', fontsize=14, fontweight='bold')
    plt.xlabel('Income', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('income_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: income_distribution.png")
    plt.close()
    
    # Numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'income' in numerical_cols:
        numerical_cols.remove('income')
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_matrix.png")
    plt.close()
    
    print("✓ Visualizations completed!")


def preprocess_data(df):
    """Preprocess the data for modeling"""
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.dropna()
    print(f"✓ Shape after handling missing values: {df_processed.shape}")
    
    # Identify categorical and numerical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df_processed['income'] = le_target.fit_transform(df_processed['income'])
    
    print(f"✓ Encoding completed!")
    print(f"✓ Target classes: {le_target.classes_}")
    
    # Prepare features and target
    X = df_processed.drop('income', axis=1)
    y = df_processed['income']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Training set size: {X_train.shape[0]}")
    print(f"✓ Testing set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Feature scaling completed!")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, le_target


def train_and_evaluate_models(X_train, X_test, y_train, y_test, 
                              X_train_scaled, X_test_scaled, le_target):
    """Train and evaluate classification models"""
    print("\n" + "=" * 80)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 80)
    
    results = {}
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nLogistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr, target_names=le_target.classes_))
    
    results['Logistic Regression'] = {
        'model': lr_model,
        'predictions': y_pred_lr,
        'probabilities': y_pred_proba_lr,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
    }
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=le_target.classes_))
    
    results['Random Forest'] = {
        'model': rf_model,
        'predictions': y_pred_rf,
        'probabilities': y_pred_proba_rf,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
    }
    
    # 3. Gradient Boosting
    print("\n3. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_gb = gb_model.predict(X_test)
    y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    
    print("\nGradient Boosting Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_gb, target_names=le_target.classes_))
    
    results['Gradient Boosting'] = {
        'model': gb_model,
        'predictions': y_pred_gb,
        'probabilities': y_pred_proba_gb,
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
    }
    
    return results


def compare_models(results, y_test, le_target):
    """Compare model performance"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'ROC-AUC': [results[model]['roc_auc'] for model in results]
    })
    
    print("\nModel Performance Comparison:")
    print(comparison)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy comparison
    axes[0].bar(comparison['Model'], comparison['Accuracy'], 
                color=['skyblue', 'lightgreen', 'salmon'])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0.7, 0.9])
    axes[0].tick_params(axis='x', rotation=45)
    
    # ROC-AUC comparison
    axes[1].bar(comparison['Model'], comparison['ROC-AUC'], 
                color=['skyblue', 'lightgreen', 'salmon'])
    axes[1].set_title('Model ROC-AUC Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('ROC-AUC Score')
    axes[1].set_ylim([0.7, 0.95])
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_comparison.png")
    plt.close()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=le_target.classes_, yticklabels=le_target.classes_)
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices.png")
    plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
        auc = result['roc_auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curves.png")
    plt.close()
    
    # Determine best model
    best_model_name = comparison.loc[comparison['ROC-AUC'].idxmax(), 'Model']
    best_roc_auc = comparison.loc[comparison['ROC-AUC'].idxmax(), 'ROC-AUC']
    
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    print(f"{'=' * 80}")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("ADULT DATASET - EDA AND CLASSIFICATION")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Explore data
    df = explore_data(df)
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, le_target = preprocess_data(df)
    
    # Train and evaluate models
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, 
        X_train_scaled, X_test_scaled, le_target
    )
    
    # Compare models
    compare_models(results, y_test, le_target)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - income_distribution.png")
    print("  - correlation_matrix.png")
    print("  - model_comparison.png")
    print("  - confusion_matrices.png")
    print("  - roc_curves.png")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
