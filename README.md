# Adult Dataset - Exploratory Data Analysis and Classification

This project downloads the Adult dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult) and performs comprehensive exploratory data analysis (EDA) and builds classification models to predict whether an individual's income exceeds $50K/year.

## Dataset

The Adult dataset (also known as "Census Income" dataset) contains demographic and employment information from the 1994 Census database. The prediction task is to determine whether a person makes over $50K a year.

**Features include:**
- Age, workclass, education, marital status
- Occupation, relationship, race, sex
- Capital gain/loss, hours per week
- Native country

## Project Structure

```
adults-eda-and-classification/
├── README.md                          # Project documentation
├── requirements.txt                    # Python dependencies
├── adult_analysis.py                   # Python script for analysis
├── adult_eda_and_classification.ipynb # Jupyter notebook for interactive analysis
└── Generated outputs:
    ├── income_distribution.png         # Target variable visualization
    ├── correlation_matrix.png          # Feature correlation heatmap
    ├── model_comparison.png            # Model performance comparison
    ├── confusion_matrices.png          # Confusion matrices for all models
    └── roc_curves.png                  # ROC curves comparison
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abebeana/adults-eda-and-classification.git
cd adults-eda-and-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Python Script

Execute the complete analysis pipeline:
```bash
python adult_analysis.py
```

This will:
- Download the Adult dataset from UCI repository
- Perform exploratory data analysis
- Generate visualizations
- Train multiple classification models
- Compare model performance
- Save all results and visualizations

### Option 2: Use Jupyter Notebook

For interactive analysis:
```bash
jupyter notebook adult_eda_and_classification.ipynb
```

Run the cells sequentially to:
- Load and explore the dataset
- Visualize data distributions and relationships
- Preprocess the data
- Train and evaluate classification models
- Compare model performance

## Analysis Pipeline

### 1. Data Loading
- Downloads Adult dataset using `ucimlrepo` library
- Combines features and target into a single DataFrame

### 2. Exploratory Data Analysis
- Dataset overview (shape, types, missing values)
- Statistical summaries
- Target variable distribution
- Numerical feature distributions and relationships
- Categorical feature analysis
- Correlation analysis

### 3. Data Preprocessing
- Handle missing values
- Encode categorical variables using Label Encoding
- Split data into training (80%) and testing (20%) sets
- Feature scaling using StandardScaler

### 4. Model Training and Evaluation

Three classification models are trained and evaluated:

1. **Logistic Regression** - Baseline linear model
2. **Random Forest Classifier** - Ensemble learning with decision trees
3. **Gradient Boosting Classifier** - Advanced ensemble method

**Evaluation Metrics:**
- Accuracy
- ROC-AUC Score
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- ROC Curves

### 5. Model Comparison
- Side-by-side performance comparison
- Visualization of metrics
- Best model identification

## Results

The analysis typically shows:
- The dataset is imbalanced with more individuals earning ≤$50K
- Key predictive features include education, age, capital-gain, and occupation
- All models achieve good performance (accuracy > 80%)
- Tree-based ensemble methods (Random Forest, Gradient Boosting) typically outperform Logistic Regression
- ROC-AUC scores typically range from 0.87 to 0.92

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- ucimlrepo
- jupyter
- notebook

See `requirements.txt` for specific versions.

## License

This project is open source and available for educational purposes.

## Data Source

Dataset: [Adult Data Set](https://archive.ics.uci.edu/dataset/2/adult)  
Source: UCI Machine Learning Repository  
Donor: Ronny Kohavi and Barry Becker  
Date: May 1, 1996

## Future Enhancements

- Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Handle class imbalance with SMOTE or class weights
- Feature engineering and selection
- Cross-validation for more robust evaluation
- Deep learning models
- Model deployment as REST API
- Interactive web dashboard