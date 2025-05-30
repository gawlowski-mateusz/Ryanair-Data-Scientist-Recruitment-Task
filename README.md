# Ryanair-Data-Scientist-Recruitment--Task

Regression model for predicting aircraft Take-Off Weight.

## Project Overview

Project tests machine learning models to accurately predict aircraft Take-Off Weight using historical flight data from Ryanair operations.

Model performance is evaluated using Root Mean Square Error (RMSE).

## Dataset

### Training Data

- **File:** `training.csv`
- **Content:** Historical flight data with comprehensive flight parameters and corresponding Take-Off Weight measurements
- **Purpose:** Model development and training

### Validation Data

- **File:** `validation.csv`
- **Content:** Flight data with identical feature structure but without Take-Off Weight values
- **Purpose:** Model validation

## Project Structure

```
├── training.csv             # Training dataset with TOW values
├── validation.csv           # Validation dataset for predictions
├── predictions_output.csv   # Validation dataset merged with predicted TOW results 
├── run_prediction.py        # Complete solution implementation
├── report.ipynb             # Jupyter notebook with analysis and methodology
├── report.html              # Converted to HTML Jupyter notebook if any trouble with Kernel
└── README.md                # Project documentation
```

## Files Description

### Core Implementation

- **`run_prediction.py`** - Complete solution containing data preprocessing, model training, validation, and prediction pipeline

### Output

- **`predictions_output.csv`** - Final predictions of Take-Off Weight for the validation dataset

### Documentation

- **`report.ipynb`** - Jupyter notebook with:
  - Methodology descriptions
  - Model development process
  - Data exploration and visualization
  - Performance analysis with diagrams
  - Results interpretation

## Getting Started

1. **Clone the repository**

   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Run the prediction model**

   ```bash
   python run_prediction.py
   ```

4. **View detailed analysis**

   ```bash
   jupyter notebook report.ipynb
   ```

## Results

Model generates Take-Off Weight predictions saved in `predictions_output.csv`, with detailed performance metrics and analysis available in the `report.ipynb`.
