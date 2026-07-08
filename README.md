# Car Price Regression Analysis

This project implements machine learning regression models to predict car selling prices based on various features like year, mileage, fuel type, and transmission.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Data Processing](#data-processing)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Overview

The goal of this project is to build and compare regression models that can accurately predict the selling price of cars. The notebook explores data preprocessing, encoding categorical variables, and training multiple regression algorithms.

## Dataset

**Source**: Car sales data (CSV file)  
**Samples**: 301 car records  
**Columns**: 9 features

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Car_Name | Categorical | Name of the car model |
| Year | Numerical | Manufacturing year |
| Selling_Price | Numerical | Target variable - selling price |
| Present_Price | Numerical | Current market price |
| Kms_Driven | Numerical | Total kilometers driven |
| Fuel_Type | Categorical | Type of fuel (Petrol, Diesel, CNG) |
| Seller_Type | Categorical | Type of seller (Dealer, Individual) |
| Transmission | Categorical | Type of transmission (Manual, Automatic) |
| Owner | Numerical | Number of previous owners |

## Data Processing

### 1. Data Loading & Exploration
- Loaded car dataset using pandas
- Checked dataset shape: (301, 9)
- Verified data types and missing values
- No missing values found in the dataset

### 2. Categorical Data Analysis
- **Fuel_Type**: Petrol (239), Diesel (60), CNG (2)
- **Seller_Type**: Dealer (195), Individual (106)
- **Transmission**: Manual (261), Automatic (40)

### 3. Encoding Categorical Variables
Categorical features were encoded to numerical values:
- **Fuel_Type**: Petrol → 0, Diesel → 1, CNG → 2
- **Seller_Type**: Dealer → 0, Individual → 1
- **Transmission**: Manual → 0, Automatic → 1

### 4. Feature-Target Split
- **Features (X)**: Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
- **Target (Y)**: Selling_Price
- **Car_Name**: Dropped (not useful for prediction)

### 5. Train-Test Split
- Training set: 90% (271 samples)
- Test set: 10% (30 samples)
- Random state: 2 (for reproducibility)

## Models

### 1. Linear Regression
A foundational regression model that assumes a linear relationship between features and the target variable.

### 2. Lasso Regression
An L1-regularized regression model that performs feature selection and helps prevent overfitting.

## Installation

### Requirements
- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/ridhima-aga/regressionML.git
cd regressionML

# Install dependencies
pip install pandas matplotlib seaborn scikit-learn
```

## Usage

### Running the Notebook

```bash
# Open Jupyter Notebook
jupyter notebook Regression.ipynb
```

Or use Google Colab by uploading the notebook directly.

### Key Imports

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
```

## File Structure

```
regressionML/
├── README.md                 # Project documentation
├── Regression.ipynb         # Main notebook with analysis and models
└── car data.csv             # Dataset (required for running notebook)
```

## Notes

- The notebook includes deprecation warnings related to pandas `replace()` function. These can be suppressed with appropriate dtype specifications.
- Ensure the CSV file path is correct when loading data locally.
- Random state is set to 2 for reproducible results.

## Future Improvements

- Implement additional regression models (Ridge, ElasticNet, SVR)
- Perform hyperparameter tuning
- Add feature scaling/normalization
- Create cross-validation analysis
- Build visualization dashboards for predictions
- Implement ensemble methods

## License

Open source - feel free to use and modify for educational purposes.

---

**Author**: Ridhima Aga  
**Last Updated**: 2026
