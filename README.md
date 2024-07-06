# Calorie Prediction Using Various Machine Learning Models

This project aims to predict the number of calories burned during physical activities using various machine-learning algorithms. We perform a comparative analysis to evaluate the performance of different models and identify the most accurate method for calorie prediction.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Models Evaluated](#models-evaluated)
  - [Linear Models](#linear-models)
  - [Tree-Based Models](#tree-based-models)
  - [Support Vector Regression](#support-vector-regression)
  - [K-Nearest Neighbors Regression](#k-nearest-neighbors-regression)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to predict the number of calories burned during various physical activities using features such as age, height, weight, duration, heart rate, and body temperature. We compare different machine learning algorithms to determine which model performs best in terms of accuracy and error metrics.

## Dataset
The dataset contains the following columns:
- `Age`
- `Height`
- `Weight`
- `Duration`
- `Heart_Rate`
- `Body_Temp`
- `Calories`

## Preprocessing
1. **Handling Missing Values**: Checked and handled any missing values in the dataset.
2. **Scaling**: Standardized the features to ensure they are on a comparable scale.
3. **Outlier Detection**: Identified and removed outliers using Z-score analysis.

## Exploratory Data Analysis (EDA)
To understand the data better, we performed the following EDA steps:
- **Distribution Plots**: Visualized the distribution of each feature.
- **Correlation Matrix**: Analyzed the correlation between features.
- **Pair Plots**: Visualized relationships between pairs of features.
- **Box Plots**: Identified the presence of outliers.

## Models Evaluated
We evaluated the performance of various machine learning models, categorized as follows:

### Linear Models
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**

### Tree-Based Models
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**

### Support Vector Regression
- **Support Vector Regressor (SVR)**

### K-Nearest Neighbors Regression
- **K-Nearest Neighbors (KNN) Regressor**

## Results
The performance of each model was evaluated using the following metrics:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R-squared (R2)**

| Model                     | MAE      | MSE       | R2     |
|---------------------------|----------|-----------|--------|
| Linear Regression         | 8.2910     | 126.12    | 0.9677 |
| Ridge Regression          | 8.29     | 126.12    | 0.9677 |
| Lasso Regression          | 9.05     | 151.14    | 0.9612 |
| Decision Tree             | 3.47     | 28.75     | 0.9926 |
| Random Forest             | 1.74     | 8.11      | 0.9979 |
| Gradient Boosting         | 2.55     | 11.65     | 0.9970 |
| XGBoost                   | 1.46     | 4.30      | 0.9989 |
| Support Vector Regression | 2.33     | 31.70     | 0.9919 |
| KNN Regression            | 3.80     | 27.09     | 0.9931 |

## Conclusion
From our analysis, **XGBoost** emerged as the best-performing model with the lowest MAE and MSE, and the highest R-squared value. Tree-based ensemble methods like **Random Forest** and **Gradient Boosting** also performed exceptionally well.

## Installation
To run this project, you need Python and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

## Video Demonstration
https://github.com/rohan12345a/ML_CompairetiveAnalysisCaloriesBurn/assets/109196424/5ca94d12-f7c4-4eac-894d-0e8fcd5dc782

You can install the required libraries using:
```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# For running streamlit file:
streamlit run calorie_dep.py



