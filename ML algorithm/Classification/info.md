# Titanic ML Predictor

This repository demonstrates a full machine learning workflow for **Titanic survival prediction**, including data preprocessing, model training, hyperparameter tuning, evaluation, and custom predictions.

## Features

- **Data Preprocessing**: Handling categorical features, feature scaling, and feature selection with `SelectKBest`.
- **Model Training**: Implementation of multiple classifiers such as Decision Tree, SVM, K-Nearest Neighbors, and Naive Bayes.
- **Hyperparameter Tuning**: Using `GridSearchCV` to optimize model parameters for maximum accuracy.
- **Evaluation**: Accuracy scores and classification reports for all models.
- **Custom Predictions**: Allows users to input passenger details (`Pclass`, `Sex`, `Fare`) and get survival predictions from all trained models.

## Technologies & Libraries

- Python 3.13.7
- pandas, numpy  
- scikit-learn  
- seaborn (optional, for visualization)

## Key Features

- **Top Feature Selection**: Uses chi-square (`SelectKBest`) to select the most important features.  
- **Multiple ML Algorithms**: Supports Decision Tree, SVM, K-Nearest Neighbors, and Naive Bayes in one pipeline.  
- **Automated Hyperparameter Tuning**: Optimizes model parameters using `GridSearchCV`.  
- **User-Friendly Custom Predictions**: Enter passenger details to get real-time predictions from trained models.  

## Algorithm
## Algorithms Used

This project implements the following supervised machine learning algorithms for Titanic survival prediction:

1. **Decision Tree (DT)**  
   - Tree-based model that splits data based on feature values.  
   - **Hyperparameters:** `criterion` (gini/entropy), `max_depth`  
   - **Pros:** Easy to interpret; handles numerical and categorical data.

2. **Support Vector Machine (SVM)**  
   - Finds the hyperplane that best separates classes in feature space.  
   - **Hyperparameters:** `kernel` (linear, poly, rbf, sigmoid), `C`  
   - **Pros:** Works well in high-dimensional spaces; effective for classification.

3. **K-Nearest Neighbors (KNN)**  
   - Classifies a sample based on the majority class of its `k` nearest neighbors.  
   - **Hyperparameters:** `n_neighbors`, `weights` (uniform/distance)  
   - **Pros:** Simple, non-parametric; good for small datasets.

4. **Naive Bayes (NB)**  
   - Probabilistic classifier based on Bayes’ theorem assuming feature independence.  
   - **Hyperparameters:** Depends on variant (GaussianNB, CategoricalNB)  
   - **Pros:** Fast; handles categorical data well; produces probabilistic output.
