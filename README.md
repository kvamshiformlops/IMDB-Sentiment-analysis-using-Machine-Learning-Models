
# IMDB Movie Reviews Sentiment Analysis

A comprehensive comparison of various machine learning models for sentiment analysis on IMDB movie reviews using TF-IDF vectorization.

## 📋 Project Overview

This project implements sentiment analysis on IMDB movie reviews using multiple machine learning algorithms. The analysis includes text preprocessing, feature extraction using TF-IDF vectorization, and evaluation of different classifiers.

## 🔍 Features

- **Text preprocessing including:**
  - HTML tag removal
  - Non-alphabetical character removal
  - Lowercase conversion
  - Tokenization
  - Stopword removal
  - Lemmatization

- **TF-IDF Vectorization with parameters:**
  - `max_features=5000`
  - `ngram_range=(1,2)`
  - `min_df=3`
  - `max_df=0.85`
  - `sublinear_tf=True`

## 🛠️ Implementation Details

### Data Split
- Training set: 37,500 reviews
- Testing set: 12,500 reviews
- Balanced dataset with equal positive and negative reviews

### Models Implemented

Here's a brief overview of each model:

1. **[Logistic Regression](#logistic-regression)**: A statistical model that predicts binary outcomes (yes/no) based on one or more predictor variables.
2. **[Support Vector Machine (SVM)](#svm-c100)**: A supervised learning model that classifies data by finding the best hyperplane that separates all data points into categories.
3. **[Decision Tree](#decision-tree)**: A model that uses a tree-like structure of decisions and their possible consequences, which splits the data into branches to make predictions.
4. **[Nu-SVC](#nu-svc)**: A version of Support Vector Classifier that allows for the control of the number of support vectors and the margin errors.
5. **[Random Forest](#random-forest)**: An ensemble learning method that creates multiple decision trees and merges them together to get a more accurate and stable prediction.
6. **[Naive Bayes](#naive-bayes)**: A simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
7. **[Gradient Boosting](#gradient-boosting)**: An ensemble technique that builds models sequentially, each correcting the errors of its predecessor to improve accuracy.
8. **[SGD Classifier](#sgd-classifier)**: A classifier that uses stochastic gradient descent for optimizing the loss function, suitable for large-scale and sparse data.
9. **[Linear SVC](#linear-svc)**: A linear Support Vector Machine used for classification tasks that works by finding the hyperplane that best separates the data.
10. **[Neural Network (MLP)](#neural-network-mlp)**: A multi-layer perceptron that simulates the way the human brain processes information to recognize patterns and make decisions.
11. **[LightGBM](#lightgbm)**: A gradient boosting framework that uses tree-based learning algorithms, designed for speed and efficiency.
12. **[Extra Trees](#extra-trees)**: An ensemble learning method that creates a forest of randomized trees and averages their predictions for improved accuracy and control of overfitting.
13. **[AdaBoost](#adaboost)**: An ensemble method that combines multiple weak classifiers to form a strong classifier, by focusing on the errors of the previous classifiers.
14. **[Histogram Gradient Boosting](#histogram-gradient-boosting)**: An optimized version of gradient boosting that uses histograms to speed up the training process.
15. **[K-Nearest Neighbors](#k-nearest-neighbors)**: A non-parametric method that classifies data points based on the majority class among the nearest neighbors.
16. **[XGBoost](#xgboost)**: An advanced implementation of gradient boosting that focuses on speed and performance, often used in competitive machine learning.
17. **[CatBoost](#catboost)**: A gradient boosting library designed specifically for categorical features, aiming for high performance and efficiency.

## 📊 Results

Here are the accuracy scores and hyperparameters for all implemented models:

| Model                        | Accuracy | Hyperparameters |
|-----------------------------|----------|-----------------|
| [Nu-SVC](#nu-svc)                      | 89.57%   | nu=0.3, kernel='rbf', gamma='scale', coef0=0.0 |
| [SVM (C=100)](#svm-c100)                 | 89.40%   | kernel='rbf', C=100, gamma='scale' |
| [SVM (C=1.0)](#svm-c10)                 | 89.53%   | kernel='rbf', C=1.0, gamma='scale' |
| [Logistic Regression](#logistic-regression)         | 89.26%   | max_iter=1000, penalty='l2', C=1.0, solver='lbfgs' |
| [Linear SVC](#linear-svc)                  | 89.19%   | penalty='l2', loss='squared_hinge', C=0.1, max_iter=1000 |
| [SGD (hinge loss)](#sgd-classifier)           | 89.05%   | loss='hinge', penalty='elasticnet', max_iter=1000, learning_rate='adaptive', eta0=0.01 |
| [SGD (log loss)](#sgd-classifier)             | 88.09%   | loss='log_loss', penalty='elasticnet', max_iter=1000, learning_rate='adaptive', eta0=0.01 |
| [Neural Network (MLP)](#neural-network-mlp)        | 87.38%   | hidden_layer_sizes=(125,), activation='relu', solver='adam', learning_rate='adaptive', nesterovs_momentum=True |
| [LightGBM](#lightgbm)                    | 87.01%   | n_estimators=150, max_depth=50 |
| [Naive Bayes](#naive-bayes)                 | 85.89%   | Default MultinomialNB parameters |
| [Histogram Gradient Boosting](#histogram-gradient-boosting) | 85.71%   | loss='log_loss', max_depth=25 |
| [Random Forest](#random-forest)               | 84.92%   | n_estimators=100, random_state=42 |
| [CatBoost](#catboost)                    | 84.26%   | iterations=100, learning_rate=0.1, depth=8 |
| [XGBoost](#xgboost)                     | 84.23%   | n_estimators=100, learning_rate=0.1, max_depth=10 |
| [Gradient Boosting](#gradient-boosting)           | 83.74%   | n_estimators=100, max_depth=10 |
| [K-Nearest Neighbors](#k-nearest-neighbors)         | 81.34%   | n_neighbors=15 |
| [Decision Tree](#decision-tree)               | 73.22%   | max_depth=10 |
| [Extra Trees](#extra-trees)                 | 65.35%   | max_depth=20, splitter='best' |
| [AdaBoost](#adaboost)                    | 65.17%   | learning_rate=0.1 |

## 🏆 Key Findings

1. Support Vector Machine variants ([Nu-SVC](#nu-svc), [SVM](#svm-c100), [SVM (C=1.0)](#svm-c10)) consistently performed the best, achieving accuracies around 89.5%
2. Linear models ([Logistic Regression](#logistic-regression), [Linear SVC](#linear-svc)) showed strong performance with accuracies above 89%
3. Tree-based models showed varying performance:
   - Ensemble methods ([Random Forest](#random-forest), [XGBoost](#xgboost), [CatBoost](#catboost)) achieved decent results (84-85%)
   - Single [Decision Tree](#decision-tree) and [Extra Trees](#extra-trees) performed relatively poorly (65-73%)
4. [Neural Network (MLP)](#neural-network-mlp) showed good performance with 87.38% accuracy

## 🔧 Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- lightgbm
- xgboost
- catboost
- seaborn
- matplotlib

## 📈 Future Improvements

1. Implement cross-validation for more robust model evaluation
2. Experiment with different text preprocessing techniques
3. Try different vectorization methods (Word2Vec, BERT embeddings)
4. Implement hyperparameter tuning for better model performance
5. Add model ensemble techniques
6. Implement deep learning models using transformers

## 📜 License

This project is licensed under the MIT License - see the LICENSE.md file for details.
