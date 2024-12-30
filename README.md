# Sentiment Analysis using IMDB Dataset

This project implements a complete pipeline for text-based sentiment analysis on the IMDB Dataset. The process includes data cleaning, preprocessing, feature engineering, label encoding, and a comprehensive comparison of various classification algorithms.

## Table of Contents
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Feature Engineering](#feature-engineering)
- [Classification Algorithms](#classification-algorithms)
- [Results](#results)
- [Dependencies](#dependencies)

---

## Dataset
We use the IMDB Sentiment Analysis Dataset,which contains movie reviews labeled as positive or negative. 

---

## Preprocessing Steps
1. **Data Cleaning**: Removal of HTML tags, special characters, and conversion to lowercase.
2. **Text Tokenization**: Using `nltk` for tokenizing text data.
3. **Stopword Removal**: Removing common English stopwords to retain meaningful words.
4. **Stemming**: Using the Lancaster Stemmer to reduce words to their base form.

---

## Feature Engineering
1. **TF-IDF Vectorization**: Transforming text data into numerical format.
2. **Label Encoding**: Converting categorical labels to numerical form using `LabelEncoder`.

---

## Classification Algorithms
The following algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. Naive Bayes Classifier
6. Gradient Boosting Machines (GBM)
7. XGBoost Classifier
8. LightGBM Classifier
9. CatBoost Classifier

---

## Results
Each algorithm's performance was evaluated based on metrics such as accuracy, precision, recall, and F1-score. Among the evaluated models, **Support Vector Machine (SVM)** demonstrated the best performance, achieving the highest overall metrics across all categories. Following SVM, **Logistic Regression** emerged as the second-best performer, showcasing robust and reliable results in terms of accuracy, precision, recall, and F1-score. 

Detailed results, along with visualizations highlighting the performance of these algorithms, are provided in the project notebook. These insights underline the effectiveness of SVM for sentiment analysis tasks while also establishing Logistic Regression as a competitive and efficient alternative.

---

## Dependencies
- Python 3.7+
- pandas
- numpy
- nltk
- scikit-learn
- xgboost
- lightgbm
- catboost
- matplotlib
- seaborn

---

Feel free to contribute by opening issues or submitting pull requests. Happy coding!

