# Logistic Regression for Medical Transcripts

## Overview
This project applies logistic regression to classify medical transcripts. The dataset (`mtsamples.csv`) contains various medical notes, and the goal is to build a predictive model using text preprocessing and machine learning techniques.

## Dataset
- **Source:** `mtsamples.csv`
- **Content:** Medical transcripts with associated categories.

## Features & Methodology
1. **Text Preprocessing:**
   - Tokenization (word and sentence)
   - Stopword removal
   - Lemmatization
   - TF-IDF vectorization

2. **Data Splitting & Resampling:**
   - Train-test split
   - Handling class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique)

3. **Modeling:**
   - Logistic Regression for classification
   - Evaluation using accuracy, confusion matrix, and classification report

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas scikit-learn imbalanced-learn nltk
```

## Usage
Run the Jupyter Notebook step by step:
```bash
jupyter notebook LogisticRegressionMedicalTranscripts.ipynb
```

## Results
- The notebook evaluates the performance of logistic regression for classifying medical transcripts.
- Metrics like accuracy, precision, recall, and F1-score are analyzed.
