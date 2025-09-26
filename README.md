# fake-news-classifier
We have created a Machine Leaning model, which will help us to detect fake news.

Fake News Classifier (ML)

Fake News Classifier — a machine-learning pipeline that classifies news articles as REAL or FAKE using text features (title + body). The project trains multiple models (Logistic Regression, Random Forest, MultinomialNB) inside a scikit-learn Pipeline with ColumnTransformer + TfidfVectorizer, and selects the best model using GridSearchCV.

**Table of contents**

  1.Project Overview

  2.Repository Structure

  3.Requirements

  4.Setup / Installation

  5.Data

  6.How to run (Training)

  7.How to run (Inference / Predict)

  8.Evaluation & Visuals

  9.Recommended Improvements / Next Steps



**Project overview**

This repo contains a training script that:

  1.loads Real.csv and Fake.csv,
  
  2.creates a combined dataset with binary labels (real → 1, fake → 0),
  
  3.vectorizes title and text separately with TfidfVectorizer (via ColumnTransformer),
  
  4.runs GridSearchCV to tune hyperparameters for multiple models,
  
  5.prints and saves summary metrics (accuracy, F1),
  
  6.saves the final best estimator (pickle).
  
  7.utilities for inference (loading the pickled model and predicting on new articles).
  
  8.The pipeline is designed to be reproducible and portable (model saved as a single pipeline file).







**How to run (Training)**

The script training.py (provided) implements the full pipeline. Example usage:

python training.py



**How to run (Inference / Predict)**

Example predict.py (simple loader & predictor). Create this file or use this snippet:

import pickle
import pandas as pd


# Load saved pipeline (replace path as needed)
with open('outputs/news_auth_model.pkl','rb') as f:
    model = pickle.load(f)

# Example single prediction
sample = pd.DataFrame([{
    'title': 'Government Approves New Tax Reform Bill',
    'text': 'In a landmark move today, the Parliament passed ...'
}])

pred = model.predict(sample)
proba = model.predict_proba(sample)  # if estimator supports predict_proba

print('Prediction:', int(pred[0]))        # 1 -> real, 0 -> fake
print('Confidence (real):', proba[0][1]) # probability for positive class
