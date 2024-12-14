# Text-classfication-using-lstm-bert-other-models
# Text Classification on IMDb Movie Reviews

This repository contains a project focused on sentiment classification for IMDb movie reviews. The task involves classifying reviews as either positive or negative using various machine learning and deep learning models such as BERT, LSTM, Logistic Regression, Naïve Bayes, Linear SVM, and XGBoost.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Models Evaluated](#models-evaluated)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview
Sentiment analysis is a crucial text classification task aimed at understanding public sentiment. This project compares the performance of traditional and deep learning models on the IMDb dataset for binary sentiment classification.

## Features
- Preprocessing pipeline for sentiment classification.
- Comparison of deep learning (BERT, LSTM) and traditional models (Logistic Regression, Naïve Bayes, etc.).
- Evaluation metrics: Accuracy and Training Time.
- Visualizations for model comparison.

## Setup

### Prerequisites
- Python 3.8 or higher
- Libraries: `transformers`, `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `xgboost`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>

**Running on Google Colab**
Upload Text_classification_G24.ipynb to Google Drive.
Open it in Google Colab and follow the setup instructions in the notebook.
Models Evaluated
BERT: State-of-the-art transformer-based language model for contextual understanding.
LSTM: Sequential model capable of learning long-term dependencies.
Logistic Regression: Efficient and interpretable traditional machine learning model.
Naïve Bayes: Probabilistic classifier based on Bayes' theorem.
Linear SVM: Effective for high-dimensional data.
XGBoost: Gradient-boosting decision tree algorithm.

Install Required Libraries Create and activate a virtual environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Launch Jupyter Notebook

bash
Copy code
jupyter notebook
Open Text_classification_G24.ipynb in the Jupyter Notebook interface and execute cells sequentially.

Dataset Ensure the IMDb dataset is available in the correct folder or upload it via the notebook interface.

Option 2: Running on Google Colab
Upload Notebook to Google Drive

Go to Google Drive.
Upload the file Text_classification_G24.ipynb.
Open in Colab

Right-click on the file and choose Open With > Google Colab.
Install Required Libraries If any libraries are missing, install them in the first cell of the notebook:

python
Copy code
!pip install transformers torch scikit-learn matplotlib pandas xgboost
Upload the Dataset If the dataset isn’t included, upload it using the following code in a new cell:

python
Copy code
from google.colab import files
uploaded = files.upload()
Run the Notebook Execute cells sequentially by pressing Shift + Enter

