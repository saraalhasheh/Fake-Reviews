# Numerical and data manipulation libraries
import numpy as np
import pandas as pd
import os

# Visualization library
import matplotlib.pyplot as plt

# NLTK libraries for text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sklearn libraries for text feature extraction and machine learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Regular expression and string manipulation
import re
import string

# Initialize stop words
stop_words = stopwords.words('english')

# Check if dataset exists
dataset_path = os.path.join("data", "reviews-dataset.csv")
if not os.path.exists(dataset_path):
    print("Error: Dataset file not found!")
    print("Expected path:", os.path.abspath(dataset_path))
    exit(1)

# Load and preprocess data
print("Loading dataset...")
df = pd.read_csv(dataset_path)

print("Preprocessing data...")
df["text_"] = df["text_"].apply(lambda x: x.lower()) #lowercase
data = df[["text_","label"]]
data["label"] = data["label"].apply(lambda x: 1 if x=="CG" else 0) #label_encoding

# Train-Test Split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(data["text_"], data["label"], test_size=0.33, random_state=42)

# TF-IDF Vectorization
print("Performing TF-IDF vectorization...")
tf_idf = TfidfVectorizer()

# applying tf idf to training data
X_train_tf = tf_idf.fit_transform(X_train)
X_train_tf = tf_idf.transform(X_train)

# transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(X_test)

print("\nData shapes:")
print("Training data shape:", X_train_tf.shape)
print("Testing data shape:", X_test_tf.shape)
print("\nPreprocessing completed successfully!") 