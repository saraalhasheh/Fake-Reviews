import pandas as pd
import requests
import os
from io import StringIO

# Download the dataset from OSF
url = "https://osf.io/3vds7/download"
response = requests.get(url)

if response.status_code == 200:
    # Save the dataset
    with open('data/fake-reviews-dataset.csv', 'wb') as f:
        f.write(response.content)
    
    # Load the dataset
    df = pd.read_csv('data/fake-reviews-dataset.csv')
    
    # Print basic information
    print("\nDataset Info:")
    print(df.info())
    
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    
    print("\nNull Values:")
    print(df.isnull().sum())
    
    # Word distribution analysis
    print("\nWord Distribution Statistics:")
    df['word_count'] = df['text_'].str.split().str.len()
    print(df['word_count'].describe())
    
else:
    print(f"Failed to download dataset. Status code: {response.status_code}") 