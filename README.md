# **Fake Review Detection System**

## Overview

This project builds an **AI-powered Fake Review Detection System** that classifies product reviews as **Real** or **Fake**.
It uses **MLflow** for tracking experiments, **MLflow Model Registry** for model management, and **Streamlit** for providing a **user-friendly web interface**.

## Abstract

The rapid growth of the e-commerce ecosystem, fueled by widespread internet access and smart device adoption, has shifted consumer behavior toward relying on online reviews as a key factor in purchasing decisions. While positive reviews help businesses build credibility and increase sales, the growing prevalence of fake reviews—intentionally crafted to manipulate customer perceptions—has emerged as a critical challenge. With the global e-commerce market projected to exceed USD 16 trillion by 2027, addressing the impact of fake reviews has become increasingly important. Existing research in this area remains limited, often relying on traditional machine learning methods and small datasets. This study advances the field by comparing multiple machine learning models, including Naive Bayes, Support Vector Machine, Multi-layer Perceptron, and Long Short-Term Memory (LSTM), using the Fake Reviews Dataset by Salminen. Experimental results show that LSTM achieves the highest detection accuracy of 92%, highlighting its effectiveness over other approaches in combating the growing issue of fake online reviews.



## Features

* **Data Preprocessing** using Tokenization and Padding.
* **Model Training** with LSTM and Hyperparameter Tuning using Hyperopt.
* **Model Versioning** with MLflow Model Registry.
* **Model Deployment** using MLflow Serve.
* **Live Predictions** via a REST API.
* **Performance Monitoring** with CSV Logging.
* **User Interface** using Streamlit.

## **Reproducibility Steps**

Follow these steps to fully reproduce the Fake Review Detection Pipeline:

1. **Create and Activate Conda Environment**

   ```bash
   conda create -n mlops-project python=3.10
   conda activate mlops-project
   ```

2. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Start MLflow Tracking Server**

   ```bash
   python mlflow_server.py
   ```

4. **Train Machine Learning Models in Sequence**

   ```bash
   python svm.py
   python naive_bayes.py
   python mlp.py
   python lstm.py
   python hybrid.py
   ```

5. **Perform Hyperparameter Tuning on LSTM**

   ```bash
   python lstm_optuna_tuning.py
   python lstm_tuning.py
   ```

6. **Serve the Best Model with MLflow**

   ```bash
   mlflow models serve -m "file:///C:/Users/their/Desktop/fake-reviews-detection/mlruns/0/f0023b3f2a3448d886f4e4236bdb51b2/artifacts/model" -p 1234 --host 0.0.0.0 --no-conda
   ```

7. **Send Test Requests via Python Client**

   ```bash
   python post_request.py
   ```

8. **Launch Streamlit Web Interface for Live Testing**

   ```bash
   streamlit run app.py
   ```



## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/fake-review-detector.git
   cd fake-review-detector
   ```

2. **Set Up Virtual Environment**

   ```bash
   conda create -n mlops-project python=3.10
   conda activate mlops-project
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```



## Model Management with MLflow

1. Visit `http://localhost:5000` to open the **MLflow Tracking UI**.
2. Register the best model in **Model Registry**.
3. Promote the model to **Staging** and **Production**.


## Example Prediction

* **Input:** "I highly recommend this product to everyone!!!"
* **Output:** Fake Review (Score: 0.87)


## Monitoring Example Log

| Timestamp                | Input                                     | Prediction | Model\_Version | Response\_Time(s) | Status\_Code |
| ------------------------ | ----------------------------------------- | ---------- | -------------- | ----------------- | ------------ |
| 2025-05-15T19:00:00.000Z | I loved this product, highly recommended! | 0.12       | v1             | 0.4321            | 200          |



## License

This project is licensed under the **MIT License**.
