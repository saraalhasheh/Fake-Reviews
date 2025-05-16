import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import json
import csv
from datetime import datetime
import time
import os
from data_preprocessing import X_train

# Rebuild tokenizer
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(X_train)

st.title("Fake Review Detector")

user_input = st.text_area("Enter your review text here:")

if st.button("Predict"):
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=100).tolist()

    url = "http://127.0.0.1:1234/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {"instances": padded_sequence}

    # Measure response time
    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    end_time = time.time()
    response_time = round(end_time - start_time, 4)

    prediction_score = response.json()["predictions"][0][0]
    label = "Fake Review" if prediction_score > 0.5 else "Real Review"

    st.write(f"Prediction: {label} (Score: {prediction_score:.2f})")

    # Log to CSV with absolute path printing
    log_file = 'monitoring_log.csv'
    log_file_path = os.path.abspath(log_file)
    print("Writing to:", log_file_path)  # Diagnostic print

    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header if file is new
        if not file_exists:
            writer.writerow(['Timestamp', 'Input', 'Prediction', 'Model_Version', 'Response_Time(s)', 'Status_Code'])
        writer.writerow([datetime.now().isoformat(), user_input, prediction_score, "v1", response_time, response.status_code])
        file.flush()
        os.fsync(file.fileno())

    st.success(f"Prediction logged successfully to {log_file_path}")
