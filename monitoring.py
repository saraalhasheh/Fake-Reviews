import requests
import json
import time
import csv
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import X_train

# Rebuild tokenizer
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(X_train)

# Example batch of texts to simulate streaming data
sample_texts_batch = [
    "I loved this product, highly recommended!",
    "Terrible experience, I will never buy this again.",
    "Best purchase I have made this year.",
    "This is a fake review with repetitive phrases and keywords."
]

# Prepare CSV to log results
with open('monitoring_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Input', 'Prediction'])

    for text in sample_texts_batch:
        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=100).tolist()

        # Prepare request
        url = "http://127.0.0.1:1234/invocations"
        headers = {"Content-Type": "application/json"}
        payload = {"instances": padded_sequence}

        # Send request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        prediction = response.json()["predictions"][0][0]

        # Log result with timestamp
        timestamp = datetime.now().isoformat()
        print(f"{timestamp} | Input: {text} | Prediction: {prediction}")
        writer.writerow([timestamp, text, prediction])

        # Optional: Wait before next request (simulate live data)
        time.sleep(2)

print("Monitoring completed. Log saved to monitoring_log.csv")
