from data_preprocessing import X_train
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import json

# Rebuild tokenizer based on your X_train
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(X_train)

# Example text to send for prediction
sample_texts = ["This product is really great and I love it"]

# Tokenize and pad with correct length (100)
sequences = tokenizer.texts_to_sequences(sample_texts)
padded_sequences = pad_sequences(sequences, maxlen=100).tolist()

# Prepare the request payload
payload = {"instances": padded_sequences}

# Send the request to the running MLflow model server
url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.text)
