import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")



from data_preprocessing import X_train, X_test, y_train, y_test
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score

# Rebuild the best model
counter = Counter(" ".join(X_train).split(" "))
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_words = 300
X_train_pad = pad_sequences(X_train_seq, maxlen=max_words)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_words)

vocabulary_size = len(counter.keys())
model = Sequential()
model.add(Embedding(vocabulary_size, 128, input_length=max_words))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=10, batch_size=32)

y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
acc = accuracy_score(y_test, y_pred)

with mlflow.start_run(run_name="Best_LSTM_Model"):
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("lstm_units", 50)
    mlflow.log_param("max_sequence_length", 300)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    model_uri = mlflow.keras.log_model(model, "lstm_model")

    # Register the model
    client = MlflowClient()
    result = client.create_registered_model("LSTM_Review_Classifier")
    client.create_model_version(
        name="LSTM_Review_Classifier",
        source=model_uri.model_uri,
        run_id=mlflow.active_run().info.run_id
    )
