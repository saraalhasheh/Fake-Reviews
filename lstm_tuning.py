import mlflow
import mlflow.keras
mlflow.set_tracking_uri("http://localhost:5000")

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from data_preprocessing import X_train, X_test, y_train, y_test
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score
import numpy as np

# Tokenization
counter = Counter(" ".join(X_train).split(" "))
tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

def build_model(params):
    max_words = int(params['max_sequence_length'])
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_words)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_words)

    vocabulary_size = len(counter.keys())
    model = Sequential()
    model.add(Embedding(vocabulary_size, int(params['embedding_dim']), input_length=max_words))
    model.add(LSTM(int(params['lstm_units'])))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=5, batch_size=32, verbose=0)

    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
    acc = accuracy_score(y_test, y_pred)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.keras.log_model(model, "model")

    return {'loss': -acc, 'status': STATUS_OK}

# Search space
space = {
    'embedding_dim': hp.choice('embedding_dim', [16, 32, 64, 128]),
    'lstm_units': hp.choice('lstm_units', [50, 100, 150, 200]),
    'max_sequence_length': hp.choice('max_sequence_length', [100, 200, 300, 400, 500])
}

trials = Trials()
best = fmin(fn=build_model, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
print("Best Parameters:", best)
