import mlflow
import mlflow.keras
import optuna
import numpy as np
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

mlflow.set_tracking_uri("http://localhost:5000")

# Prepare your raw text data (not TF-IDF)
from data_preprocessing import X_train, X_test, y_train, y_test

def objective(trial):
    # Suggest hyperparameters
    embedding_dim = trial.suggest_categorical('embedding_dim', [16, 32, 64])
    lstm_units = trial.suggest_categorical('lstm_units', [50, 100, 150])
    max_sequence_length = trial.suggest_categorical('max_sequence_length', [100, 200, 300])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Tokenization and padding
    tokenizer = Tokenizer(num_words=300)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

    vocab_size = len(tokenizer.word_index) + 1

    # Build LSTM Model
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
    model.add(LSTM(lstm_units))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    history = model.fit(
        X_train_pad, y_train,
        validation_data=(X_test_pad, y_test),
        epochs=5,  # Reduce epochs for faster trials
        batch_size=batch_size,
        verbose=0
    )

    val_accuracy = history.history['val_accuracy'][-1]

    with mlflow.start_run(nested=True):
        mlflow.log_param('embedding_dim', embedding_dim)
        mlflow.log_param('lstm_units', lstm_units)
        mlflow.log_param('max_sequence_length', max_sequence_length)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_metric('val_accuracy', val_accuracy)
        mlflow.keras.log_model(model, "lstm_model")

    return val_accuracy

# Run Optuna Study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Print Best Trial Results
print("Best trial:")
print(f"  Value: {study.best_trial.value}")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")
