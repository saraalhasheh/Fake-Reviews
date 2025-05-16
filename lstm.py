import mlflow
import mlflow.keras
mlflow.set_tracking_uri("http://localhost:5000")

from data_preprocessing import X_train, X_test, y_train, y_test
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import seaborn as sns

with mlflow.start_run(run_name="LSTM Classifier"):
    # Tokenization
    counter = Counter(" ".join(X_train).split(" "))
    myTokenizer = Tokenizer(num_words=300)
    myTokenizer.fit_on_texts(X_train)

    X_train_seq = myTokenizer.texts_to_sequences(X_train)
    X_test_seq = myTokenizer.texts_to_sequences(X_test)

    max_words = 300
    X_train_fin = pad_sequences(X_train_seq, maxlen=max_words)
    X_test_fin = pad_sequences(X_test_seq, maxlen=max_words)

    # Build LSTM model
    vocabulary_size = len(counter.keys())
    model = Sequential()
    model.add(Embedding(vocabulary_size, 32, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_fin, y_train, validation_data=[X_test_fin, y_test], epochs=25, batch_size=32)

    # Save training history
    pd.DataFrame(history.history).to_csv("lstm_training_history.csv")

    # Plot training history
    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Training History - LSTM')
    plt.savefig("lstm_training_plot.png")
    plt.close()

    # Make predictions
    pred = model.predict(X_test_fin)
    y_pred = [round(i[0]) for i in pred]

    # Print classification report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=['fake', 'correct']))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['fake', 'correct'],
                yticklabels=['fake', 'correct'])
    plt.title('Confusion Matrix - LSTM')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("lstm_confusion_matrix.png")
    plt.close()

    # Log parameters
    mlflow.log_param("embedding_dim", 32)
    mlflow.log_param("lstm_units", 100)
    mlflow.log_param("max_sequence_length", max_words)

    # Log metrics
    mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", metrics.precision_score(y_test, y_pred))
    mlflow.log_metric("recall", metrics.recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", metrics.f1_score(y_test, y_pred))

    # Log artifacts
    mlflow.log_artifact("lstm_training_plot.png")
    mlflow.log_artifact("lstm_confusion_matrix.png")
    mlflow.log_artifact("lstm_training_history.csv")

    # Log the Keras model
    mlflow.keras.log_model(model, "lstm_model")
