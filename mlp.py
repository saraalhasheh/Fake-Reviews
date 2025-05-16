import mlflow
import mlflow.keras
mlflow.set_tracking_uri("http://localhost:5000")

from data_preprocessing import X_train_tf, X_test_tf, y_train, y_test
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import seaborn as sns

with mlflow.start_run(run_name="MLP Classifier"):
    # Define the MLP model
    model = Sequential()
    model.add(Dense(12, input_dim=X_train_tf.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train the model
    history = model.fit(
        X_train_tf.toarray(),
        y_train,
        validation_data=[X_test_tf.toarray(), y_test],
        epochs=25,
        batch_size=32
    )

    # Save training history
    pd.DataFrame(history.history).to_csv("mlp_training_history.csv")

    # Plot training history
    plt.figure(figsize=(10,4))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.title('Training History - MLP')
    plt.savefig("mlp_training_plot.png")
    plt.close()

    # Make predictions
    pred = model.predict(X_test_tf.toarray())
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
    plt.title('Confusion Matrix - MLP')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("mlp_confusion_matrix.png")
    plt.close()

    # Log parameters
    mlflow.log_param("layer1_units", 12)
    mlflow.log_param("layer2_units", 8)
    mlflow.log_param("activation", "relu")
    mlflow.log_param("output_activation", "sigmoid")

    # Log metrics
    mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", metrics.precision_score(y_test, y_pred))
    mlflow.log_metric("recall", metrics.recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", metrics.f1_score(y_test, y_pred))

    # Log artifacts
    mlflow.log_artifact("mlp_training_plot.png")
    mlflow.log_artifact("mlp_confusion_matrix.png")
    mlflow.log_artifact("mlp_training_history.csv")

    # Log the model
    mlflow.keras.log_model(model, "mlp_model")
