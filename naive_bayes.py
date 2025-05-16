import mlflow
import mlflow.sklearn
from data_preprocessing import X_train_tf, X_test_tf, y_train, y_test
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Tracking URI to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_name="Naive Bayes Classifier"):
    # Initialize and train the Naive Bayes model
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(X_train_tf, y_train)

    # Make predictions
    y_pred = naive_bayes_classifier.predict(X_test_tf)

    # Print classification report
    print("\nClassification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=['fake', 'correct']))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['fake', 'correct'],
                yticklabels=['fake', 'correct'])
    plt.title('Confusion Matrix - Naive Bayes')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('naive_bayes_confusion_matrix.png')
    plt.close()

    # Log parameter
    mlflow.log_param("model_type", "MultinomialNB")

    # Log metrics
    mlflow.log_metric("accuracy", metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", metrics.precision_score(y_test, y_pred))
    mlflow.log_metric("recall", metrics.recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", metrics.f1_score(y_test, y_pred))

    # Log confusion matrix plot as artifact
    mlflow.log_artifact("naive_bayes_confusion_matrix.png")

    # Log the trained model
    mlflow.sklearn.log_model(naive_bayes_classifier, "naive_bayes_model")
