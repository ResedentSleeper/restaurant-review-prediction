import pickle
import time
import os
from sklearn.metrics import accuracy_score
from database_utils import get_connection, get_vectorizer, get_data, get_model_by_version, update_model_metrics, get_version


def load_vectorizer(connection, version, vectorizer_name):
    filename = "./resources/vector.txt"
    get_vectorizer(connection, vectorizer_name, filename, str(version), "vectorizer")

    with open(filename, "rb") as file:
         vectorizer = pickle.load(file)

    os.remove(filename)
    return vectorizer


def save_model(connection, vectorizer, version, model_name, test_x, test_y):
    vectorized_test_x = vectorizer.transform(test_x)
    filename = "./resources/model.txt"

    get_model_by_version(connection, model_name, filename, str(version), "modelsTable")
    with open(filename, "rb") as file:
        model = pickle.load(file)
    os.remove(filename)

    start = time.time()
    predicted_y = model.predict(vectorized_test_x)
    duration = time.time() - start

    accuracy = accuracy_score(test_y, predicted_y)
    update_model_metrics(connection, model_name, str(version), str(accuracy), str(duration), "modelsTable")


if __name__ == "__main__":

    connection = get_connection()
    version = get_version()

    data = get_data(connection, "test")

    vectorizer = load_vectorizer(connection, version, "tf_idf")
    
    save_model(connection, vectorizer, version, "random_forest", data[2], data[1])
    save_model(connection, vectorizer, version, "logistic_regression", data[2], data[1])

    connection.close()
