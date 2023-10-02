import os
import pickle
from database_utils import get_connection, get_vectorizer, update_model_version, get_data, save_vectorizer, get_model_by_version, save_model, get_version


def additional_vectorizer(connection, version, vectorizer_name, add_train_X):
    filename = "D:/labs/restaurant-review-prediction/resources/vector.txt"
    get_vectorizer(connection, vectorizer_name, filename, str(version), "vectorizer")
    
    with open(filename, "rb") as file:
         vectorizer = pickle.load(file)
    os.remove(filename)
    add_train_X = vectorizer.fit_transform(add_train_X)

    filename = "D:/labs/restaurant-review-prediction/resources/vector.txt"
    with open(filename, "wb") as file:
        pickle.dump(vectorizer, file)
    version = version + 1
    save_vectorizer(connection, "tf_idf", filename, str(version), "vectorizer")
    os.remove(filename)
    
    return add_train_X



def additional_train_models(connection, model_version, model_name, test_x, test_y):
    filename = "D:/labs/restaurant-review-prediction/resources/model.txt"
    get_model_by_version(connection, model_name, filename, str(model_version),"modelsTable")

    with open(filename, "rb") as file:
        model = pickle.load(file)
    os.remove(filename)
    model.fit(test_x, test_y)

    with open(filename, "wb") as file:
        pickle.dump(model, file)
    model_version = model_version + 1
    save_model(connection, model_name, filename, str(model_version), "modelsTable")

    os.remove(filename)


if __name__ == "__main__":

    connection = get_connection()
    version = get_version()
    
    additional_data = get_data(connection, "trainAdd")
    data = get_data(connection, "train")
    
    extra_train_X = additional_vectorizer(connection, version, "tf_idf", additional_data[2])
    additional_train_models(connection, version, "logistic_regression", extra_train_X,  additional_data[1])
    additional_train_models(connection, version, "random_forest", extra_train_X,  additional_data[1])

    update_model_version()

    connection.close()