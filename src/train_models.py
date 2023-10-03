import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from database_utils import get_data, save_vectorizer, get_connection, save_model, update_model_version


#Create and serealize vector 
def tf_idf_vectorizer(connection, model_version):
    vectorizer = TfidfVectorizer()

    train_data = get_data(connection, "train")
    train_x = vectorizer.fit_transform(train_data[2])

    filename = "./resources/vector.txt"
    with open(filename, "wb") as file:
        pickle.dump(vectorizer, file)
    
    save_vectorizer(connection, "tf_idf", filename, model_version, 'vectorizer')
    os.remove(filename)
    return train_x, train_data[1]


def logistic_regression_model(connection, model_version, train_x, train_y):
    model = LogisticRegression(random_state=100)

    model.fit(train_x, train_y)

    filename = "./resources/model.txt"
    with open(filename, "wb") as file:
        pickle.dump(model, file)

    save_model(connection, "logistic_regression", filename, model_version, 'modelsTable')
    os.remove(filename)


def random_forest_classifier(connection, model_version, train_x, train_y):
    model = RandomForestClassifier(random_state=100)

    model.fit(train_x, train_y)

    filename = "./resources/model.txt"
    with open(filename, "wb") as file:
        pickle.dump(model, file)

    save_model(connection, "random_forest", filename, model_version,'modelsTable')
    os.remove(filename)


if __name__ == "__main__":

    connection = get_connection()
    version = update_model_version()

    #Vectorizer
    train_x, train_y = tf_idf_vectorizer(connection, version)
    
    #Regression
    logistic_regression_model(connection, version, train_x, train_y)

    #Classifier
    random_forest_classifier(connection, version, train_x, train_y)
    connection.close()
