from database_utils import get_connection, get_model_metrics, save_model_to_deploy, get_version, get_model_by_version, get_vectorizer

class model:
    def __init__(self, name, score):
        self.name = name
        self.score = score


def select_model(connection, version):
    filename_m = "D:/labs/restaurant-review-prediction/resources/model.txt" 
    filename_v = "D:/labs/restaurant-review-prediction/resources/vector.txt"

    model_metrics = get_model_metrics(connection, str(version), "modelsTable")
    result = []
    
    for i, row in model_metrics.iterrows():
        acc = row[1]
        time = row[2]
        score = 1/float(acc) + float(time)
        result.append(model(row[0], score))
    result.sort(key=lambda x: x.score)

    get_model_by_version(connection,  result[0].name, filename_m, str(version), "modelsTable")
    get_vectorizer(connection, "tf_idf", filename_v, str(version), "vectorizer")
    
    save_model_to_deploy(connection, str(version), result[0].name, str(result[0].score), filename_m, filename_v,"deployTable")


if __name__ == "__main__":

    connection = get_connection()
    version = get_version()

    select_model(connection, version)

    connection.close()
