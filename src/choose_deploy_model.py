from database_utils import get_connection, get_model_metrics, save_model_to_deploy, get_version, get_model_by_version, get_vectorizer

class Model:
    def __init__(self, name, score):
        self.name = name
        self.score = score

class Data_s:
    def __init__(self, model, vector):
        self.model = model
        self.vector = vector

def select_model(connection, version):
    filename_m = "./resources/model.txt" 
    filename_v = "./resources/vector.txt"

    model_metrics = get_model_metrics(connection, str(version), "modelsTable")
    result = []
    
    for i, row in model_metrics.iterrows():
        acc = row[1]
        time = row[2]
        score = 1/float(acc) + float(time)
        result.append(Model(row[0], score))
    result.sort(key=lambda x: x.score)

    get_model_by_version(connection,  result[0].name, filename_m, str(version), "modelsTable")
    get_vectorizer(connection, "tf_idf", filename_v, str(version), "vectorizer")
    
    save_model_to_deploy(connection, str(version), result[0].name, str(result[0].score), filename_m, filename_v,"deployTable")
    return Data_s(filename_m, filename_v)


if __name__ == "__main__":

    connection = get_connection()
    version = get_version()

    select_model(connection, version)

    connection.close()
