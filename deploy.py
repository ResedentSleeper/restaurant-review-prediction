from flask import Flask, render_template, request, jsonify
import joblib
from src.database_utils import get_connection, get_model_metrics, save_model_to_deploy, get_version, get_model_by_version, get_vectorizer


class Server:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

class Model:
    def __init__(self, name, score):
        self.name = name
        self.score = score

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
    model = joblib.load(filename_m)
    get_vectorizer(connection, "tf_idf", filename_v, str(version), "vectorizer")
    vectorizer = joblib.load(filename_v)

    save_model_to_deploy(connection, str(version), result[0].name, str(result[0].score), filename_m, filename_v,"deployTable")
    return Server(model, vectorizer)

app = Flask(__name__)
connection = get_connection()
version = get_version()
server = select_model(connection, version)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def prediction():
    req = request.get_json()
    review = req['data']
    print(review)
    vectorized = server.vectorizer.transform([review])
    score = str(server.model.predict(vectorized))
    print(score)
    data = {'score' : score}
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
    connection.close()
