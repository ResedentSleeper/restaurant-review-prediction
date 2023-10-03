import psycopg2
from psycopg2 import sql
from pandas import DataFrame

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="reviewDB",
        user="postgres",
        password="kpk22"
    )

#Update model version before train
def update_model_version():
    with open("./resources/version.txt", "r+") as file:
        version = int(file.readline(1)) + 1
        file.seek(0)
        file.write(str(version))
        file.truncate()
        return version

def get_version():
    with open("./resources/version.txt", "r") as file:
        return int(file.readline(1))

#Save clear data to table 
def save_data(connection, data, table_name):
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {table}(id, rating, review) values (%s, %s, %s)").format(table=sql.Identifier(table_name)),(index,row['Rating'], row['Review']))
    cursor.close()
    connection.commit()

#Get data from existing table
def get_data(connection, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select * from {table}").format(table=sql.Identifier(table_name)))
    result = DataFrame(cursor.fetchall(), index=None)
    result.set_index(0, inplace=True)
    cursor.close()
    return result

#Save data for train, test and additional train
def save_additional_data(connection, data, table_name):
    cursor = connection.cursor()
    for index, row in data.iterrows():
        cursor.execute(sql.SQL("INSERT INTO {table} (id, score, summary) values(%s, %s, %s)").format(table=sql.Identifier(table_name)), (index, row[1], row[2]))
    cursor.close()
    connection.commit()

#Save serialize vectorizer to table 
def save_vectorizer(connection, vectorizer_name, filename, current_version, table_name):
    cursor = connection.cursor()
    with open(filename, "rb") as file:
                cursor.execute(sql.SQL("INSERT INTO {table} (name, vectorizer_serialize, version) VALUES (%s, %s, %s)").format(table=sql.Identifier(table_name)),((vectorizer_name, psycopg2.Binary(file.read()), current_version)))
    cursor.close()
    connection.commit()

#Save serialize model to table 
def save_model(connection, model_name, filename, current_model_version, table_name):
    cursor = connection.cursor()
    with open(filename, "rb") as file:
        cursor.execute(sql.SQL("INSERT INTO {table} (name, model_serialize, version) VALUES (%s, %s, %s)").format(table=sql.Identifier(table_name)),(model_name, psycopg2.Binary(file.read()), current_model_version))
    cursor.close()
    connection.commit()

#Get serialize vectorizer from table
def get_vectorizer(connection, vectorizer_name, filename, current_version, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT vectorizer_serialize from {table} where version=%s and name=%s").format(table=sql.Identifier(table_name)),(current_version, vectorizer_name))
    vectorizer_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:
        file.write(vectorizer_data)
    cursor.close()

#Get serialize model by version from table
def get_model_by_version(connection, model_name, filename, current_version, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("SELECT model_serialize from {table} where version=%s and name=%s").format(table=sql.Identifier(table_name)), (current_version, model_name))
    model_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:
        file.write(model_data)
    cursor.close()

#Set model metrics and add to the table
def update_model_metrics(connection, model_name, current_version, accuracy_score, duration, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("update {table} set accuracy=%s, time=%s where name=%s and version=%s").format(table=sql.Identifier(table_name)), (accuracy_score, duration, model_name, current_version))
    cursor.close()
    connection.commit()

#Get model metrics from table
def get_model_metrics(connection, current_version, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select name, accuracy, time from {table} where version=%s").format(table=sql.Identifier(table_name)), [current_version])
    result = DataFrame(cursor.fetchall(), index=None)
    cursor.close()
    return result

#Set model for deploy
def save_model_to_deploy(connection, version, name, score, filename_m, filename_v, table_name):
    cursor = connection.cursor()
    with open(filename_m, "rb") as file_1:
        with open(filename_v, "rb") as file_2:
            cursor.execute(sql.SQL("INSERT INTO {table} (version, name, score, model_serialize, vectorizer_serialize) values (%s, %s, %s, %s, %s)").format(table=sql.Identifier(table_name)),(version, name, score, psycopg2.Binary(file_1.read()), psycopg2.Binary(file_2.read())))
    cursor.close()
    connection.commit()

#Get model for deploy
def get_model_to_deploy(connection, filename, version, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select model_serialize from {table} where version=%s").format(table=sql.Identifier(table_name)), version)
    model_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:        
        file.write(model_data)
    cursor.close()

def get_vector_to_deploy(connection, filename, version, table_name):
    cursor = connection.cursor()
    cursor.execute(sql.SQL("select vectorizer_serialize from {table} where version=%s").format(table=sql.Identifier(table_name)), version)
    model_data = cursor.fetchone()[0]
    with open(filename, "wb") as file:        
        file.write(model_data)
    cursor.close()
