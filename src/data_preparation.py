import re 
import nltk
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from database_utils import get_data, get_connection, save_additional_data


def edit_text(text, morph):
    modified = []
    for row in text:
        row = re.sub(r'[^\w\s]', '', str(row).lower().strip())
        tokens = word_tokenize(row)
        modified.append(" ".join([morph.normal_forms(word)[0] for word in tokens if not word in stopwords.words('english')]))
    return modified

def rate(rate):
    if float(rate) > 3:
        return 1
    else:
        return -1

if __name__ == "__main__":
    #setup
    nltk.download('stopwords')
    nltk.download('punkt')

    #get rewievs and rating from table
    connection = get_connection()
    data = get_data(connection, 'mainTable')

    #test and train split 
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    test_data, train_data_add = train_test_split(test_data, test_size=0.5, random_state=42)
    
    morph_analyzer = pymorphy2.MorphAnalyzer()

    #add data for train
    train_data[2] = edit_text(train_data[2], morph_analyzer)
    train_data[1] = train_data[1].apply(lambda v: rate(v))
    save_additional_data(connection, train_data, "train")

    #add data for test
    test_data[2] = edit_text(test_data[2], morph_analyzer)
    test_data[1] = test_data[1].apply(lambda v: rate(v))
    save_additional_data(connection, test_data, "test")
    
    #add data for additional training
    train_data_add[2] = edit_text(train_data_add[2], morph_analyzer)
    train_data_add[1] = train_data_add[1].apply(lambda v: rate(v))
    save_additional_data(connection, train_data_add, "trainAdd")

    connection.close()