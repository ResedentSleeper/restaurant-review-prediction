import pandas as pd
import sys
from database_utils import get_connection, save_data


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    
    data.drop_duplicates(inplace = True)

    data = data.drop(columns=["Restaurant","Reviewer","Metadata","Time","Pictures", "7514"])
    data = data.dropna()
    data['Rating'] = data['Rating'].replace({'Like':5})

    connection = get_connection()
    save_data(connection, data, 'mainTable')
    connection.close()
    