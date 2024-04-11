from preProcessing import *
from tabulate import tabulate
import pandas as pd

def getDf():
    df = data_preprocessing.load_data()
    return df

file_path = "../Dataset/german.data"
data_preprocessing = DataPreprocessing(file_path)
#Load the dataset
df = getDf()
data_preprocessing.check_class_distribution(df)
df_one_hot_encoded = data_preprocessing.one_hot_encoding(df)
df_normalized = pd.DataFrame(data_preprocessing.normalization(df_one_hot_encoded), columns=df_one_hot_encoded.columns)
print(tabulate(df_normalized.head(), headers='keys', tablefmt='psql'))

data_splitted = SplittedDataset(X=df_normalized.drop(columns="result"), y=df_normalized["result"])
X_train, y_train, X_test, y_test = data_splitted.split_data()



