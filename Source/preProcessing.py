from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from tabulate import tabulate


class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.columns = ["existing_account", "month_duration", "credit_history",
                        "purpose", "credit_amount", "saving_bonds",
                        "employment_status", "installment_rate", "status_sex",
                        "debts_status", "resident_since", "property",
                        "age", "installment_plans", "housing_status",
                        "credit_number", "job", "people_liability",
                        "telephone", "foreign", "result"]
        self.numerical_attributes = ["month_duration", "credit_amount", "installment_rate", "resident_since", "age",
                                     "credit_number", "people_liability"]

    def load_data(self):
        return pd.read_csv(self.file_path, sep=" ", header=None, names=self.columns)

    def check_class_distribution(self, df):
        good_creditors = df["result"] == 1
        bad_creditors = df["result"] == 2
        # print(f"Good creditors in percentage are: {df[good_creditors].shape[0] / df.shape[0] * 100}%")
        # print(f"Bad creditors in percentage are: {df[bad_creditors].shape[0] / df.shape[0] * 100}%")
        # print(tabulate(df.head(), headers='keys', tablefmt='psql'))

    def one_hot_encoding(self, df):
        df_numerical = df.copy()
        dummy_columns = ["credit_history", "purpose", "status_sex",
                         "debts_status", "property", "installment_plans",
                         "housing_status", "foreign", "existing_account",
                         "saving_bonds", "telephone", "job", "employment_status"]
        df_numerical = pd.get_dummies(df_numerical, columns=dummy_columns, drop_first=True)
        df_numerical_hot = df_numerical.replace([True, False], [1, 0])
        df_numerical_hot['result'] = df_numerical_hot['result'].replace([2, 1], [1, 0])
        # print(tabulate(df_numerical_hot.head(), headers='keys', tablefmt='psql'))
        return df_numerical_hot

    def normalization(self, df):
        numerical_columns = self.numerical_attributes
        self.scaler = MinMaxScaler()
        df_copy = df.copy()
        df_normalized = self.scaler.fit_transform(df_copy)
        return df_normalized



class SplittedDataset:
    def __init__(self, X, y, test_size=0.3):
        self.X = X
        self.y = y
        self.test_size = test_size

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            stratify=self.y,
            test_size=self.test_size
        )
        return X_train, y_train, X_test, y_test
