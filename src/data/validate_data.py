import pandas as pd


def check_missing_values(df):
    return df.isnull().sum()


def check_duplicates(df):
    return df.duplicated().sum()


def validate_data(df):
    print("premiere lignes :")
    print(df.head())
    print("\ndernieres lignes :")
    print(df.tail())
    print("\nInformations generale :")
    df.info()
    print("\nNoms des colonnes :", df.columns.tolist())
    print("\nnombre des ligne duplique :", check_duplicates(df))
    print("nombre total de valeurs manquantes :", check_missing_values(df).sum())
    print("\nvaleurs manquantes par colonne :")
    print(check_missing_values(df))

