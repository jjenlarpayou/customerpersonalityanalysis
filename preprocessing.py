import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

def preprocessing(filename):
    df = pd.read_csv(filename,sep="\t")
    for idx, value in enumerate(df.duplicated()):
        if value == True:
            print(idx, value)
    
    df.drop_duplicates(inplace=True)
    for idx, value in enumerate(df['Income']):
        if value == df['Income'].max():
            print(idx)
    df = df.drop(index=2233)
    df['Age'] = df['Year_Birth'].max() - df['Year_Birth']
    df = df[df['Age']<=85]
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
    df['Tenure'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days
    income_mean = df['Income'].mean()
    df['Income'] = df['Income'].fillna(income_mean)
    return df

def group_features(df):
    conditions = [
        df['Income'] <= 20000,
        (df['Income'] > 20000) & (df['Income'] <= 40000),
        (df['Income'] > 40000) & (df['Income'] <= 60000),
        (df['Income'] > 60000) & (df['Income'] <= 80000),
        (df['Income'] > 80000) & (df['Income'] <= 100000),
        df['Income'] > 100000
    ]

    choices = ['20K', '20K–40K', '40K–60K', '60K–80K', '80K-100k', '100k']
    df['Group Income'] = np.select(conditions, choices)
    conditions_age = [
        df['Age'] <= 10,
        (df['Age'] > 10) & (df['Age'] <= 20),
        (df['Age'] > 20) & (df['Age'] <= 30),
        (df['Age'] > 30) & (df['Age'] <= 40),
        (df['Age'] > 40) & (df['Age'] <= 50),
        df['Age'] > 50
    ]

    choices_age = ['10','10-20','20-30','30-40','40-50','50']
    df['Group Age'] = np.select(conditions_age, choices_age)
    df['Household Size'] = df['Kidhome'] + df['Teenhome']
    return df

def label_encoder(df):
    label_enc = LabelEncoder()
    df_edu_feature = label_enc.fit_transform(df['Education'])
    df_marital_feature = label_enc.fit_transform(df['Marital_Status'])
    df_group_income = label_enc.fit_transform(df['Group Income'])
    df_group_age = label_enc.fit_transform(df['Group Age'])
    df['Edu Encoded'] = df_edu_feature
    df['Marital Encoded'] = df_marital_feature
    df['Group Income Encoded'] = df_group_income
    df['Group Age Encoded'] = df_group_age
    return df
    