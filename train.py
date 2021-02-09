import os
import subprocess
import sys
import argparse
import numpy as np
from google.oauth2 import service_account
from google.cloud import bigquery
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime


import warnings
warnings.filterwarnings("ignore")

class Config:
    inputType: "bq"
    initialData: None


def main(args):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    file = args.file
    f = open(file, "r")
    lines = f.readlines()

    o = parse_config(lines)

    #Set GOOGLE_APPLICATION_CREDENTIAL to access bigquery
    key_path = o.credential

    rows = [];
    limit = 100
    offset = 0

    print("Retrieving data...")
    if(o.inputType == "csv"):
        rows = retrieve_data_csv(key_path, o.input, None, None)
    else:
        rows = retrieve_data(key_path, o.input, None, None)

    print("Table :")
    head = rows.head()
    print(head)

    print("Summary:")
    # summarize the dataset
    print(rows.describe())

    print("Processing...")
    model, score = process_data(rows, o)

    print("Score: " + str(score))

    print("Saving the model as " + o.output)
    save_model(model, o.output)

    print("Done.")


def parse_config(lines):
    o = Config()
    o.initialData = None
    o.credential = None
    o.algorithm = "RF"
    o.scaler = None
    o.denc = None
    o.lenc = None
    counter = 0
    while counter < len(lines):
        line = lines[counter]
        if("#" in line):
            counter = counter + 1
            continue
        elif("-input-type " in line):
            s = line.replace("-input-type ", "")
            s = s.strip()
            o.inputType = s
        elif("-output-type " in line):
            s = line.replace("-output-type ", "")
            s = s.strip()
            o.outputType = s
        elif("-credential " in line):
            s = line.replace("-credential ", "")
            s = s.strip()
            if(len(s) == 0):
                s = None
            o.credential = s
        elif("-input " in line):
            s = line.replace("-input ", "")
            s = s.strip()
            o.input = s
        elif("-output " in line):
            s = line.replace("-output ", "")
            s = s.strip()
            o.output = s
        elif("-features " in line):
            s = line.replace("-features ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.features = s
        elif("-target " in line):
            s = line.replace("-target ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.target = s
        elif("-lenc " in line):
            s = line.replace("-lenc ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.lenc = s
        elif("-denc " in line):
            s = line.replace("-denc ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.denc = s
        elif("-scaler " in line):
            s = line.replace("-scaler ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.scaler = s
        elif("-model " in line):
            s = line.replace("-model ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.model = s
        elif("-initial-data " in line):
            s = line.replace("-initial-data ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.initialData = s
        counter = counter + 1
    return o


def get_client(key_path=None, project_id=None):
    if(key_path is None):
        client = bigquery.Client(project_id)
    else:
        credentials = service_account.Credentials.from_service_account_file(
                key_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
                credentials=credentials,
                project=credentials.project_id,
        )
    return client

def retrieve_data(key_path, tablename, offset, limit):
    job_config=bigquery.QueryJobConfig()
    if(offset is not None):
        #BigQuery parameter
        params=[
            bigquery.ScalarQueryParameter('limit', 'INT64', limit), bigquery.ScalarQueryParameter('offset', 'INT64', offset)
        ]
        job_config.query_parameters=params

    ss = tablename.split(".")
    project_id = ss[0]

    #BigQuery client
    client = get_client(key_path, project_id)

    # Perform a query.
    QUERY = ""
    if(limit is not None):
        QUERY = (
            'SELECT * FROM `' + tablename  + '` '
            'WHERE LIMIT @limit OFFSET @offset')
        query_job = client.query(QUERY,job_config=job_config)  # API request
    else:
        QUERY = (
            'SELECT * FROM `' + tablename  + '` ')
        query_job = client.query(QUERY)  # API request

    result = query_job.result()
    print("total rows :"  + str(result.total_rows))

    #if rows is 0 return null
    if(result.total_rows == 0):
        return None
    else:
        rows = query_job.result().to_dataframe()  # Waits for job to complete.
        return rows

def retrieve_data_csv(key_path, csvfile, offset, limit):
    if(offset is not None):
        dataset = pd.read_csv(csvfile, skiprows=lambda idx: idx < offset, nrows=limit)
    else:
        dataset = pd.read_csv(csvfile)

    total = len(dataset.index)
    if(total  == 0):
        return None
    else:
        return dataset



def split_X_Y(rows, o):
    cols = o.features.split(",")
    target = o.target
    X = rows[cols]
    Y = rows[target]
    return X,Y

def label_encode(X,Y, o):
    cols = o.lenc.split(",")
    headx = X.head()
    heady = Y.head()
    le = LabelEncoder()

    for col in cols:
        if(col in headx):
            X[col] = le.fit_transform(X[col])

    for col in cols:
        if(col in heady):
            Y[col] = le.fit_transform(Y[col])
    return X, Y


def process_data(rows, o):

    print("Splitting X and Y data")
    X, Y = split_X_Y(rows, o)

    if(o.lenc is not None):
        print("Label encode")
        X, Y = label_encode(X, Y, o)
        #print(rows.head())

    if(o.denc is not None):
        print("Dummy encode")
        X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

    if(o.scaler is not None):
        print("StandardScaler")
        sc = StandardScaler(with_mean=False)
        X = sc.fit_transform(X)

    X_train, X_test, Y_train, Y_test = split_train_test(X, Y)

    print("Create ml engine " + o.algorithm)
    engine = create_ml(o.algorithm)
    score = train(engine, X_train, Y_train, X_test, Y_test)

    return engine, score

def create_ml(type):
    if(type == "RF"):
        return create_random_forest_classifier()

def create_random_forest_classifier():
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, criterion='entropy', random_state=0)
    return classifier

def split_train_test(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    return X_train, X_test, Y_train, Y_test

def train(classifier, X_train, Y_train, X_test, Y_test):
    current_time = datetime.datetime.now()

    print("Training start at " + str(current_time) + " ...")
    classifier.fit(X_train, Y_train)
    current_time = datetime.datetime.now()
    print("Training end at " + str(current_time) + " ...")

    ## Predicting the result
    print("Predicting...")
    y_pred = classifier.predict(X_test)

    print("Score...")
    score = accuracy_score(Y_test, y_pred)

    return score

def save_model(engine, model_file):
    pickle.dump(engine, open(model_file, 'wb'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Age group prediction')
    required = parser.add_argument_group("Required argument")
    required.add_argument('-f', '--file', type=str, help="configuration filename", required=True)
    args = parser.parse_args()
    main(args)
