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
import time

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

    print("Output")
    print(o.output)

    delete_result_table(key_path, o.output)
    while (rows is not None):
        print("\n-----======ooooooooooooo=====------\nRetrieving data from " + o.input + " : " + str(offset) + " to " + str((offset + limit)))
        if(o.inputType == "csv"):
            rows = retrieve_data_csv(key_path, o.input, offset, limit)
        else:
            rows = retrieve_data(key_path, o.input, offset, limit)


        if(rows is not None):
            predicteddata = process_data(rows, o)
            print("Saving the result...")
            if(o.outputType == "csv"):
                save_data_csv(rows, predicteddata, key_path, o.input, o.output)
            else:
                save_data(rows, predicteddata, key_path, o.input, o.output)
        else:
            print("No more data")
        print("------=======ooooooooooooo======------")
        offset = offset + limit


    print("Done.")

def delete_result_table(key_path, tablename):
    newtablename = tablename
    ss = tablename.split(".")
    project_id = ss[0]
    client = get_client(key_path, project_id)
    client.delete_table(newtablename, not_found_ok=True)

def parse_config(lines):
    o = Config()
    o.initialData = None
    o.credential = None
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
        elif("-input-project " in line):
            s = line.replace("-input-project ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.inputProject = s
        elif("-output-project " in line):
            s = line.replace("-output-project ", "")
            s = s.strip()
            s = s.replace(" ", "")
            o.outputProject = s
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
    #BigQuery parameter
    params=[
        bigquery.ScalarQueryParameter('limit', 'INT64', limit), bigquery.ScalarQueryParameter('offset', 'INT64', offset)
    ]
    job_config=bigquery.QueryJobConfig()
    job_config.query_parameters=params

    ss = tablename.split(".")
    project_id = ss[0]

    #BigQuery client
    client = get_client(key_path, project_id)

    # Perform a query.
    QUERY = (
        'SELECT * FROM `' + tablename  + '` '
        'LIMIT @limit OFFSET @offset')
    query_job = client.query(QUERY,job_config=job_config)  # API request
    result = query_job.result()

    print("total rows :"  + str(result.total_rows))

    #if rows is 0 return null
    if(result.total_rows == 0):
        return None
    else:
        rows = query_job.result().to_dataframe()  # Waits for job to complete.
        return rows

def retrieve_data_csv(key_path, csvfile, offset, limit):
    dataset = pd.read_csv(csvfile, skiprows=lambda idx: idx < offset, nrows=limit)
    total = len(dataset.index)
    if(total  == 0):
        return None
    else:
        return dataset

def merge_data_with_training_data(predictedData, initialDataFile):
    initialData = pd.read_csv(initialDataFile)
    frames = [initialData, predictedData]
    mergeData = pd.concat(frames)
    return mergeData, initialData, predictedData


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

def drop_training_data(X, c):
    #print("Drop training data..")
    X = X.values
    i = 0
    arr = []
    while i < c:
        arr.append(i)
        i = i + 1

    X = np.delete(X, arr, 0)
    return X

def process_data(rows, o):
    c = 0
    mergedata = rows
    predictedData = rows
    initialData = None
    if(o.initialData is not None):
        print("Merge with training data to get the right encoding process..")
        mergedata, initialData, predictedData = merge_data_with_training_data(rows, o.initialData)
        #Get total row of training data
        c = initialData.shape[0]


    print("Splitting X and Y data")
    X, Y = split_X_Y(mergedata, o)

    if(o.lenc is not None):
        print("Label encode")
        X, Y = label_encode(X, Y, o)
        #print(rows.head())

    if(o.denc is not None):
        print("Dummy encode")
        X = pd.get_dummies(X, prefix_sep='_', drop_first=True)

    if(o.initialData is not None):
        X = drop_training_data(X, c)

    if(o.scaler is not None):
        print("StandardScaler")
        sc = StandardScaler(with_mean=False)
        X = sc.fit_transform(X)

    print("Predicting...")
    model = load_model(o.model)
    y_pred = model.predict(X)
    result = pd.DataFrame(y_pred, columns=[o.target])

    #Replace the target column with predicted result
    predictedData = predictedData.drop(o.target, 1)
    predictedData[o.target] = result[o.target]
    return predictedData

def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


def tbl_exists(client, table_ref):
    from google.cloud.exceptions import NotFound
    try:
        client.get_table(table_ref)
        return True
    except NotFound:
        return False

def save_data(data, result, key_path, tablename, newtablename):
    rows = []
    for idx, row in result.iterrows():
        rows.append(row)

    ss = newtablename.split(".")
    project_id = ss[0]
    client = get_client(key_path, project_id)

    ss = tablename.split(".")
    table_id = ss[2] # replace with your table ID
    table_ref = client.dataset(ss[1]).table(table_id)
    table_schema = client.get_table(table_ref).schema

    ss = newtablename.split(".")
    table_id = ss[2] # replace with your table ID
    table_ref = client.dataset(ss[1]).table(table_id)


    #If table does not exist create it first
    if(tbl_exists(client, table_ref) == False):
        print("Create table " + newtablename)
        table = bigquery.Table(newtablename, schema=table_schema)
        table = client.create_table(table)  # Make an API request.

        #table = client.get_table(table_ref)  # API request
        errors = client.insert_rows(table, rows)  # API request
        print(errors)


    table = client.get_table(table_ref)  # API request
    errors = client.insert_rows(table, rows)  # API request
    print(errors)

    print("Saved to " + newtablename)

def save_data_csv(data, result, key_path, tablename, newtablename):
    newtablename = tablename
    result.to_csv(newtablename)
    print("Saved to " + newtablename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Age group prediction')
    required = parser.add_argument_group("Required argument")
    required.add_argument('-f', '--file', type=str, help="configuration filename", required=True)
    args = parser.parse_args()
    main(args)
