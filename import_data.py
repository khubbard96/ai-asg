import pandas as pd
import pdb

def import_data():
    return 0,0

def build_feature_vectors(location):
    df=import_csv(location)
    vector_length=len(df.values.tolist()[0]) - 1
    X = df.iloc[:,0:vector_length].values.tolist()
    y = df.iloc[:,vector_length].values.tolist()
    return X,y

def build_feature_vector_no_class(location):
    df=import_csv(location)
    vector_length=len(df.values.tolist()[0]) - 1
    X = df.iloc[:,0:vector_length].values.tolist()
    return X



#returns a csv file as a pandas DataFrame
#location is given as relative to the top-level project folder
def import_csv(location):
    return pd.read_csv(location)
