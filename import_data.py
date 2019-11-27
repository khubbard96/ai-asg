import pandas as pd

def import_data():
    return 0,0

def build_feature_vectors(location):
    #right now, going to hard-code values for positional results
    df=import_csv(location)
    X = df.iloc[:,2:19].values.tolist()
    y = df.iloc[:,19].values.tolist()
    return X,y

def build_feature_vector_no_class(location):
    df=import_csv(location)
    X = df.iloc[:,2:19].values.tolist()
    return X


#returns a csv file as a pandas DataFrame
#location is given as relative to the top-level project folder
def import_csv(location):
    return pd.read_csv(location)
