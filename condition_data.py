import pandas as pd
from random import sample
import pdb

def condition(file_loc):
    df=pd.read_csv(file_loc)
    all_vectors=df.values.tolist()
    yes=list()
    no=list()
    for v in all_vectors:
        if v[len(v)-1]=="YES":
            yes.append(v)
        else:
            no.append(v)
            
    if len(yes) > len(no):
        yes=sample(yes,len(no))
    else:
        no=sample(no,len(yes))

    vectors=list()
    classes=list()
    #refigure the data
    for vec in (yes + no):
        vectors.append(vec[0:-1])
        classes.append(vec[-1])

    return vectors,classes,df.columns
