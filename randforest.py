from sklearn.ensemble import RandomForestClassifier

#constants
trees=20



#logic
def create_random_forest(vectors,classes):
    rfc=RandomForestClassifier(n_estimators=trees)
    rfc.fit(vectors,classes)
    return rfc

def predict_csv(file_location):
    return ''