from sklearn.ensemble import RandomForestClassifier
import random
import import_data as impd
import pdb

#logic
def create_random_forest(vectors,classes,num_trees=20):
    rfc=RandomForestClassifier(n_estimators=num_trees)
    rfc.fit(vectors,classes)
    return rfc

def predict_data(testing_data,clf):
    fvs=testing_data["vectors"]
    class_results=testing_data["classes"]
    predictions=clf.predict(fvs)
    total_vectors=len(fvs)
    idx=0
    correct_results=0
    for _ in predictions:
        if(class_results[idx]==predictions[idx]):
            correct_results += 1
        idx += 1
    return correct_results,total_vectors

def split_data(feature_vectors,classes):
    percent_split = 1
    training_vectors = list()
    training_classes=list()

    testing_vectors = list()
    testing_classes = list()
    idx=0
    for _ in feature_vectors:
        z=random.randint(1,10)
        if z <= percent_split:
            testing_vectors.append(feature_vectors[idx])
            testing_classes.append(classes[idx])
        else:
            training_vectors.append(feature_vectors[idx])
            training_classes.append(classes[idx])
        idx += 1
    return {"vectors":training_vectors,"classes":training_classes}, {"vectors":testing_vectors,"classes":testing_classes}