import randforest as rf
import import_data as impd

feature_vectors, classes = impd.build_feature_vectors('positional_results.csv')

clf = rf.create_random_forest(vectors=feature_vectors, classes=classes)
print(clf)
cont=True
while cont:
    new_action=str(raw_input(">>>"))
    if(new_action=="exit"):
        cont=False
    elif(new_action=="predict"):
        new_location=str(raw_input("CSV location:"))
        #class_predictions=rf.predict_csv(new_location)
        print(clf.predict(impd.build_feature_vector_no_class(new_location)))
    elif(new_action=="show"):
        print(clf)