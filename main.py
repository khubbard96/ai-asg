import randforest as rf
import import_data as impd
import pdb

data_loc=str(raw_input('Data location: '))
num_trees=int(raw_input('Num trees: '))

feature_vectors, classes = impd.build_feature_vectors(data_loc)

training, testing = rf.split_data(feature_vectors,classes)

clf = rf.create_random_forest(num_trees=num_trees,vectors=training["vectors"], classes=training["classes"])
print(clf)
cont=True
while cont:
    new_action=str(raw_input(">>>"))
    if(new_action=="exit"):
        cont=False
    elif(new_action=="predict"):
        correct_results,total_vectors=rf.predict_data(testing,clf)
        print("Num trees: " + str(num_trees))
        print("results: total vectors - " + str(total_vectors) + " , correct matches - " + str(correct_results) + "(" + str((float(correct_results)/float(total_vectors)) * 100) + ")")
    elif(new_action=="rebuild"):
        num_trees=int(raw_input('Num trees: '))
        clf = rf.create_random_forest(num_trees=num_trees,vectors=feature_vectors, classes=classes)

    