import randforest as rf
import import_data as impd
import pdb
from util import int_try, bool_try
from condition_data import condition
from sklearn.tree import export_graphviz
import os

data_loc=str(raw_input('Data location: '))

default_parameters = {
    "criterion":"gini",
    "n_estimators": 20,
    "bootstrap": True
}

default_parameters["n_estimators"] = int_try(raw_input("n_estimators: "))
default_parameters["bootstrap"] = bool_try(raw_input("bootstrap: "))
default_parameters["criterion"] = str(raw_input("criterion: "))

max_depth = int(raw_input("max_depth"))
if max_depth == -1:
    max_depth=None

do_condition=bool(raw_input("Condition?: "))
feature_vectors, classes, column_names = condition(data_loc) if do_condition else impd.build_feature_vectors(data_loc) 

training, testing = rf.split_data(feature_vectors,classes)

clf = rf.create_random_forest(
    num_trees=default_parameters["n_estimators"],
    criterion=default_parameters["criterion"],
    bootstrap=default_parameters["bootstrap"],
    vectors=training["vectors"], 
    classes=training["classes"],
    max_depth=max_depth
)
print(clf)
cont=True
while cont:
    new_action=str(raw_input(">>>"))
    if(new_action=="exit"):
        cont=False
    elif(new_action=="predict"):
        training, testing = rf.split_data(feature_vectors,classes)
        clf.fit(training["vectors"],training["classes"])
        export_graphviz(clf.estimators_[0],
            feature_names=column_names[0:-1],
            filled=True,
            rounded=True)
        os.system('dot -Tpng tree.dot -o tree.png')
        correct_results,total_vectors,results_by_class=rf.predict_data(testing,clf)
        #print report
        print("============")
        print(default_parameters)
        print("Condition: " + str(do_condition))
        print(clf)
        print("results: total vectors - " + str(total_vectors) + " , correct matches - " + str(correct_results) + "(" + str((float(correct_results)/float(total_vectors)) * 100) + ")")
        print(results_by_class)
        print("============")
    elif new_action=="predict_multest":
        attr_in = str(raw_input("estimators list: "))
        vals = attr_in.split(",")
        default_parameters["bootstrap"] = bool(raw_input("bootstrap: "))
        default_parameters["criterion"] = str(raw_input("criterion: "))
        max_depth = int(raw_input("max_depth"))
        if max_depth == -1:
            max_depth=None
        
        for num in vals:
            default_parameters["n_estimators"]=int(num)
            clf = rf.create_random_forest(
                num_trees=default_parameters["n_estimators"],
                criterion=default_parameters["criterion"],
                bootstrap=default_parameters["bootstrap"],
                vectors=training["vectors"], 
                classes=training["classes"],
                max_depth=max_depth
            )
            training, testing = rf.split_data(feature_vectors,classes)
            clf.fit(training["vectors"],training["classes"])

            correct_results,total_vectors,results_by_class=rf.predict_data(testing,clf)
            #print report
            print("============")
            print(default_parameters)
            print("Condition: " + str(do_condition))
            print(clf)
            print("results: total vectors - " + str(total_vectors) + " , correct matches - " + str(correct_results) + "(" + str((float(correct_results)/float(total_vectors)) * 100) + ")")
            print(results_by_class)
            print("============")
    elif new_action == "predict_multlevel":
        attr_in = str(raw_input("max levels list: "))
        step=int(raw_input("step: "))
        vals = attr_in.split(",")
        default_parameters["n_estimators"] = int(raw_input("estimators: "))
        default_parameters["bootstrap"] = bool(raw_input("bootstrap: "))
        default_parameters["criterion"] = str(raw_input("criterion: "))
        
        for num in range(int(vals[0]),int(vals[-1]),step):
            clf = rf.create_random_forest(
                num_trees=default_parameters["n_estimators"],
                criterion=default_parameters["criterion"],
                bootstrap=default_parameters["bootstrap"],
                vectors=training["vectors"], 
                classes=training["classes"],
                max_depth=int(num)
            )
            training, testing = rf.split_data(feature_vectors,classes)
            clf.fit(training["vectors"],training["classes"])

            correct_results,total_vectors,results_by_class=rf.predict_data(testing,clf)
            #print report
            #print(default_parameters)
            #print("results: total vectors - " + str(total_vectors) + " , correct matches - " + str(correct_results) + "(" + str((float(correct_results)/float(total_vectors)) * 100) + ")")
            #print(results_by_class)
            print("Max levels: " + str(num) + ", acc: " + str((float(correct_results)/float(total_vectors)) * 100))
    elif(new_action=="rebuild"):
        default_parameters["n_estimators"] = int_try(raw_input("n_estimators: "))
        default_parameters["bootstrap"] = bool(raw_input("bootstrap: "))
        default_parameters["criterion"] = str(raw_input("criterion: "))
        max_depth = int(raw_input("max_depth"))
        if max_depth == -1:
            max_depth=None
        clf = rf.create_random_forest(
            num_trees=default_parameters["n_estimators"],
            criterion=default_parameters["criterion"],
            bootstrap=default_parameters["bootstrap"],
            vectors=training["vectors"], 
            classes=training["classes"],
            max_depth=max_depth
        )


    