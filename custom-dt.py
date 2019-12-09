#RUN USING python

import numpy as np
from math import log
from pdb import set_trace
import pandas as pd
from operator import itemgetter
import random
import queue


class DecisionNode():
    def __init__(self,attr_idx,value):
        self.attr_idx=attr_idx
        self.value=value
        self.children=list()
        self.lt_node=None
        self.gt_node=None

    def add_child(self,child_node):
        self.children.append(child_node)

    def add_nodes(self,lt=None,gt=None):
        if lt != None:
            self.lt_node=lt
        if gt != None:
            self.gt_node=gt
    
    def add_leaf_node(self,leaf):
        self.leaf_node=leaf

    def test_feature_vector(self,vector):
        val=vector[self.attr_idx]
        if val >= self.value:
            return self.gt_node
        else:
            return self.lt_node

class LeafNode():
    def __init__(self,class_name):
        self.class_name=class_name

def get_entropy(classes):
    #this method assumes 2 classes
    total_results=float(len(classes))
    unique_classes=set(classes)
    class_count=list()
    for _class in unique_classes:
        class_count.append(float(sum(1 for i in classes if i==_class)))
    
    entropy=0
    for val in class_count:
        entropy += (val/total_results)*log(val/total_results,2)
    if(entropy != 0.0):
        entropy=-entropy
    return entropy

def get_midpoints(data_set,attr_idx):
    this_midpoints=list()
    values=data_set[:,attr_idx]
    values.sort()
    for i in range(len(values)-1):
        val=float(float(values[i+1])+float(values[i]))/2.0
        if(val < 1):
            val=int(val * 1000) / 1000.0
        this_midpoints.append(val)
    return sorted(set(this_midpoints))

#MUCHO IMPORTANTE
#Here we split the given data set and classes on a given attribute at a given value
def split_on(attr_idx,split_point,data_set,classes):
    greater_than={
        "DATA_SET":list(),
        "CLASSES":list()
    }
    less_than={
        "DATA_SET":list(),
        "CLASSES":list()
    }
    for i in range(len(data_set[:,0])):
        if data_set[i,attr_idx] >= split_point:
            greater_than["DATA_SET"].append(data_set[i,:])
            greater_than["CLASSES"].append(classes[i])
        else:
            less_than["DATA_SET"].append(data_set[i,:])
            less_than["CLASSES"].append(classes[i])
    #reformat the data set 2d list into a numpy 2d table
    greater_than["DATA_SET"]=np.array(greater_than["DATA_SET"])
    less_than["DATA_SET"]=np.array(less_than["DATA_SET"])
    return greater_than,less_than


#this will 
#data set should come in as a 2D numpy array

iterations_run=0
def find_next_split(data_set,classes,branch_root,node_type,available_attrs=None):
    global iterations_run
    #print("itr: " + str(iterations_run))
    num_attrs=len(data_set[0,:])

    if available_attrs==None:
        available_attrs=list()
        for i in range(num_attrs):
            available_attrs.append(i)

    starting_entropy=get_entropy(classes)
    #print(starting_entropy)
    if starting_entropy == 0:
        #print("Adding a leaf node as " + node_type + " to " + str(branch_root))
        if node_type=="GT":
            branch_root.add_nodes(gt=LeafNode(classes[0]))
        else:
            branch_root.add_nodes(lt=LeafNode(classes[0]))
        iterations_run += 1
        return
    all_midpoints={}
    num_examples=len(classes)
    
    #entries for attribute, split point, and resultant information gain
    info_gain_record=list()

    #first want to get all the midpoint data for all of the attributes
    for attr in available_attrs:
        all_midpoints[attr]=get_midpoints(data_set,attr)

    #next, we want to test each attribute and then each splitpoint in that attribute to find out which has the highest info gain
    for attr_idx in available_attrs:
        for split_point in all_midpoints[attr_idx]:
            #now, we have the attribute that we are splitting on and the value we want to split on
            #lets get an information gain
            g_t,l_t=split_on(attr_idx,split_point,data_set,classes)
            #print(split_point)
            '''info_gain_record.append({
                "ATTR_IDX": attr_idx,
                "SPLIT_POINT":split_point,
                "IG":starting_entropy-( ( ( len(g_t["CLASSES"]) / num_examples ) * get_entropy(g_t["CLASSES"])) + ( ( len(l_t["CLASSES"]) / num_examples ) * get_entropy(l_t["CLASSES"]) ) )
            })'''


            #print('greater than weight: ' + str(( float(len(g_t["CLASSES"])) / float(num_examples) )) + ', greater than entropy: ' + str(get_entropy(g_t["CLASSES"])))
            info_gain_record.append([attr_idx,split_point,starting_entropy-( ( ( float(len(g_t["CLASSES"])) / float(num_examples) ) * get_entropy(g_t["CLASSES"])) + ( ( float(len(l_t["CLASSES"])) / float(num_examples) ) * get_entropy(l_t["CLASSES"]) ) )])
            #print("Children entr: " + str((((len(g_t["CLASSES"])/num_examples)*get_entropy(g_t["CLASSES"]))+((len(l_t["CLASSES"])/num_examples)*get_entropy(l_t["CLASSES"])))))
            #print(starting_entropy-(((len(g_t["CLASSES"])/num_examples)*get_entropy(g_t["CLASSES"]))+((len(l_t["CLASSES"])/num_examples)*get_entropy(l_t["CLASSES"]))))
            #b = "Loading " + str(attr_idx) + ", " + str(split_point)
            #print (b+"\r")

    split_crit=sorted(info_gain_record,key=itemgetter(2))
    #print(len(split_crit))
    #if(iterations_run > 900):
        #print(info_gain_record)
    if len(split_crit) > 1:
        split_crit=split_crit[-1]
    else:
        split_crit=split_crit[0]
    #print(split_crit)

    #if the best info gain is 0 but entropy is still not 0
    gt_node,lt_node=None,None
    if split_crit[2] == 0:
        new_attr=0
        for _attr in available_attrs:
            if np.var(data_set[:,_attr]) > 0:
                new_attr=_attr
                break
        num_rows=len(data_set[:,0])
        halfway=(num_rows/2)
        lt_node={
            "DATA_SET":data_set[0:halfway,:],
            "CLASSES":classes[0:halfway]
        }
        gt_node={
            "DATA_SET":data_set[halfway:num_rows,:],
            "CLASSES":classes[halfway:num_rows]
        }
        split_crit[0]=new_attr
        if len(data_set[:,0])==2:
            split_crit[1]=data_set[-1,new_attr]
        else:
            split_crit[1]=data_set[halfway+1,new_attr]
    else:
        gt_node,lt_node=split_on(split_crit[0],split_crit[1],data_set,classes)

    this_d_node=DecisionNode(split_crit[0],split_crit[1])

    #first itr
    if(branch_root==None):
        branch_root=this_d_node

    if this_d_node==None:
        set_trace()

    #print("Iteration: " + str(iterations_run))
    if node_type=='GT':
        #print("Adding GT Node to " + str(branch_root))
        branch_root.add_nodes(lt=None,gt=this_d_node)
    elif node_type=='LT':
        #print("Adding LT Node to " + str(branch_root))
        branch_root.add_nodes(lt=this_d_node,gt=None)
    '''branch_root.add_nodes(lt=lt_node,gt=gt_node)'''

    #get the next nodes
    #for n in [gt_node,lt_node]:
    #    find_next_split(n["DATA_SET"],n["CLASSES"],this_d_node,'')

    iterations_run += 1


    find_next_split(gt_node["DATA_SET"],gt_node["CLASSES"],this_d_node,'GT')
    find_next_split(lt_node["DATA_SET"],lt_node["CLASSES"],this_d_node,'LT')

    return this_d_node


def make_decision_tree(training_data,training_classes,a_attrs):
    tree_head=find_next_split(training_data,training_classes,None,None,a_attrs)
    return tree_head





class RandomForest():
    def __init__(self):
        pass

    def make(self,data,classes,estimators=1):
        self.trees=list()
        num_attrs=len(data[0,:])
        attr_idxs=list()
        for x in range(num_attrs):
            attr_idxs.append(x)
    
        for _ in range(estimators):
            random.shuffle(attr_idxs)
            tree_attrs_avail=attr_idxs[0:random.randint(3,5)]
            idx = np.random.choice(np.arange(len(data)), len(data), replace=True)
            data_sample = data[idx]
            classes_sample = np.array(classes)[idx].tolist()
            self.trees.append(make_decision_tree(data_sample,classes_sample,tree_attrs_avail))

    def predict(self,test_data):
        predicted_result=list()
        for i in range(len(test_data)):
            test_vector=test_data[i]
            result_count={"YES":0,"NO":0}
            for esti in self.trees:
                curr_node=esti
                cont=True
                while(curr_node and cont==True):
                    curr_node=curr_node.test_feature_vector(test_vector)
                    if isinstance(curr_node, LeafNode):
                        cont=False
                result_count[curr_node.class_name] += 1
            if result_count["YES"] > result_count["NO"]:
                predicted_result.append("YES")
            elif result_count["NO"] > result_count["YES"]:
                predicted_result.append("NO")
            else:
                num=random.randint(0,1)
                if num == 0:
                    predicted_result.append("YES")
                else:
                    predicted_result.append("NO")
        return predicted_result




#print(get_entropy(["YES","YES","YES","NO","NO","NO"]))


def import_data(location):
    df=pd.read_csv(location)
    vector_length=len(df.values.tolist()[0]) - 1
    all_v=df.iloc[:,:].values.tolist()
    training_data=list()
    training_classes=list()
    testing_data=list()
    testing_classes=list()
    for i in range(len(all_v)):
        x=random.randint(0,9)
        if(x <= 0):
            testing_data.append(all_v[i][0:-1])
            testing_classes.append(all_v[i][-1])
        else:
            training_data.append(all_v[i][0:-1])
            training_classes.append(all_v[i][-1])

    #X = df.iloc[:,0:vector_length].values.tolist()
    #y = df.iloc[:,vector_length].values.tolist()
    #training,testing=np.array(training,dtype=float),np.array(testing,float)
    
    return {
        "training_data":np.array(training_data,dtype=float),
        "training_classes":training_classes,
        "testing_data":np.array(testing_data,dtype=float),
        "testing_classes":testing_classes
    }


data=import_data(str(raw_input("csv location: ")))
top_root=DecisionNode(-1,-1)
#find_next_split(data["training_data"],data["training_classes"],None,None)

#estimators=list()
rf=RandomForest()
estimators=[1,5,10,20]
for e_val in estimators:
    rf.make(data["training_data"],data["training_classes"],e_val)
    predictions=rf.predict(data["testing_data"])

    correct_results=0   
    for z in range(len(predictions)):
        if predictions[z]==data["testing_classes"][z]:
            correct_results+=1
    print("Result for " + str(e_val) + " estimators: " + str(float(correct_results)/len(predictions)))












