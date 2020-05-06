import os
from os import listdir
from os.path import isfile, join
from gpr_model import GPRNP
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

pruned=[ 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 's1',
       's2', 's3', 's4', 'latency', 'mimic_cpu_util',
       'driver.BlockManager.disk.diskSpaceUsed_MB.avg',
       'driver.jvm.non-heap.used.avg']
model_dict  = defaultdict(dict)
total_models = 0
workload_data = {}
target_data={}
error = []
scores = {}


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_squared_error(Y_true,Y_pred):
    return np.square(np.subtract(Y_true,Y_pred)).mean() 


def loadWorkloadFileNames(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    return onlyfiles

def gprModel(workloadId,colId,X_workload,y_col,X_target):
    global total_models
    if workloadId in model_dict and colId in model_dict[workloadId]:
        #print("Workload: ",workloadId," col id: ",colId)
        #print("Loading the model.............: ",X_target.shape)
        model = model_dict[workloadId][colId]
        
        gpr_result = model.predict(X_target)
        return gpr_result
    model = GPRNP()
    total_models = total_models + 1
    # print(total_models)
    # print(X_scaled)
    model.fit(X_workload, y_col, ridge=1.0)
    #print("workload id:",workloadId,"  col id:",colId)
    model_dict[workloadId][colId] = model
    gpr_result = model.predict(X_target)
    return gpr_result

def MakeMatrix():
    global total_models
    #Get TargetCSV (WorkloadMapping_C)

    X_scaler = StandardScaler(copy=False)
    y_scaler = StandardScaler(copy=False)

    offline_path = "../../Data/New_Workloads/offline/"
    target_path = "../../Data/New_Workloads/WorkloadB/divided/"
    # find all the files in workload
    files = loadWorkloadFileNames(offline_path)
    targetfiles=loadWorkloadFileNames(target_path)
    '''
    for filename in files:
        if filename.endswith(".pkl"):
            workload_mtrx = pd.read_pickle(offline_path + filename)
            workload = workload_mtrx.columns[0]
            cols_list = list(workload_mtrx.columns)
            X_matrix = workload_mtrx[cols_list[1:13]].to_numpy()
            y_matrix = workload_mtrx[cols_list[13:]].to_numpy()
            X_scaler.fit(X_matrix)
            #y_scaler.fit(y_matrix)
    
    for targetworkload in targetfiles:
        if targetworkload.endswith(".pkl") and "5_" in  targetworkload:
            target_mtrx = pd.read_pickle(target_path + targetworkload)
            target_workload = target_mtrx.columns[0]
            t_cols_list = list(target_mtrx.columns)
            X_matrix = target_mtrx[t_cols_list[1:13]].to_numpy()
            y_matrix = target_mtrx[t_cols_list[13:]].to_numpy()
            X_scaler.fit(X_matrix)
            #y_scaler.fit(y_matrix)
    '''
    for targetworkload in targetfiles:
        if targetworkload.endswith(".pkl") and "5_" in  targetworkload:
            #print("File name:",targetworkload)
            target_mtrx = pd.read_pickle(target_path + targetworkload)
            target_workload = target_mtrx.columns[0]
            t_cols_list = list(target_mtrx.columns)
            X_matrix = target_mtrx[t_cols_list[1:13]].to_numpy()
            y_matrix = target_mtrx[t_cols_list[13:]].to_numpy()
            unique_target_workload = np.unique(np.array(target_mtrx[target_workload]))[0]

            X_scaler.fit_transform(X_matrix)

            #y_scaler.transform(y_matrix)
            target_data[unique_target_workload] = {
                'X_matrix': X_matrix,
                'y_matrix': y_matrix,
            }

    for filename in files:
        if filename.endswith(".pkl"):
		    #print("FIlename:",filename)
                    workload_mtrx = pd.read_pickle(offline_path + filename)
                    workload = workload_mtrx.columns[0]
                    cols_list = list(workload_mtrx.columns)
                    X_matrix = workload_mtrx[cols_list[1:13]].to_numpy()
                    y_matrix = workload_mtrx[cols_list[13:]].to_numpy()
                    unique_workload = np.unique(np.array(workload_mtrx[workload]))[0]
                    X_scaler.fit_transform(X_matrix)
                    #y_scaler.transform(y_matrix)
                    workload_data[unique_workload] = {
                    'X_matrix': X_matrix,
                    'y_matrix': y_matrix,
                    }

    return X_scaler,y_scaler

def FindEuclideanDistance():

    for target_workload_id, target_workload_entry in list(target_data.items()):
        #print("NEW TARGET:",target_workload_id)
        X_target=target_workload_entry['X_matrix']
        y_target = target_workload_entry['y_matrix']
        distances={}
        for workload_id, workload_entry in list(workload_data.items()):
                #print("Started new workload matching:",workload_id)
                predictions = np.empty_like(y_target)
                X_workload = workload_entry['X_matrix']
                y_workload = workload_entry['y_matrix']
                for j, y_col in enumerate(y_workload.T):
                   #Predict for each metrics
                                y_col = y_col.reshape(-1, 1)
                                #X_target = X_target.reshape(-1, 1)
                                '''
                                model = GPRNP()
                                model= model.fit(X_workload,y_col)
                                gpr_result=model.predict(X_target)
                                print("X target normal:",X_target.shape)
                                '''
                                gpr_result= gprModel(int(workload_id),int(j),X_workload,y_col,X_target)
                                predictions[:, j] = gpr_result.ypreds.ravel()

                dists = np.mean(np.sqrt(np.sum(np.square(np.subtract(predictions, y_target)), axis=1)))
                # dists=np.linalg.norm(predictions-y_target)
                distances[workload_id]=dists
                scores[target_workload_id]= distances
                #print(scores)
    return scores
# Find the best (minimum) score
def find_best_scores(scores):
    scores_info={}
    for target_workload_id, dist_score in list(scores.items()):
        best_target_score = np.inf
        best_workload_id = None
        for source_workload_id,distance in list(dist_score.items()):
                if distance < best_target_score:
                                    best_target_score = distance
                                    best_workload_id = source_workload_id
        scores_info[target_workload_id]=best_workload_id
        #print("For",target_workload_id," the mean distance and best workload id respectively are: ",best_target_score,",", best_workload_id)
    return scores_info

def AugmentWorkload(best_scores):
    X_augmented={}

    for target_workload_id,source_workload_id in list(best_scores.items()):
        for i,x in enumerate( target_data[target_workload_id]['X_matrix']):

            for j,X in enumerate(workload_data[source_workload_id]['X_matrix']):

                if(np.all(x == X)):


                    workload_data[source_workload_id]['X_matrix']=np.delete(workload_data[source_workload_id]['X_matrix'],j,axis=0)
                    workload_data[source_workload_id]['y_matrix'] = np.delete(workload_data[source_workload_id]['y_matrix'], j,axis=0)

            X_augmented[target_workload_id]=\
                    {'X_matrix': np.append(target_data[target_workload_id]['X_matrix'], workload_data[source_workload_id]["X_matrix"],axis=0),
                    'y_matrix':np.append(target_data[target_workload_id]['y_matrix'],workload_data[source_workload_id]["y_matrix"], axis=0)}

            X_df=pd.DataFrame(data= np.append(X_augmented[target_workload_id]['X_matrix'],X_augmented[target_workload_id]['y_matrix'],axis=1),columns=pruned)
                    # If augmented workload is created for any target workload- use that, otherwise use original target workload
                    #print(X_df.shape)
            X_df.to_pickle("Augmented"+str(target_workload_id)+".pkl")
    return

def latencyPrediction(mapping,X_scaler,y_scaler):
    target_path="../../Data/New_Workloads/Test_workloadB.pkl"
    #Reading and Scaling Target files
    if ".pkl" in target_path:
         target_df=pd.read_pickle(target_path)
    else:
         target_df=pd.read_csv(target_path)
    workload=target_df.columns[0]
    X_columnlabels=target_df.columns[1:13]
    Y_columnlabels=target_df.columns[13:14] #get latency column only

    workload_id_list = np.array(target_df[workload])

    X_target = np.array(target_df[X_columnlabels])
    y_target = np.array(target_df[Y_columnlabels])

    predictions = np.zeros(len(X_target))

    for i in range(0,len(X_target)):
         workload_id =int(workload_id_list[i]) #np.unique(np.array(workload_mtrx[workload]))[0]
         #print("Workload id:",workload_id)  
         closest_workload = mapping[workload_id]
         X_mat = X_target[i].reshape(-1, 1).T
         X_scaler.transform(X_mat)
         gpr_res = gprModel(closest_workload,0,None,None,X_mat)
         predictions[i] = gpr_res.ypreds.ravel()
         print("ravel:",gpr_res.ypreds.ravel())
    print("y preds: ",predictions)
    print("target:",y_target)
    mape = mean_absolute_percentage_error(y_target,predictions)
    mse = mean_squared_error(y_target,predictions)
    print("MAPE:",mape,"  MSE:",mse)


X_scaler,y_scaler = MakeMatrix()
distances=FindEuclideanDistance()
print("Done with Euclidean distance ")
best_scores=find_best_scores(distances)
AugmentWorkload(best_scores)
#mapping={101:57} should be of this format
latencyPrediction(best_scores,X_scaler,y_scaler)

