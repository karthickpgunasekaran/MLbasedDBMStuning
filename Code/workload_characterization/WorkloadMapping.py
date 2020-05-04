import os
from os import listdir
from os.path import isfile, join
from gpr_model import GPRNP
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler


model_dict = {}
total_models = 0
def loadWorkloadFileNames(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    return onlyfiles

def gprModel(workloadId,colId,X_workload,y_col,X_target):
    global total_models
    if workloadId in model_dict and colId in model_dict[workloadId]:
         print("Loading the model.............")
         model = model_dict[workloadId][colId]
         gpr_result = model.predict(X_target)
         return gpr_result 
    model = GPRNP()
    total_models= total_models+1
    #print(total_models)
    # print(X_scaled)
    model.fit(X_workload, y_col, ridge=1.0)
    model_dict[workloadId][colId] = model
    gpr_result = model.predict(X_target)    
    return gpr_result

def WorkloadMapping():
    global total_models
    #Get TargetCSV (WorkloadMapping_C)
    target_path=r"../../Data/Augmented_Pruned_target.csv"
    #Reading and Scaling Target files
    target_df=pd.read_csv(target_path)
    workload=target_df.columns[0]
    X_columnlabels=target_df.columns[1:13]
    Y_columnlabels=target_df.columns[13:]
    X_target = np.array(target_df[X_columnlabels])
    y_target = np.array(target_df[Y_columnlabels])

    # # X_scaler = StandardScaler(copy=False)
    # # y_scaler = StandardScaler(copy=False)
    # X_target = X_scaler.fit_transform(X_target)
    # y_target = y_scaler.fit_transform(y_target)
    error = []
    workload_data = {}
    scores = {}
    pruned_file = r"../../Data/PrunedWorkloadFiles/"
    # find all the files in workload
    files = loadWorkloadFileNames(pruned_file)

    for filename in files:
        if filename.endswith(".pkl"):
            workload_mtrx = pd.read_pickle(pruned_file + filename)
            #  print(filename,workload_mtrx.shape)
            cols_list = list(workload_mtrx.columns)
            X_matrix = workload_mtrx[cols_list[1:13]].to_numpy()
            y_matrix = workload_mtrx[cols_list[13:]].to_numpy()
            unique_workload = np.unique(np.array(workload_mtrx[workload]))[0]
            # print(unique_workload)
            #             X_scaler.fit(X_matrix)
            #             y_scaler.fit_transform(y_matrix)
            # print(X_scaler)
            workload_data[unique_workload] = {
                'X_matrix': X_matrix,
                'y_matrix': y_matrix,
            }

            for workload_id, workload_entry in list(workload_data.items()):
                predictions = np.empty_like(y_target)
                X_workload = workload_entry['X_matrix']

                # X_scaled = X_scaler.transform(X_workload)
                y_workload = workload_entry['y_matrix']
                #  y_scaled = y_scaler.transform(y_workload)

                for j, y_col in enumerate(y_workload.T):
                    # Using this workload's data, train a Gaussian process model
                    # and then predict the performance of each metric for each of
                    # the knob configurations attempted so far by the target.
                    try:
                        #  print(y_col.shape,X_workload.shape)
                        y_col = y_col.reshape(-1, 1)

                        #model = GPRNP()
                        # print(X_scaled)
                        #model.fit(X_workload, y_col, ridge=1.0)
                        gpr_result = gprModel(workload_id,j,X_workload,y_col,X_target)
                        predictions[:, j] = gpr_result.ypreds.ravel()
                    except:
                        error.append(workload_id)

            # compute the score (i.e., distance) between the target workload
            # and each of the known workloads
            dists = np.sqrt(np.sum(np.square(np.subtract(predictions, y_target)), axis=1))
            scores[workload_id] = np.mean(dists)
            #print("score is:", scores)
            print("Total models:",total_models)

    # Find the best (minimum) score
    best_score = np.inf
    best_workload_id = None
    scores_info = {}
    for workload_id, similarity_score in list(scores.items()):

        if similarity_score < best_score:
            best_score = similarity_score
            best_workload_id = workload_id
    print(best_score, best_workload_id)

WorkloadMapping()



