 
import os
import shutil
from os import path
from os import listdir
from os.path import isfile, join
from gpr_model import GPRNP
import numpy as np
import pandas as pd
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel,RBF

from Code.workload_characterization.RF import RF
from Code.workload_characterization.nn import NN

pruned=[ 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 's1',
       's2', 's3', 's4', 'latency', 'mimic_cpu_util',
       'driver.BlockManager.disk.diskSpaceUsed_MB.avg',
       'driver.jvm.non-heap.used.avg']

model_dict  = defaultdict(dict)
total_models = 0
X_scaler = StandardScaler()
y_scaler = StandardScaler()
workload_data = {}
target_data={}
error = []
scores = {}


def mean_absolute_percentage_error(y_true, y_pred):
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    #print("after : ",(y_true - y_pred))
    print("preds:",y_pred)
    print("true:",y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_squared_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.square(np.subtract(y_true,y_pred)).mean()


def loadWorkloadFileNames(folder):
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    return onlyfiles

def get_prediction_model(model_type):
    if(model_type=="gpr"):
        return GPRNP(length_scale=1.0, magnitude=0.8)
    elif(model_type=="NN"):
        return NN()
    else:
        return RF(50,200,2)



def gprModel(workloadId,colId,X_workload,y_col,X_target):
    global total_models

    if workloadId in model_dict and colId in model_dict[workloadId]:
        #print("Loading the model.............")
        model = model_dict[workloadId][colId]
        gpr_result = model.predict(X_target)
        return gpr_result
    #print("Creating the model.....")
    model_type="nn"
    model = get_prediction_model(model_type)
    total_models = total_models + 1
    # print(total_models)
    # print(X_scaled)
    model.fit(X_workload, y_col) # em - 0.01
    #print("workload id:",workloadId,"  col id:",colId)
    model_dict[workloadId][colId] = model
    print("--------------------------Model: ", model)
    gpr_result = model.predict(X_target)
    return gpr_result

def gprModel_scipy(workloadId,colId,X_workload,y_col,X_target):
    global total_models

    if workloadId in model_dict and colId in model_dict[workloadId]:
        #print("Loading the model.............")

        model = model_dict[workloadId][colId]
        gpr_result = model.predict(X_target) #model.predict(X_target)
        return gpr_result
    #print("Creating the model.....")
    #kernel = DotProduct() + WhiteKernel()
    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    #print("IN")
    model = GaussianProcessRegressor(alpha= 0.1).fit(X_workload,y_col)

    #model = GPRNP( length_scale=1.0, magnitude=0.8)
    total_models = total_models + 1
    # print(total_models)
    # print(X_scaled)
    #model.fit(X_workload, y_col, ridge=17.5)
    #print("workload id:",workloadId,"  col id:",colId)
    model_dict[workloadId][colId] = model
    gpr_result = model.predict(X_target)
    return gpr_result

def MakeMatrix(offline_path,target_path):
    global total_models
    global workload_data
    global target_data
    global X_scaler
    workload_data = {}
    target_data={}

    y_scaler = StandardScaler(copy=False)

    # find all the files in workload
    files = loadWorkloadFileNames(offline_path)
    targetfiles=loadWorkloadFileNames(target_path)

    for targetworkload in targetfiles:
        if targetworkload.endswith(".pkl"):
            #print("File name:",targetworkload)
            target_mtrx = pd.read_pickle(target_path + targetworkload)
            target_workload = target_mtrx.columns[0]
            t_cols_list = list(target_mtrx.columns)
            X_matrix = target_mtrx[t_cols_list[1:13]].to_numpy()
            y_matrix = target_mtrx[t_cols_list[13:]].to_numpy()
            unique_target_workload = np.array(target_mtrx[target_workload])[0]
            #print("Before:",X_matrix)
            #X_matrix= X_matrix/X_matrix.max(axis=0)
            X_matrix=X_scaler.fit_transform(X_matrix)
            #print("After:",X_matrix)
            #print("out:",out)
      
            y_matrix[:,1:] =  y_scaler.fit_transform(y_matrix[:,1:])
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
                    unique_workload = np.array(workload_mtrx[workload])[0]
                    X_matrix = X_scaler.fit_transform(X_matrix)
                    #print("y mat:",y_matrix.shape)
                    y_matrix[:,1:] =  y_scaler.fit_transform(y_matrix[:,1:])
                    #X_matrix= X_matrix/X_matrix.max(axis=0)
         
                    workload_data[unique_workload] = {
                    'X_matrix': X_matrix,
                    'y_matrix': y_matrix,
                    }
     
    #print("target data:",target_data,"  workload_data: ",workload_data) 

def TrainGPRwithWorkloadMapping(latency_preds=False):
    global scores
    scores = {}
    for target_workload_id, target_workload_entry in list(target_data.items()):
        #print("IN")
        X_target=target_workload_entry['X_matrix']
        y_target = target_workload_entry['y_matrix']
        distances={}
        
        for workload_id, workload_entry in list(workload_data.items()):
                predictions = np.zeros_like(y_target)
                X_workload = workload_entry['X_matrix']
                y_workload = workload_entry['y_matrix']
                
                for j, y_col in enumerate(y_workload.T):
                   #Predict for each metrics
                                y_col = y_col.reshape(-1, 1)
                                '''
                                model = GPRNP()
                                model= model.fit(X_workload,y_col)
                                gpr_result=model.predict(X_target)
                                '''
                                

                                gpr_result= gprModel(int(workload_id),int(j),X_workload,y_col,X_target)
                                #print("gpr_res",gpr_result)
                                #save = gpr_result[:][:]
                                #print(save)
                                #print("worklaod id:",workload_id," y_col:",j," y_col:",y_col.shape," preds:",gpr_result.ypreds.shape)
                                #print("save:",save.shape)
                                #print("gpr_result.ypreds",gpr_result.ypreds.shape)
                                predictions[:, j] = gpr_result.ypreds[:][0]#.ravel()#gpr_result[:][0]#
                #print("y_target:",y_target)
                #y_target[y_target==0] = 0.000000000007
                #dists = mean_absolute_percentage_error(y_target,predictions)
                #print("dists:",dists)
                #print(predictions)
                dists = np.square(np.subtract(y_target,predictions)).mean()  #np.mean(np.sqrt(np.sum(np.square(np.subtract(predictions, y_target)), axis=1)))
                # dists=np.linalg.norm(predictions-y_target)
                distances[workload_id]=dists
        scores[target_workload_id]= distances
        #print(scores)
    return scores


# Find the best (matched) workload
def find_right_mapping(scores,path):
    scores_info={}
    #print("SCORES:",scores)
    #return
    for target_workload_id, dist_score in list(scores.items()):
        #print()
        best_target_score = np.inf
        best_workload_id = None
        for source_workload_id,distance in list(dist_score.items()):
                if distance < best_target_score:
                                    #print("value:",distance)
                                    best_target_score = distance
                                    best_workload_id = source_workload_id
        scores_info[int(target_workload_id)]=int(best_workload_id)
        #print("For",target_workload_id," the mean distance and best workload id respectively are: ",best_target_score,",", best_workload_id)
    #pickle_out = open(path+"mapping.pickle", "wb")
    #pickle.dump(scores_info, pickle_out)
    #pickle_out.close()
    return scores_info
def MergeWorkloads(mapping,source,target,destination):
    for tar_wk in mapping:
          tar_name = "5pointsworkload_"+str(tar_wk)+".pkl"
          if not path.exists(target+tar_name):
                tar_name = "workload_"+str(tar_wk)+".pkl"
          sor_name = "workload_"+str(mapping[tar_wk])+".pkl"
          tar_df = pd.read_pickle(target+tar_name)
          sor_df = pd.read_pickle(source+sor_name)
          frames = [sor_df,tar_df]
          result = pd.concat(frames)
          result.to_pickle(destination+sor_name)
          result.to_excel(destination+sor_name[:-4]+".xlsx",index=False,header=True)

    for f in loadWorkloadFileNames(source):
          if not path.exists(destination+f):
                shutil.copy(source+f, destination+f)

def AugmentWorkload(best_scores,path_to_save):
    X_augmented={}
    error = []
    for target_workload_id,source_workload_id in list(best_scores.items()):
        #print("Initial shape of X:", workload_data[source_workload_id]['X_matrix'].shape)
        for i,x in enumerate( target_data[target_workload_id]['X_matrix']):

            for j,X in enumerate(workload_data[source_workload_id]['X_matrix']):

                try:
                    if(np.all(x == X)):


                        workload_data[source_workload_id]['X_matrix']=np.delete(workload_data[source_workload_id]['X_matrix'],j,axis=0)
                        workload_data[source_workload_id]['y_matrix'] = np.delete(workload_data[source_workload_id]['y_matrix'], j,axis=0)

                except:
                    error.append(source_workload_id)
        X_augmented[target_workload_id] = \
            {'X_matrix': np.append(target_data[target_workload_id]['X_matrix'],
                                   workload_data[source_workload_id]["X_matrix"], axis=0),
             'y_matrix': np.append(target_data[target_workload_id]['y_matrix'],
                                   workload_data[source_workload_id]["y_matrix"], axis=0)}
        workload_data[source_workload_id]={
            'X_matrix': X_augmented[target_workload_id]['X_matrix'],
            'y_matrix': X_augmented[target_workload_id]['y_matrix']

        }
        '''
        print(source_workload_id)
        print("Final shape of X:", X_augmented[target_workload_id]['X_matrix'].shape)
        print("Final shape of y :", X_augmented[target_workload_id]['y_matrix'].shape)
        print("Final shape overall :",np.append(X_augmented[target_workload_id]['X_matrix'],X_augmented[target_workload_id]['y_matrix'],axis=1).shape)
        '''
        X_df=pd.DataFrame(data= np.append(X_augmented[target_workload_id]['X_matrix'],X_augmented[target_workload_id]['y_matrix'],axis=1),columns=pruned)
        X_df.insert(loc=0, column='workload id', value=source_workload_id)
        #print(X_df.shape)
        X_df.to_pickle(path_to_save+"Augmented_" + str(source_workload_id) + ".pkl")
        X_df.to_csv(path_to_save+"Augmented_" + str(source_workload_id) + ".csv",index=False,header=True)

    return

def latencyPrediction(mapping,path,plot_val=False,save_output=False):
    target_path=path
    target_df1 = pd.DataFrame()
    #Reading and Scaling Target files
    if ".pkl" in target_path:
        target_df = pd.read_pickle(target_path)
    else:
        target_df = pd.read_csv(target_path)
    workload = target_df.columns[0] 

    X_columnlabels = target_df.columns[1:13]

    target_df1[workload] = target_df[workload].astype(str).str.replace('-', '')

    workload_id_list = np.array(target_df1[workload])
    #print("all workloads:",workload_id_list)
    X_target = np.array(target_df[X_columnlabels])


    predictions = np.zeros(len(X_target))

    X_target = X_scaler.transform(X_target)

    for i in range(0, len(X_target)):
        workload_id = int(float(workload_id_list[i]))  # np.unique(np.array(workload_mtrx[workload]))[0]
        # print("Workload id:",workload_id)
        closest_workload = mapping[workload_id]
        X_mat = X_target[i].reshape(-1, 1).T
        

 
  
        gpr_result = gprModel(closest_workload, 0, None, None, X_mat)
        predictions[i] = gpr_result.ypreds.ravel() 
   
    #print("y preds: ",predictions)
    if save_output==False:
	    Y_columnlabels = target_df.columns[13:14]  # get latency column only
	    y_target = np.array(target_df[Y_columnlabels]).ravel()
	    #print("target:",y_target.shape," predsd:",predictions.shape)
	    mape = mean_absolute_percentage_error(y_target, predictions)
	    mse = mean_squared_error(y_target, predictions)
	    print("MAPE:", mape, "  MSE:", mse)

	    if plot_val:
	          plot_lat_pred(predictions,y_target)

    else:
            target_df["latency prediction"] = list(predictions) 
            target_df.to_excel("../../Data/Test_Submission.xlsx",index=False,header=True)

def plot_lat_pred(predictions,y_target):
	plt.rcParams['legend.numpoints'] = 1
	#generate some random data
	pred =predictions  #np.random.rand(10)*70
	GT = y_target #pred+(np.random.randint(8,40,size= len(pred))*2.*(np.random.randint(2,size=len(pred))-.5 ))
	fig, ax = plt.subplots(figsize=(6,4))
	# plot a black line between the 
	# ith prediction and the ith ground truth 
	for i in range(len(pred)):
		ax.plot([i,i],[pred[i], GT[i]], c="k", linewidth=0.5)
	ax.plot(pred,'o', label='Prediction', color = 'g')
	ax.plot(GT,'^', label='Ground Truth', color = 'r' )
	ax.set_xlim((-1,len(pred)))
	plt.xlabel('Workloads')
	plt.ylabel('Latency')
	#plt.title('test')
	plt.legend()             
	plt.show()

def main():
    global model_dict
    global X_scaler
    ##Define all the paths
    offline_raw_path = "../../Data/New_Workloads/offline/raw/"
    offline_B_path = "../../Data/New_Workloads/offline/AugmentedwithB/"
    offline_C_path = "../../Data/New_Workloads/offline/AugmentedwithC/"


    mapping_path_B = "../../Data/New_Workloads/WorkloadB/divided/"

    #target_path_B =  "../../Data/New_Workloads/WorkloadB/divided/predict/"

    mapping_path_C = "../../Data/New_Workloads/WorkloadC/"


    #Augmented_path_offline="offline/AugmentedwithC\"
    Mapping_path="../../Data/New_Workloads/"

    #Part-1
    #Make source and target matrix
    MakeMatrix(offline_raw_path,mapping_path_B)

    #Train GPR with source matrix
    #TrainGPR(workload_data)  - dont understand why this is there
    #Repeat workload mapping again if no mapping is available
    print("workload mapping....")
    distances=TrainGPRwithWorkloadMapping(latency_preds=True)
    #print("dist:",distances)
    print("right match....")
    mapping=find_right_mapping(distances,Mapping_path)
    #AugmentWorkload(mapping, offline_B_path)
    print("mergeworkloads..")
    MergeWorkloads(mapping,offline_raw_path,mapping_path_B,offline_B_path)
    #Load previous mapping if available and predict
    #Fired after we have mapping for workload B and workload C, make sure to change mapping path


    #Part-2 
    #Make source and target matrix again for retraining gpr
    MakeMatrix(offline_B_path,mapping_path_B)
    #reset all the models
    model_dict  = defaultdict(dict)
    #Re-Train GPR with updated source matrix
    print("workload mapping....")
    distances=TrainGPRwithWorkloadMapping(latency_preds=True)
    print("latency prediction....")
    #print("map ;",sorted (mapping.keys()))
    latencyPrediction(mapping,"../../Data/New_Workloads/Test_workloadB.pkl",plot_val=True)
    
   
    
    #Part-3
    #Make source and target matrix again for workload mapping C
    MakeMatrix(offline_B_path,mapping_path_C)
    print("workload mapping....")
    #reset all the models
    model_dict  = defaultdict(dict)
    #Re-Train GPR with updated source matrix
    distances=TrainGPRwithWorkloadMapping()
    #print("dist:",distances)
    print("latency prediction....")
    mapping=find_right_mapping(distances,Mapping_path)
    #print("map ;",sorted (mapping.keys()))
    MergeWorkloads(mapping,offline_B_path,mapping_path_C,offline_C_path) #(mapping, offline_C_path)
  

    #Part-4
    #Make source and target matrix again for retraining gpr
    MakeMatrix(offline_C_path,mapping_path_C)
    #reset all the models
    model_dict  = defaultdict(dict)
    #Re-Train GPR with updated source matrix
    distances=TrainGPRwithWorkloadMapping()
    mapping=find_right_mapping(distances,Mapping_path)
    #do latency prediction
    latencyPrediction(mapping,"../../Data/test.csv",save_output=True)

  

    '''
    try:
        pickle_in = open(Mapping_path + "mapping.pickle", "rb")
        mapping = pickle.load(pickle_in)
        latencyPrediction(mapping)
        exit(0)
    except IOError:
    #Repeat workload mapping again if no mapping is available
        distances=FindEuclideanDistance()
        best_scores=find_best_scores(distances,Mapping_path)
        # print(best_scores)
        AugmentWorkload(best_scores, Augmented_path)
        exit(0)
    '''



if __name__ == "__main__":
    main()
