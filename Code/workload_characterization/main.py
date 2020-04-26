
import numpy as np
import pandas as pd
import random
import glob
import os
from fa import FA
from cluster import KMeansClusters, DetK
from sklearn.model_selection import train_test_split
def writeCSV(file_name,data):
    pd_data = pd.DataFrame(data)
    pd_data.to_csv(file_name,sep='\t')
def columnsToPrune(arr_li):
    idx_li = []
    for vals in arr_li:
        val_id = int(vals.split("_")[1])
        idx_li.append(val_id+13)
    return set(idx_li)
def workload_characterization():
    '''path = r'/Users/rachananarayanacharya/MLbasedDBMStuning/Data/'  # use your path
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    train_df, test_df = train_test_split(df, test_size=0.2)'''

    #train_df=pd.read_pickle("../../Data/WorkloadFiles/workload_11.pkl",compression=None)
    train_df3=pd.read_csv("../../Data/CSVFiles/Online_workloadB_preprocessed.csv")
    train_df1=pd.read_csv("../../Data/CSVFiles/Online_workloadC_preprocessed.csv")
    train_df2=pd.read_csv("../../Data/CSVFiles/Offline_workload_preprocessed.csv")
    frames = [train_df1, train_df2, train_df3]

    train_df = pd.concat(frames)

    train_df=train_df.fillna(0)
    model = FA()

    X=train_df.to_numpy()
    print("Shape of the data:",X.shape)
    n_rows, n_cols = X.shape
    model=model._fit(X, 70, 10000)

    components = model.components_.T.copy()
    print("After Components: ",components.shape)
    kmeans_models = KMeansClusters()
    kmeans_models.fit(components, min_cluster=1,
                      max_cluster=min(n_cols - 1, 20),
                      estimator_params={'n_init': 50})
    
    # Compute optimal # clusters, k, using gap statistics
    det = DetK()#create_kselection_model("gap-statistic")
    
    det.fit(components, kmeans_models.cluster_map_)
    print("Optimal no of clusters:",det.optimal_num_clusters_)
    pruned_metrics = kmeans_models.cluster_map_[9].get_closest_samples()
    print("pruned metrics:",pruned_metrics)
    idx_set = columnsToPrune(pruned_metrics)
    cols_rem = []
    for i in range(14,373):
        if i not in idx_set:
            cols_rem.append(i)
    loadWorkLoadRemCols(cols_rem)
    #writeCSV("pruned_metric.csv",pruned_metrics)
def returnFilesInDir(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
             files.append(os.path.join(r, file))
    return files

def loadWorkLoadRemCols(cols_li):
    files = returnFilesInDir("../../Data/WorkloadFiles/")
    #print("FIles:",files)
    final_df = pd.DataFrame()
    for f in files:
         train_df=pd.read_pickle(f)
         #print(train_df)
         df = train_df.drop(train_df.columns[cols_li], axis=1) 
         #print(df)
         spl = f.split("/")
         name = spl[len(spl)-1].split(".")[0]
         df.to_excel("../../Data/PrunedWorkloadFiles/"+name+".xlsx", index=False)
         df.to_pickle("../../Data/PrunedWorkloadFiles/"+name+".pkl")
         final_df = final_df.append(df)
    final_df.to_pickle("../../Data/ConcatPrunedFile.pkl")
    final_df.to_csv("../../Data/ConcatPrunedFile.csv")
workload_characterization()




