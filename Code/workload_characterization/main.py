
import numpy as np
import pandas as pd
import random
import glob
from fa import FA
from cluster import KMeansClusters, DetK
from sklearn.model_selection import train_test_split

def workload_characterization():
    '''path = r'/Users/rachananarayanacharya/MLbasedDBMStuning/Data/'  # use your path
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    train_df, test_df = train_test_split(df, test_size=0.2)'''
    train_df=pd.read_csv("../PreProcessing/PreProcessedData/Online_workloadB_preprocessed.csv")
    train_df=train_df.fillna(0)
    model = FA()

    X=train_df.to_numpy()
    print(X.shape)
    n_rows, n_cols = X.shape
    model=model._fit(X, 10, 1000)

    components = model.components_.T.copy()
    print("Components: ",components.shape)
    kmeans_models = KMeansClusters()
    kmeans_models.fit(components, min_cluster=1,
                      max_cluster=min(n_cols - 1, 20),
                      estimator_params={'n_init': 50})
    
    # Compute optimal # clusters, k, using gap statistics
    det = DetK()#create_kselection_model("gap-statistic")
    
    det.fit(components, kmeans_models.cluster_map_)
    print("Optimal no of clusters:",det.optimal_num_clusters_)
    
workload_characterization()




