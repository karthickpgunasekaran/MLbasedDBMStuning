
import numpy as np
import pandas as pd
import random
import glob
from Code.workload_characterization.fa import FA
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
    train_df=pd.read_csv("/Users/rachananarayanacharya/MLbasedDBMStuning/Data/Online_workloadB_preprocessed.csv")
    train_df=train_df.fillna(0)
    model = FA()

    X=train_df.to_numpy()
    print(X.shape)
    model=model._fit(X, 200, 1000)

workload_characterization()